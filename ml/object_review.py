"""Durable, versioned human boxes over existing native-resolution train frames."""
import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from autolabel.object_frames import CLASSES, validate_decision
from autolabel.dataset import snapshot


class ObjectConflict(ValueError):
    status_code = 409


class ObjectReview:
    def __init__(self, root):
        self.root = Path(root)
        self.source = self.root/'data/autolabel/objects_v1'
        self.directory = self.root/'data/human_review'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory/'objects.sqlite'
        self._base_rows = {r['image_id']: r for r in map(json.loads, (self.source/'object_review_manifest.jsonl').read_text().splitlines())}
        self.protected = json.loads((self.source/'summary.json').read_text())['protected_inputs']
        self.source_hashes={name:hashlib.sha256((self.source/name).read_bytes()).hexdigest()
                            for name in ('summary.json','object_review_manifest.jsonl')}
        with (self.root/'data/training/manifest.csv').open() as stream:
            self.training = {(r['video'], r['group_id'], r['sha256']) for r in csv.DictReader(stream) if r['split']=='train'}
        for row in self._base_rows.values():
            if row['split']!='train' or (row['video'], row['group_id'], row['source_sha256']) not in self.training:
                raise ValueError('Object frame is outside the frozen train set')
            path=(self.source/row['image']).resolve()
            if not path.is_relative_to((self.source/'images').resolve()):
                raise ValueError('Invalid object image path')
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS decisions (image_id TEXT PRIMARY KEY, version INTEGER NOT NULL, payload TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, image_id TEXT NOT NULL, version INTEGER NOT NULL, payload TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS frames (image_id TEXT PRIMARY KEY, payload TEXT NOT NULL)')

    def connect(self):
        return sqlite3.connect(self.database, timeout=10)

    @property
    def rows(self):
        with self.connect() as db:
            return dict(self._base_rows,**{key:json.loads(raw) for key,raw in db.execute('SELECT image_id,payload FROM frames ORDER BY image_id')})

    def image_path(self,row):
        base=self.directory/'frames' if row.get('origin')=='selected_video_frame' else self.source/'images'
        path=(self.directory/row['image']) if row.get('origin')=='selected_video_frame' else (self.source/row['image'])
        if not path.resolve().is_relative_to(base.resolve()) or path.is_symlink():raise ValueError('Invalid image path')
        return path

    def register_frame(self,metadata):
        """Register a server-extracted frame; never creates a human annotation."""
        from PIL import Image
        self.check_sources()
        identity=tuple(metadata.get(k) for k in ('video','group_id','source_sha256'))
        if identity not in self.training or metadata.get('split')!='train':raise ValueError('Кадр не относится к train')
        timestamp=metadata.get('timestamp_sec')
        if isinstance(timestamp,bool) or not isinstance(timestamp,(int,float)) or not math.isfinite(timestamp) or timestamp<0:
            raise ValueError('Некорректное время кадра')
        path=Path(metadata['path']).resolve()
        if not path.is_relative_to(self.directory.resolve()):raise ValueError('Only server-extracted frames may be registered')
        raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        if sha!=metadata.get('image_sha256'):raise ObjectConflict('Извлечённый кадр изменился')
        with Image.open(path) as image:
            width,height=image.size
            if image.format!='JPEG' or (width,height)!=(metadata.get('width'),metadata.get('height')):
                raise ValueError('Некорректный формат кадра')
        existing=next((r for r in self.rows.values() if r['source_sha256']==identity[2]
                       and abs(r['timestamp']-timestamp)<1e-6 and r['image_sha256']==sha),None)
        if existing is not None:
            return dict(image_id=existing['image_id'],object_url='/label/objects?image_id='+existing['image_id'],frame=existing)
        image_id='human_'+hashlib.sha256(f'{identity[2]}:{timestamp:.6f}:{sha}'.encode()).hexdigest()[:24]
        frames=self.directory/'frames';frames.mkdir(exist_ok=True)
        destination=frames/(image_id+'.jpg')
        if not destination.exists():
            temporary=destination.with_suffix('.'+uuid.uuid4().hex+'.tmp')
            temporary.write_bytes(raw);os.replace(temporary,destination)
        row=dict(image_id=image_id,image='frames/'+destination.name,image_sha256=sha,video=identity[0],group_id=identity[1],
                 source_sha256=identity[2],split='train',timestamp=float(timestamp),width=width,height=height,
                 origin='selected_video_frame',requested_timestamp_sec=metadata.get('requested_timestamp_sec',timestamp))
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO frames VALUES(?,?)',(image_id,json.dumps(row,ensure_ascii=False)))
        return dict(image_id=image_id,object_url='/label/objects?image_id='+image_id,frame=row)

    def check_sources(self):
        if snapshot(self.root)!=self.protected or any(hashlib.sha256((self.source/name).read_bytes()).hexdigest()!=sha
                                                       for name,sha in self.source_hashes.items()):
            raise ObjectConflict('Исходная разметка изменилась. Обновите очередь перед сохранением.')

    def decisions(self):
        with self.connect() as db:
            decisions={key: json.loads(payload) for key,payload in db.execute('SELECT image_id,payload FROM decisions')}
        for key,decision in decisions.items():
            row=self.rows.get(key)
            decision['stale']=row is None or decision.get('image_sha256')!=row['image_sha256']
            if decision['stale']:decision['eligible_for_detector_training']=False
        return decisions

    def catalog(self):
        self.check_sources()
        decisions=self.decisions()
        return dict(rows=[dict(r,image=('/api/human-objects/images/'+r['image_id']+'.jpg') if r.get('origin')=='selected_video_frame' else '/autolabel/objects_v1/'+r['image']) for r in self.rows.values()],
                    classes=CLASSES,decisions=decisions,counts=self.counts(decisions))

    @staticmethod
    def counts(decisions):
        accepted=[d for d in decisions.values() if d['eligible_for_detector_training']]
        return dict(reviewed=len(decisions),training_frames=len(accepted),
                    human_boxes=sum(len(d['boxes']) for d in accepted),
                    confirmed_negative_frames=sum(d['image_assessment']=='confirmed_no_target' for d in accepted))

    def save(self, payload):
        self.check_sources()
        if not isinstance(payload,dict) or payload.get('image_id') not in self.rows:
            raise ValueError('Неизвестный кадр')
        row=self.rows[payload['image_id']]
        version=payload.get('expected_version')
        if type(version) is not int or version<0:raise ValueError('Нужна версия решения')
        if not isinstance(payload.get('reviewer'),str) or not 1<=len(payload['reviewer'].strip())<=80:
            raise ValueError('Укажите имя проверяющего до 80 символов')
        if not isinstance(payload.get('note',''),str) or len(payload.get('note',''))>2000:
            raise ValueError('Примечание слишком длинное')
        if not isinstance(payload.get('boxes'),list) or len(payload['boxes'])>100:
            raise ValueError('Допускается до 100 рамок на кадр')
        if any(not isinstance(b,dict) for b in payload['boxes']):raise ValueError('Некорректная рамка')
        record={k:payload.get(k) for k in ('image_id','image_sha256','reviewer','review_status','image_assessment','boxes')}
        record.update(note=payload.get('note',''),reviewer=payload['reviewer'].strip())
        eligible=validate_decision(record,row)
        path=self.image_path(row)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['image_sha256']:
            raise ObjectConflict('Файл кадра изменился; решение не сохранено')
        record.update(version=version+1,reviewed_at=datetime.now(timezone.utc).isoformat(),
                      eligible_for_detector_training=eligible,annotation_origin='explicit_human_decision')
        raw=json.dumps(record,ensure_ascii=False,allow_nan=False)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT version FROM decisions WHERE image_id=?',(row['image_id'],)).fetchone()
            if (old[0] if old else 0)!=version:
                raise ObjectConflict('Другой проверяющий уже изменил этот кадр. Обновите страницу; ваш вариант не перезаписал его решение.')
            db.execute('INSERT INTO decisions VALUES(?,?,?) ON CONFLICT(image_id) DO UPDATE SET version=excluded.version,payload=excluded.payload',
                       (row['image_id'],version+1,raw))
            db.execute('INSERT INTO history(image_id,version,payload) VALUES(?,?,?)',(row['image_id'],version+1,raw))
        return dict(decision=record,counts=self.counts(self.decisions()))

    def export_jsonl(self):
        return ''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in self.decisions().values())

    def export_dataset(self):
        """Snapshot only explicitly complete, certain human decisions to COCO/YOLO."""
        self.check_sources()
        decisions=self.decisions()
        accepted=[d for d in decisions.values() if d['eligible_for_detector_training']]
        export_id='objects_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
        final=self.directory/'exports'/export_id
        out=final.with_name('.'+export_id+'.partial')
        out.mkdir(parents=True,exist_ok=False);(out/'images').mkdir();(out/'labels').mkdir()
        images=[];annotations=[];manifest=[]
        for index,d in enumerate(accepted,1):
            source=self.rows[d['image_id']];original=self.image_path(source)
            if hashlib.sha256(original.read_bytes()).hexdigest()!=source['image_sha256']:
                raise ObjectConflict('Исходный кадр изменился; экспорт не завершён')
            name=d['image_id']+'.jpg';shutil.copy2(original,out/'images'/name)
            images.append(dict(id=index,file_name='images/'+name,width=source['width'],height=source['height']))
            lines=[]
            for b in d['boxes']:
                x1,y1,x2,y2=b['xyxy'];category=CLASSES.index(b['class'])
                annotations.append(dict(id=len(annotations)+1,image_id=index,category_id=category+1,bbox=[x1,y1,x2-x1,y2-y1],area=(x2-x1)*(y2-y1),iscrowd=0))
                lines.append(f'{category} {(x1+x2)/2/source["width"]:.8f} {(y1+y2)/2/source["height"]:.8f} {(x2-x1)/source["width"]:.8f} {(y2-y1)/source["height"]:.8f}')
            (out/'labels'/(d['image_id']+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''))
            manifest.append(dict(source,human_decision=d,eligible_for_detector_training=True,annotation_origin='human'))
        self.check_sources()
        data=dict(images=images,annotations=annotations,categories=[dict(id=i+1,name=c) for i,c in enumerate(CLASSES)])
        (out/'coco_train.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        (out/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifest))
        (out/'dataset.yaml').write_text('path: .\ntrain: images\n# No independent validation set is supplied.\nnames:\n'+''.join(f'  {i}: {name}\n' for i,name in enumerate(CLASSES)))
        summary=dict(export_id=export_id,training_frames=len(images),human_boxes=len(annotations),
                     independent_validation_set=False,protected_inputs=self.protected,
                     note='Only complete human-reviewed frames; uncertain frames excluded, not negative examples.')
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        # Publish the archive atomically with its immutable source snapshot.
        with zipfile.ZipFile(out/'dataset.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(out.rglob('*')):
                if file.is_file() and file.name!='dataset.zip':
                    archive.write(file,file.relative_to(out))
        out.rename(final)
        return dict(summary,path=str(final.relative_to(self.root)))
