"""Read-only human snapshot; full frames, crops and local photometric synthetic variants."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import sqlite3

from PIL import Image,ImageEnhance,ImageFilter
from ml.autolabel.object_frames import CLASSES,validate_decision

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def overlap(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))

def prepare(root,out):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Use a new snapshot')
    out.mkdir(parents=True);rng=random.Random(42)
    base=root/'data/autolabel/objects_v1';human=root/'data/human_review'
    source={r['image_id']:r for r in map(json.loads,(base/'object_review_manifest.jsonl').read_text().splitlines())}
    db=sqlite3.connect('file:'+str((human/'objects.sqlite').resolve())+'?mode=ro',uri=True)
    db.execute('BEGIN')
    source.update({key:json.loads(raw) for key,raw in db.execute('select image_id,payload from frames')})
    decisions=[json.loads(row[0]) for row in db.execute('select payload from decisions order by image_id')];db.rollback()
    manifest=root/'data/training/manifest.csv';manifest_sha=sha(manifest)
    train={(r['video'],r['group_id'],r['sha256']) for r in csv.DictReader(manifest.open()) if r['split']=='train'}
    old=json.loads((root/'runs/human_training/batch_20261017_211756/detector/protocol.json').read_text())
    held=set(old['heldout_groups']);rows=[];excluded=[];exports=[]
    def save(im,boxes,name,split,s,kind,parents):
        image=out/'dataset/images'/split/(name+'.jpg');label=out/'dataset/labels'/split/(name+'.txt');image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        im.save(image,quality=95);w,h=im.size;lines=[]
        for b in boxes:
            x1,y1,x2,y2=b['xyxy']
            if not 0<=x1<x2<=w or not 0<=y1<y2<=h:raise ValueError('Invalid transformed box')
            lines.append(f'{CLASSES.index(b["class"])} {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
        label.write_text('\n'.join(lines)+('\n' if lines else ''))
        exports.append(dict(image=str(image.relative_to(out)),label=str(label.relative_to(out)),image_sha256=sha(image),label_sha256=sha(label),
            group_id=s['group_id'],source_image_id=s['image_id'],split=split,kind=kind,source_sha256=s['source_sha256'],parents=parents,
            boxes=boxes,human_confirmed=kind=='human_original',synthetic=kind.startswith('synthetic_'),new_independent_video=False))
    for d in decisions:
        s=source[d['image_id']]
        if (s['video'],s['group_id'],s['source_sha256']) not in train:raise ValueError('Frozen source split mismatch')
        if not validate_decision(d,s) or not d.get('eligible_for_detector_training'):
            excluded.append(dict(image_id=d['image_id'],reason='not_complete_certain_human_review'));continue
        p=(human if s.get('origin')=='selected_video_frame' else base)/s['image']
        if sha(p)!=s['image_sha256']:raise ValueError('Changed source frame')
        im=Image.open(p).convert('RGB');split='val' if s['group_id'] in held else 'train';boxes=d['boxes']
        if im.size!=(s['width'],s['height']):raise ValueError('Image size mismatch')
        rows.append(dict(s,human_decision=d,diagnostic_split=split))
        save(im,boxes,s['image_id'],split,s,'human_original',[s['image_id']])
        if split=='val':continue
        w,h=im.size
        for j,b in enumerate(boxes):
            x1,y1,x2,y2=b['xyxy'];cx=(x1+x2)/2;cy=(y1+y2)/2
            side=int(min(w,h,max(640,3*max(x2-x1,y2-y1))))
            left=int(max(0,min(w-side,cx-side/2)));top=int(max(0,min(h-side,cy-side/2)));rect=[left,top,left+side,top+side]
            cropboxes=[];valid=True
            for bb in boxes:
                q=bb['xyxy'];area=overlap(q,rect)
                if not area:continue
                if area/((q[2]-q[0])*(q[3]-q[1]))<.5:valid=False;break
                cropboxes.append(dict(bb,xyxy=[max(q[0],left)-left,max(q[1],top)-top,min(q[2],left+side)-left,min(q[3],top+side)-top]))
            if valid:save(im.crop(rect),cropboxes,s['image_id']+f'_crop{j}',split,s,'human_derived_crop',[s['image_id']])
        targets=[b for b in boxes if b['class'] in ['cash','document_id']]
        if not targets:continue
        # Preserve actual perspective, size and hand occlusions. No new object
        # is pasted into an invented scene. All visible targets retain boxes.
        synthetic=ImageEnhance.Brightness(im).enhance(rng.uniform(.8,1.2))
        synthetic=ImageEnhance.Contrast(synthetic).enhance(rng.uniform(.85,1.15))
        synthetic=synthetic.filter(ImageFilter.GaussianBlur(rng.uniform(0,.5)))
        save(synthetic,boxes,s['image_id']+'_lighting',split,s,'synthetic_photometric',[s['image_id']])
    # Detect an edit/retraction to any included decision during export.
    current={key:json.loads(raw) for key,raw in db.execute('select image_id,payload from decisions')}
    if any(current[r['image_id']]!=r['human_decision'] for r in rows):raise ValueError('A human decision changed during export; rerun a new snapshot')
    if sha(manifest)!=manifest_sha:raise ValueError('Frozen split changed')
    (out/'source_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (out/'augmentation_manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in exports))
    (out/'dataset/dataset.yaml').write_text('path: '+str((out/'dataset').resolve())+'\ntrain: images/train\nval: images/val\nnames:\n'+''.join(f'  {i}: {name}\n' for i,name in enumerate(CLASSES)))
    summary=dict(status='prepared',manual_frames=len(rows),manual_boxes=dict(Counter(b['class'] for r in rows for b in r['human_decision']['boxes'])),
        excluded=excluded,groups=len({r['group_id'] for r in rows}),heldout_groups=sorted(held),
        split_counts={part:dict(original_frames=sum(r['diagnostic_split']==part for r in rows),groups=sorted({r['group_id'] for r in rows if r['diagnostic_split']==part}),boxes=dict(Counter(b['class'] for r in rows if r['diagnostic_split']==part for b in r['human_decision']['boxes']))) for part in ['train','val']},
        generated_counts=dict(Counter(r['kind'] for r in exports)),manifest_sha256=manifest_sha,validation_is_real_only=True,
        new_independent_videos=0,external_ai_used=False,synthetic_caveat='Photometric synthetic variants retain original geometry, perspective and visible hand occlusions. They add no independent examples; arbitrary copy-paste was excluded before training.',
        id_class_semantics='document_id = document / ID candidate; does not establish document authenticity or identity verification')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.root,a.output)
