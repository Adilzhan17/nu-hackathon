"""Diverse native-resolution train frames and explicit human-only detector annotations."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from .dataset import registry,snapshot,digest,read_jsonl
from .pipeline import inside,save,jsonl
from .sampling import frames

CLASSES=['cash','document_id','key','stamp']
VISIBILITY={'clear','partly_occluded','uncertain'}


def choose(pool,duration,count):
    """One frame per time stratum, favoring appearance diversity within each stratum.

    Selection does not assert the presence of any target object.
    """
    chosen=[];features={}
    for f in pool:
        with Image.open(f['_path']) as im:
            features[f['timestamp']]=np.asarray(im.convert('RGB').resize((24,24)),dtype=float).reshape(-1)/255
    for i in range(count):
        a=i*duration/count;b=(i+1)*duration/count
        options=[f for f in pool if a<=f['timestamp']<b]
        if not options:options=[f for f in pool if f not in chosen]
        if not options:break
        def score(f):
            x=features[f['timestamp']]
            diversity=min(float(np.abs(x-features[c['timestamp']]).mean()) for c in chosen) if chosen else float(x.std())
            center=1-abs(f['timestamp']-(a+b)/2)/max(duration/count,1e-6)
            return diversity+.015*center
        chosen.append(max(options,key=score))
    return sorted(chosen,key=lambda f:f['timestamp'])


def prepare(root,run,out,per_video=12):
    root=Path(root).resolve();run=inside(run,root/'data/autolabel');out=inside(out,root/'data/autolabel')
    if out.exists():raise FileExistsError('Use a new object dataset directory')
    if not 1<=per_video<=100:raise ValueError('Invalid frame budget')
    plan=json.loads((run/'plan.json').read_text());reg,_,_=registry(root)
    if snapshot(root)!=plan['protected_inputs'] or reg!=plan['registry']:raise ValueError('Changed sources')
    pools=defaultdict(dict)
    for c in plan['candidates']:
        for f in c['frames']:pools[c['group_id']].setdefault(f['timestamp'],dict(f,_path=str(run/f['path'])))
    selected=[]
    for s in plan['scans']:
        source=reg[s['video']]
        if source['split']!='train':raise ValueError('Train only')
        # Limit selection feature work to approximately 2 samples/s; native pixels decoded below.
        pool=sorted(pools[s['group_id']].values(),key=lambda f:f['timestamp']);coarse=[];last=-1.
        for f in pool:
            if f['timestamp']-last>=.45:coarse.append(f);last=f['timestamp']
        for f in choose(coarse,s['duration_sec'],per_video):
            selected.append(dict(video=s['video'],group_id=s['group_id'],split='train',timestamp=f['timestamp'],
                                 width=s['width'],height=s['height'],source_sha256=source['sha256'],duration_sec=s['duration_sec']))
    out.mkdir(parents=True);(out/'images').mkdir()
    def extract(item):
        i,r=item;r=dict(r);t=r['timestamp']
        samples=frames(root/'data/raw/open'/r['video'],max(0,t-.000001),min(r['duration_sec'],t+.12),30,max_side=max(r['width'],r['height']))
        actual,im=next(samples);samples.close()
        if abs(actual-t)>.00001 or im.size!=(r['width'],r['height']):raise ValueError('Native frame provenance mismatch')
        image_id=f'frame_{i:04d}';dest=out/'images'/(image_id+'.jpg');im.save(dest,quality=95,subsampling=0)
        return dict(r,image_id=image_id,image=f'images/{image_id}.jpg',image_sha256=digest(dest),
                    selection='temporal_strata_and_visual_diversity',review_status='unreviewed',
                    annotation_origin=None,objects=None,eligible_for_detector_training=False)
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(extract,enumerate(selected)))
    jsonl(out/'object_review_manifest.jsonl',rows)
    summary=dict(frames=len(rows),train_groups=len({r['group_id'] for r in rows}),classes=CLASSES,
                 reviewed_frames=0,ground_truth_boxes=0,protected_inputs=snapshot(root),
                 source_plan_sha256=digest(run/'plan.json'),native_resolution=True,
                 note='Selection is class-agnostic. Unreviewed/uncertain frames are not negative detection examples.')
    save(out/'summary.json',summary)
    template=Path(__file__).with_name('objects.html').read_text()
    data=json.dumps(dict(rows=rows,classes=CLASSES),ensure_ascii=False).replace('<','\\u003c')
    (out/'index.html').write_text(template.replace('/*__DATA__*/',data),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def validate_decision(d,source):
    if d.get('image_id')!=source['image_id'] or d.get('image_sha256')!=source['image_sha256']:raise ValueError('Image provenance mismatch')
    if not isinstance(d.get('reviewer'),str) or not d['reviewer'].strip():raise ValueError('Named human reviewer required')
    if d.get('review_status') not in {'complete','needs_context'}:raise ValueError('Explicit review status required')
    if d.get('image_assessment') not in {'targets_annotated','confirmed_no_target','uncertain'}:raise ValueError('Explicit image assessment required')
    if not isinstance(d.get('boxes'),list):raise ValueError('Boxes must be a list')
    for b in d['boxes']:
        if b.get('class') not in CLASSES or b.get('visibility') not in VISIBILITY:raise ValueError('Unknown class/visibility')
        coords=b.get('xyxy')
        if not isinstance(coords,list) or len(coords)!=4 or any(type(x) not in {int,float} or not math.isfinite(x) for x in coords):raise ValueError('Invalid box coordinates')
        x1,y1,x2,y2=coords
        if not 0<=x1<x2<=source['width'] or not 0<=y1<y2<=source['height']:raise ValueError('Box outside original frame')
    if d['image_assessment']=='confirmed_no_target' and d['boxes']:raise ValueError('Negative image contains boxes')
    if d['image_assessment']=='targets_annotated' and not d['boxes']:raise ValueError('Missing target boxes')
    # Partial images and uncertain target identities cannot silently become background.
    return d['review_status']=='complete' and d['image_assessment']!='uncertain' and all(b['visibility']!='uncertain' for b in d['boxes'])


def import_decisions(root,source_dir,decisions,out):
    root=Path(root).resolve();source_dir=inside(source_dir,root/'data/autolabel');out=inside(out,root/'data/autolabel')
    if out.exists():raise FileExistsError('Use a new annotation version')
    meta=json.loads((source_dir/'summary.json').read_text())
    if snapshot(root)!=meta['protected_inputs']:raise ValueError('Protected inputs changed')
    reg,_,_=registry(root);source={r['image_id']:r for r in read_jsonl(source_dir/'object_review_manifest.jsonl')}
    records=read_jsonl(decisions);seen=set();accepted=[];pending=[]
    for d in records:
        image_id=d.get('image_id')
        if image_id not in source or image_id in seen:raise ValueError('Unknown or repeated image ID')
        seen.add(image_id);s=source[image_id];r=reg[s['video']]
        if s['split']!='train' or (r['group_id'],r['split'],r['sha256'])!=(s['group_id'],'train',s['source_sha256']):raise ValueError('Group leakage/source mismatch')
        path=inside(source_dir/s['image'],source_dir/'images')
        if digest(path)!=s['image_sha256']:raise ValueError('Image changed')
        with Image.open(path) as im:
            if im.size!=(s['width'],s['height']):raise ValueError('Image dimensions changed')
        eligible=validate_decision(d,s)
        row=dict(s,human_decision=d,annotation_origin='human',review_status=d['review_status'],eligible_for_detector_training=eligible)
        (accepted if eligible else pending).append(row)
    out.mkdir(parents=True);(out/'images').mkdir();(out/'labels').mkdir()
    images=[];annotations=[]
    for i,r in enumerate(accepted,1):
        name=r['image_id']+'.jpg';shutil.copy2(source_dir/r['image'],out/'images'/name)
        images.append(dict(id=i,file_name='images/'+name,width=r['width'],height=r['height'],video=r['video'],group_id=r['group_id'],split='train',
                           source_timestamp=r['timestamp'],source_sha256=r['source_sha256'],reviewer=r['human_decision']['reviewer']))
        labels=[]
        for b in r['human_decision']['boxes']:
            x1,y1,x2,y2=b['xyxy'];w=x2-x1;h=y2-y1;category=CLASSES.index(b['class'])
            annotations.append(dict(id=len(annotations)+1,image_id=i,category_id=category+1,bbox=[x1,y1,w,h],area=w*h,iscrowd=0,
                                    visibility=b['visibility'],annotation_origin='human',review_status='accepted'))
            labels.append(f'{category} {(x1+x2)/2/r["width"]:.8f} {(y1+y2)/2/r["height"]:.8f} {w/r["width"]:.8f} {h/r["height"]:.8f}')
        (out/'labels'/(r['image_id']+'.txt')).write_text('\n'.join(labels)+('\n' if labels else ''))
    save(out/'coco_train.json',dict(images=images,annotations=annotations,categories=[dict(id=i+1,name=c) for i,c in enumerate(CLASSES)]))
    jsonl(out/'accepted_images.jsonl',accepted);jsonl(out/'pending_images.jsonl',pending)
    save(out/'summary.json',dict(reviewed_decisions=len(records),accepted_images=len(accepted),pending_images=len(pending),human_boxes=len(annotations),
                                category_counts=dict(Counter(CLASSES[b['category_id']-1] for b in annotations)),protected_inputs=meta['protected_inputs'],
                                decisions_sha256=digest(decisions),validation_set=None,
                                note='Train-only annotations. No val labels synthesized. Uncertain images excluded entirely from YOLO/COCO training.'))
    (out/'train.txt').write_text('\n'.join(str(out/'images'/Path(im['file_name']).name) for im in images)+'\n')
    save(out/'class_names.json',CLASSES)
    print(json.dumps(dict(accepted_images=len(accepted),human_boxes=len(annotations),pending_images=len(pending))))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','import']);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--run',type=Path,default=Path('data/autolabel/detailed_train'));p.add_argument('--source',type=Path,default=Path('data/autolabel/objects_v1'));p.add_argument('--output',type=Path,required=True);p.add_argument('--decisions',type=Path);p.add_argument('--per-video',type=int,default=12);a=p.parse_args()
    if a.command=='prepare':prepare(a.root,a.run,a.output,a.per_video)
    else:
        if a.decisions is None:p.error('--decisions is required')
        import_decisions(a.root,a.source,a.decisions,a.output)

if __name__=='__main__':main()
