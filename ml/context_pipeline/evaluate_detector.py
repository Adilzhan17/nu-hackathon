"""Old/new × full/native tiles on exactly the same real held-out source frames."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from .bridge import iou
from .objects import detect_image
from ml.autolabel.object_frames import CLASSES

def match(predictions,targets,threshold):
    used=set();correct=[];matches=[]
    for p in predictions:
        candidates=[(iou(p['xyxy'],g['xyxy']),i) for i,g in enumerate(targets) if i not in used and p['class_id']==g['class_id']]
        score,index=max(candidates,default=(0,None));ok=score>=threshold
        correct.append(ok)
        if ok:used.add(index);matches.append(index)
        else:matches.append(None)
    return correct,used,matches

def measure(rows):
    import numpy as np
    from ultralytics.utils.metrics import ap_per_class
    confidence=[];classes=[];truth=[];tp=[];counts={i:Counter() for i in range(len(CLASSES))};errors=[]
    for r in rows:
        p=sorted(r['predictions'],key=lambda x:-x['score']);g=r['targets']
        tp.append(np.array([match(p,g,t)[0] for t in np.arange(.5,.951,.05)],dtype=bool).T.reshape(len(p),10))
        confidence.extend(x['score'] for x in p);classes.extend(x['class_id'] for x in p);truth.extend(x['class_id'] for x in g)
        selected=[x for x in p if x['score']>=.25];correct,used,matches=match(selected,g,.5)
        for target in g:counts[target['class_id']]['support']+=1
        for box,ok in zip(selected,correct):counts[box['class_id']]['tp' if ok else 'fp']+=1
        for i,target in enumerate(g):
            if i not in used:counts[target['class_id']]['fn']+=1
        errors.append(dict(image_id=r['image_id'],group_id=r['group_id'],missed_targets=[g[i] for i in range(len(g)) if i not in used],false_positives=[b for b,ok in zip(selected,correct) if not ok]))
    ap={}
    if confidence:
        result=ap_per_class(np.concatenate(tp),np.array(confidence),np.array(classes),np.array(truth),plot=False)
        ap={int(c):result[5][i] for i,c in enumerate(result[6])}
    per={}
    for i,name in enumerate(CLASSES):
        c=counts[i];per[name]=dict(support=c['support'],tp=c['tp'],fp=c['fp'],fn=c['fn'],
            precision_defined=bool(c['tp']+c['fp']),precision=c['tp']/max(1,c['tp']+c['fp']),recall=c['tp']/c['support'] if c['support'] else None,
            ap50=float(ap[i][0]) if i in ap else (0. if c['support'] else None),map50_95=float(ap[i].mean()) if i in ap else (0. if c['support'] else None))
    return dict(per_class=per,errors=errors,precision_recall_confidence=.25,precision_recall_iou=.5,ap_confidence_floor=.001,
        frame_count=len(rows),source_groups=sorted({r['group_id'] for r in rows}),hand_table_subgroups='not_available: no confirmed per-box hand/table context labels')

def run(a):
    import torch
    from PIL import Image
    from ultralytics import YOLO,settings
    torch.set_num_threads(4);settings.update({'sync':False});a.output.mkdir(parents=True,exist_ok=True)
    source=[json.loads(x) for x in (a.dataset/'source_rows.jsonl').read_text().splitlines()];source=[r for r in source if r['diagnostic_split']=='val']
    image_manifest=[json.loads(x) for x in (a.dataset/'augmentation_manifest.jsonl').read_text().splitlines()]
    if any(r['kind']!='human_original' for r in image_manifest if r['split']=='val'):raise ValueError('Synthetic val forbidden')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    model=YOLO(str(a.weights));checkpoint_sha=sha(a.weights);began=time.monotonic();all_metrics={}
    for tiled in [False,True]:
        name=a.name+('_tiles' if tiled else '_full');folder=a.output/name;folder.mkdir(exist_ok=True);rows=[]
        for s in source:
            image=a.dataset/'dataset/images/val'/(s['image_id']+'.jpg');target=folder/(s['image_id']+'.json')
            identity=dict(checkpoint_sha256=checkpoint_sha,image_sha256=sha(image),source_manifest_sha256=sha(a.dataset/'source_rows.jsonl'),tiled=tiled,size=960,native_tile_size=384,confidence=.001,nms_iou=.5)
            if target.exists():
                r=json.loads(target.read_text())
                if r['identity']!=identity:raise ValueError('Incompatible cached evaluation')
            else:
                predictions,views=detect_image(model,Image.open(image).convert('RGB'),device='cpu',conf=.001,tiled=tiled)
                targets=[dict(class_id=CLASSES.index(b['class']),xyxy=[b['xyxy'][0]/s['width'],b['xyxy'][1]/s['height'],b['xyxy'][2]/s['width'],b['xyxy'][3]/s['height']]) for b in s['human_decision']['boxes']]
                r=dict(identity=identity,image_id=s['image_id'],group_id=s['group_id'],predictions=predictions,targets=targets,views=views)
                target.write_text(json.dumps(r))
            rows.append(r)
            print(json.dumps(dict(condition=name,completed=len(rows),total=len(source),elapsed=time.monotonic()-began)),flush=True)
        metrics=measure(rows);metrics['checkpoint_sha256']=checkpoint_sha;metrics['source_rows_sha256']=sha(a.dataset/'source_rows.jsonl');all_metrics[name]=metrics
        (a.output/(name+'.json')).write_text(json.dumps(metrics,indent=2))
    results={p.stem:json.loads(p.read_text()) for p in a.output.glob('*_full.json')}
    results.update({p.stem:json.loads(p.read_text()) for p in a.output.glob('*_tiles.json')})
    text=[]
    for name,r in sorted(results.items()):
        for cls in ['cash','document_id']:
            m=r['per_class'][cls];precision=f'{m["precision"]:.1%}' if m.get('precision_defined',True) else '— (no predictions)'
            text.append(f'{name} / {cls}: P {precision}, R {m["recall"]:.1%}, AP50 {m["ap50"]:.1%}, mAP50–95 {m["map50_95"]:.1%}; GT={m["support"]}')
    report=dict(conditions=results,summary_text='\n'.join(text),threshold_tuned=False,independent_test=False,
        protocol='Fixed 0.25 for precision/recall; AP over scores >=0.001; real original frames only; same source groups; native 384 tiles with 25% overlap, model input 960; class-aware NMS 0.5.',
        precision_zero_division_convention='Numeric precision=0 when there are no predictions; precision_defined=false and UI displays a dash.',
        limitations=['Only one held-out source video; correlated frames are not independent test videos.','No confirmed hand/table tags, so no trustworthy subgroup metric.','document_id combines documents and ID candidates; no proof of identity.'])
    (a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(dict(summary_text=report['summary_text'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--name',choices=['old','new'],required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
