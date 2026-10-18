"""Fixed-epoch YOLO11m human-box experiment; video-disjoint diagnostic holdout."""
import argparse,hashlib,json,shutil
from pathlib import Path
from collections import Counter

CLASSES=['cash','document_id','key','stamp']
def partition(rows):
    groups=sorted({r['group_id'] for r in rows},key=lambda g:hashlib.sha256(('human-detector-v1:'+g).encode()).hexdigest())
    if len(groups)<3: raise ValueError('At least three source videos required')
    return set(groups[:max(1,round(len(groups)*.2))])

def run(args):
    from ultralytics import YOLO,settings
    from ml.narration_training.extract import gpu_lock
    settings.update({'sync':False,'wandb':False,'mlflow':False,'comet':False})
    root=Path.cwd();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(s) for s in (args.snapshot/'manifest.jsonl').read_text().splitlines()]
    held=partition(rows);parts={'train':[],'val':[]}
    for r in rows:
        assert r['eligible_for_detector_training'] and r['split']=='train'
        name=r['image_id'];im=args.snapshot/'images'/f'{name}.jpg'
        assert hashlib.sha256(im.read_bytes()).hexdigest()==r['image_sha256']
        split='val' if r['group_id'] in held else 'train';parts[split].append(r)
        for folder,ext in [('images','jpg'),('labels','txt')]:
            dest=out/'dataset'/folder/split;dest.mkdir(parents=True,exist_ok=True)
            shutil.copy2(args.snapshot/folder/f'{name}.{ext}',dest/f'{name}.{ext}')
    data=out/'dataset/dataset.yaml'
    data.write_text(f'path: {out / "dataset"}\ntrain: images/train\nval: images/val\nnames:\n'+''.join(f'  {i}: {n}\n' for i,n in enumerate(CLASSES)))
    protocol=dict(snapshot=str(args.snapshot),heldout_groups=sorted(held),train_groups=sorted({r['group_id'] for r in parts['train']}),counts={s:dict(frames=len(rs),classes=dict(Counter(b['class'] for r in rs for b in r['human_decision']['boxes']))) for s,rs in parts.items()},epochs=50,imgsz=1280,checkpoint_selection='fixed final epoch, never best validation',independent_test=False,production_replaced=False,license='Ultralytics AGPL-3.0; experimental checkpoint, not Apache-2.0')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with gpu_lock(root/'.gpu-work.lock'):
        model=YOLO('yolo11m.pt')
        protocol['pretrained_sha256']=hashlib.sha256(Path('yolo11m.pt').read_bytes()).hexdigest()
        protocol['pretrained_parameters']=sum(p.numel() for p in model.model.parameters())
        model.train(data=str(data),project=str(out),name='fit',epochs=50,patience=0,imgsz=1280,batch=4,workers=2,device=0,seed=42,deterministic=True,optimizer='AdamW',lr0=.0003,freeze=10,amp=False,mosaic=0,mixup=0,copy_paste=0,fliplr=0,degrees=0,translate=.05,scale=.15,close_mosaic=0,val=False,plots=False,save=True,verbose=False)
        final=out/'fit/weights/last.pt';model=YOLO(str(final))
        result=model.val(data=str(data),split='val',imgsz=1280,batch=4,device=0,workers=2,plots=False,project=str(out),name='diagnostic')
        report=dict(status='completed',**protocol,metrics={k:float(v) for k,v in result.results_dict.items()},checkpoint=str(final),human_boxes=sum(len(r['human_decision']['boxes']) for r in rows),auto_confirmation_allowed=False)
        (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
