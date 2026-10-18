"""Short T4 refinement, fixed epochs, immutable real holdout, explicit GPU lease."""
import argparse
import hashlib
import json
from pathlib import Path
import time

def run(a):
    import os
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_AUTOINSTALL='false')
    from ultralytics import YOLO,settings
    from ml.narration_training.organizer.analyze import gpu_lease
    from ml.narration_training.prepare import write_json
    settings.update({'sync':False,'wandb':False,'mlflow':False,'comet':False})
    summary=json.loads((a.dataset/'summary.json').read_text());out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    def emit(**kw):write_json(out/'status.json',dict(updated_at=time.time(),elapsed_sec=time.monotonic()-began,**kw))
    def metrics(model,name):
        result=model.val(data=str(data),split='val',imgsz=960,batch=2,device=0,workers=0,plots=False,project=str(out),name=name,verbose=False)
        per_class={model.names[int(c)]:dict(precision=float(result.box.p[i]),recall=float(result.box.r[i]),ap50=float(result.box.ap50[i]),ap50_95=float(result.box.ap[i])) for i,c in enumerate(result.box.ap_class_index)}
        return dict(aggregate={k:float(v) for k,v in result.results_dict.items()},per_class=per_class)
    data=a.dataset/'dataset/dataset.yaml'
    # Relocation changes only the YAML path; all images and labels are hash checked.
    for r in map(json.loads,(a.dataset/'augmentation_manifest.jsonl').read_text().splitlines()):
        if r['split']=='val' and r['kind']!='human_original':raise ValueError('Synthetic validation is forbidden')
        for k in ['image','label']:
            if hashlib.sha256((a.dataset/r[k]).read_bytes()).hexdigest()!=r[k+'_sha256']:raise ValueError('Dataset checksum mismatch')
    data.write_text('path: '+str((a.dataset/'dataset').resolve())+'\ntrain: images/train\nval: images/val\nnames:\n  0: cash\n  1: document_id\n  2: key\n  3: stamp\n')
    if hashlib.sha256(a.weights.read_bytes()).hexdigest()!='66392416b0d2add78157768bcd5fdb7bc630e5019c45bcb33a30e400a654f464':raise ValueError('Wrong initial detector')
    protocol=dict(**summary,epochs=a.epochs,imgsz=960,freeze=0,learning_rate=.0003,selection='fixed_final_epoch; no threshold/epoch tuning on holdout',baseline='previous human YOLO11m checkpoint',synthetic_training=True,independent_test=False,license='AGPL-3.0 / Enterprise',production_replaced=False)
    write_json(out/'protocol.json',protocol)
    try:
        with gpu_lease(Path.cwd()/'.gpu-work.lock'):
            model=YOLO(str(a.weights));emit(status='running',stage='baseline',epochs=a.epochs)
            before=metrics(model,'baseline');write_json(out/'baseline.json',before)
            def epoch(trainer):
                if time.monotonic()-began>a.max_seconds:raise TimeoutError('Bounded detector pilot exhausted')
                emit(status='running',stage='training',epoch=trainer.epoch+1,epochs=a.epochs,loss=[float(x) for x in trainer.tloss.detach().cpu().flatten()],dataset=summary)
            model.add_callback('on_train_epoch_end',epoch)
            model.train(data=str(data),project=str(out),name='fit',epochs=a.epochs,patience=0,imgsz=960,batch=2,workers=0,device=0,seed=42,deterministic=True,optimizer='AdamW',lr0=.0003,freeze=0,amp=False,mosaic=0,mixup=0,copy_paste=0,fliplr=0,degrees=0,translate=.03,scale=.1,close_mosaic=0,val=False,plots=False,save=True,verbose=False)
            final=out/'fit/weights/last.pt';after=metrics(YOLO(str(final)),'diagnostic')
            report=dict(status='completed',protocol=protocol,baseline=before,trained=after,checkpoint_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),seconds=time.monotonic()-began,auto_confirmation_allowed=False)
            write_json(out/'report.json',report);emit(**report);print(json.dumps(report),flush=True)
    except Exception as exc:emit(status='failed',error=str(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--max-seconds',type=int,default=900);run(p.parse_args())
