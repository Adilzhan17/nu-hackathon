"""Bounded CPU experiment: frozen VideoMAE + action head on confirmed train intervals."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

def run(args):
    import numpy as np
    import torch
    from torch.utils.data import Dataset,DataLoader
    root=Path.cwd();sys.path.insert(0,str(root/'ml'))
    from model_suite import load_bundle,load_frames
    from ml.narration_training.prepare import write_json
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if args.output.exists():raise ValueError('Use a new experiment directory')
    args.output.mkdir(parents=True);began=time.monotonic();torch.set_num_threads(4);torch.manual_seed(42)
    manifest=root/'data/training/manifest.csv';manifest_hash=sha(manifest)
    mapping={r['video']:r for r in csv.DictReader(manifest.open())}
    rows=[json.loads(x) for x in args.source.read_text().splitlines()];used=[];excluded=[];verified=set()
    def status(**kw):write_json(args.output/'status.json',dict(updated_at=time.time(),elapsed_sec=time.monotonic()-began,**kw))
    def budget():
        if time.monotonic()-began>args.max_seconds:raise TimeoutError('CPU training budget exhausted')
    status(status='running',stage='audit',submitted=len(rows))
    try:
        for r in rows:
            reason=None;m=mapping.get(r.get('video'))
            if not m or m['split']!='train' or r.get('split')!='train':reason='not_frozen_train'
            elif not (r.get('action_confirmed') is True and r.get('eligible_for_action_training') is True and r.get('decision')=='accept'):reason='not_human_confirmed'
            elif r.get('observed_action') in ['other_observed_action','unclear_action']:reason='unspecified_action'
            elif r.get('video_sha256')!=m['sha256'] or r['group_id']!='video_'+m['sha256'][:16]:reason='source_mismatch'
            elif not 0<=r['start_sec']<r['end_sec']:reason='invalid_bounds'
            if reason:excluded.append(dict(task_id=r.get('task_id'),reason=reason));continue
            if r['video'] not in verified:
                if sha(root/'data/raw/open'/r['video'])!=m['sha256']:raise ValueError('Changed video')
                verified.add(r['video'])
            used.append(r)
        classes=sorted({r['observed_action'] for r in used})
        if len(classes)<2 or len({r['group_id'] for r in used})<2:raise ValueError('Insufficient confirmed groups/classes')
        class Clips(Dataset):
            def __len__(self):return len(used)
            def __getitem__(self,i):
                r=used[i]
                return load_frames(root/'data/raw/open'/r['video'],r['start_sec'],r['end_sec'],16,224),i
        extractor,_=load_bundle(root/'artifacts/models/videomaev2/model.pt',root/'artifacts/assets','cpu')
        features=[]
        for x,indices in DataLoader(Clips(),batch_size=1,shuffle=False,num_workers=0):
            budget();features.append(extractor.encode(x).float().cpu());status(status='running',stage='features',processed=len(features),total=len(used),device='CPU',trainable_component='action_head')
        del extractor
        x=torch.cat(features);y=torch.tensor([classes.index(r['observed_action']) for r in used]);groups=[r['group_id'] for r in used]
        np.savez_compressed(args.output/'features.npz',features=x.numpy(),targets=y.numpy(),groups=np.array(groups))
        def fit(indices):
            torch.manual_seed(42);a=x[indices];mu=a.mean(0);std=a.std(0,unbiased=False).clamp_min(.1);z=(a-mu)/std
            head=torch.nn.Linear(x.shape[1],len(classes));opt=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.1)
            for epoch in range(80):
                budget();opt.zero_grad();loss=torch.nn.functional.cross_entropy(head(z),y[indices]);loss.backward();opt.step()
            return head,mu,std,float(loss.detach())
        predictions=[]
        for g in sorted(set(groups)):
            tr=[i for i,v in enumerate(groups) if v!=g];va=[i for i,v in enumerate(groups) if v==g]
            head,mu,std,_=fit(tr);p=head((x[va]-mu)/std).argmax(1)
            z=torch.nn.functional.normalize((x-mu)/std,dim=1)
            nearest=(z[va]@z[tr].T).argmax(1)
            for j,i in enumerate(va):predictions.append(dict(task_id=used[i]['task_id'],group_id=g,target=classes[y[i]],predicted=classes[p[j]],baseline=classes[y[tr[int(nearest[j])]]],class_seen_in_fold=bool((y[tr]==y[i]).any())))
            status(status='running',stage='grouped_evaluation',completed_groups=len({p['group_id'] for p in predictions}),total_groups=len(set(groups)),device='CPU')
        head,mu,std,loss=fit(list(range(len(used))))
        torch.save(dict(schema_version=1,classes=classes,weight=head.weight.detach(),bias=head.bias.detach(),feature_mean=mu,feature_std=std,
            backbone_sha256=sha(root/'artifacts/models/videomaev2/model.pt'),manifest_sha256=manifest_hash,source_sha256=sha(args.source),
            training_scope='human_confirmed_train_intervals',score_semantics='uncalibrated_action_ranking_not_violation',license='CC-BY-NC-4.0'),args.output/'action_head.pt')
        report=dict(status='completed',submitted=len(rows),used=len(used),excluded=excluded,groups=len(set(groups)),classes=classes,
            epochs=80,device='CPU',frozen_backbone=True,trainable_parameters=sum(p.numel() for p in head.parameters()),
            baseline_top1=sum(r['baseline']==r['target'] for r in predictions)/len(predictions),
            trained_top1=sum(r['predicted']==r['target'] for r in predictions)/len(predictions),
            unseen_class_examples=sum(not p['class_seen_in_fold'] for p in predictions),predictions=predictions,
            evaluation='leave-one-source-group-out action agreement on sparse confirmed intervals',independent_test=False,
            hyperparameters_fixed_before_evaluation=True,threshold_tuned=False,violation_metric=None,background_class_trained=False,
            manifest_sha256=manifest_hash,source_sha256=sha(args.source),checkpoint_sha256=sha(args.output/'action_head.pt'),
            loss=loss,production_replaced=False,seconds=time.monotonic()-began)
        if sha(manifest)!=manifest_hash:raise ValueError('Frozen split changed during training')
        write_json(args.output/'report.json',report);write_json(args.output/'training_rows.json',used)
        status(**report);print(json.dumps({k:v for k,v in report.items() if k not in ['predictions','classes']},ensure_ascii=False))
    except Exception as exc:
        status(status='failed',error=str(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-seconds',type=int,default=500);run(p.parse_args())
