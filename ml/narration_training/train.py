"""Nested grouped train comparison; reused val is diagnostic only, fixed 0.5 threshold."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import FeatureStore, VideoBagDataset, collate_bags, loaders
from .model import TemporalNarrationModel, narration_loss
from .prepare import write_json, write_jsonl, new_output, validate_prepared_split
from ml.autolabel.dataset import digest


def metrics(rows):
    y=np.array([r['target'] for r in rows]);p=np.array([r['score'] for r in rows]);pred=p>=.5
    tn,fp,fn,tp=[int(np.sum((y==a)&(pred==b))) for a,b in [(0,0),(0,1),(1,0),(1,1)]]
    auc=None
    if tp+fn and tn+fp:
        differences=p[y==1][:,None]-p[y==0][None,:]
        auc=float(((differences>0)+.5*(differences==0)).mean())
    clipped=np.clip(p,1e-7,1-1e-7)
    recalls=[tp/(tp+fn)] if tp+fn else []
    if tn+fp: recalls.append(tn/(tn+fp))
    return dict(n=len(y),accuracy=float((y==pred).mean()),balanced_accuracy=float(np.mean(recalls)),
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,auroc=auc,
                log_loss=float(-(y*np.log(clipped)+(1-y)*np.log(1-clipped)).mean()),
                confusion_matrix=[[tn,fp],[fn,tp]],threshold=.5)


def train_fit(store,groups,method,lr,epochs,seed):
    if any(store.group_split[g]!='train' for g in groups): raise ValueError('Validation group in fitting')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=TemporalNarrationModel(store.dim)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.02)
    bag_loader,pair_loader=loaders(store,groups)
    bag_batch=next(iter(bag_loader)); pair_batch=next(iter(pair_loader),None)
    used=[] if method=='baseline' or pair_batch is None else list(pair_batch['pair_id'])
    trace=[]
    for epoch in range(epochs):
        model.train();optimizer.zero_grad()
        logits=model.video_logits(bag_batch['video'],bag_batch['sizes'])
        binary=F.binary_cross_entropy_with_logits(logits,bag_batch['target'])
        loss=binary; aux={}
        if method=='narration' and pair_batch is not None:
            weak,aux=narration_loss(model,pair_batch,store.prototypes)
            loss=loss+.8*weak
        # Anchor the adapter to the pretrained space in both experimental arms.
        regularizer=.002*sum(p.square().mean() for p in model.adapter.parameters())
        loss=loss+regularizer;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if epoch==0 or epoch==epochs-1:
            trace.append(dict(epoch=epoch+1,loss=float(loss.detach()),video_loss=float(binary.detach()),**aux))
    excluded=Counter()
    for p in store.pairs:
        if method=='baseline': excluded['baseline_does_not_consume_text']+=1
        elif p['group_id'] not in groups: excluded['held_out_train_group']+=1
        elif not p['eligible_text_loss']: excluded['ineligible_text']+=1
    usage=dict(method=method,groups=sorted(groups),video_bags=len(groups),unique_text_pairs=len(used),
      pair_ids=used,pair_epoch_exposures=len(used)*epochs,excluded_pairs=dict(excluded),
      action_positive_pairs=sum(p['pair_id'] in used and p['eligible_action_positive_loss'] for p in store.pairs),
      strong_action_pairs=0,epochs=epochs,lr=lr,seed=seed,trace=trace)
    return model,usage


def predict(store,model,groups):
    ds=VideoBagDataset(store,groups);loader=DataLoader(ds,batch_size=max(1,len(ds)),collate_fn=collate_bags)
    model.eval();rows=[]
    with torch.no_grad():
        for batch in loader:
            scores=model.video_logits(batch['video'],batch['sizes']).sigmoid().tolist()
            rows.extend(dict(group_id=g,target=int(y),score=p,scope='whole_source_video',score_is_calibrated=False)
                        for g,y,p in zip(batch['group_ids'],batch['target'].tolist(),scores))
    return rows


def folds(store,groups,n_splits,seed):
    groups=sorted(groups); labels={b['group_id']:b['video_target'] for b in store.bags}
    rng=np.random.default_rng(seed);held=[[] for _ in range(n_splits)]
    for label in (0,1):
        members=[g for g in groups if labels[g]==label]
        if len(members)<n_splits: raise ValueError('Too few independent groups for stratification')
        rng.shuffle(members)
        for i,part in enumerate(np.array_split(members,n_splits)): held[i]+=part.tolist()
    return [(sorted(set(groups)-set(part)),sorted(part)) for part in held]


def select(store,groups,method,config,seed,log,scope):
    candidates=[]
    for lr in config['learning_rates']:
        held=[]
        for k,(fit,evaluate) in enumerate(folds(store,groups,3,seed)):
            if set(fit)&set(evaluate): raise AssertionError('Group leakage')
            model,usage=train_fit(store,fit,method,lr,config['epochs'],seed+100+k)
            held+=predict(store,model,evaluate)
            log.append(dict(scope=scope,role='inner_selection',evaluation_groups=evaluate,**usage))
        candidates.append(dict(lr=lr,metrics=metrics(held)))
    best=min(candidates,key=lambda c:c['metrics']['log_loss'])
    return best['lr'],candidates


def paired_bootstrap(a,b,seed=412):
    a=sorted(a,key=lambda r:r['group_id']);b=sorted(b,key=lambda r:r['group_id'])
    assert [r['group_id'] for r in a]==[r['group_id'] for r in b]
    rng=np.random.default_rng(seed);differences=[]
    y=np.array([r['target'] for r in a]);pa=np.array([r['score']>=.5 for r in a]);pb=np.array([r['score']>=.5 for r in b])
    for _ in range(2000):
        ids=rng.integers(0,len(a),len(a)); differences.append(float((pb[ids]==y[ids]).mean()-(pa[ids]==y[ids]).mean()))
    return dict(metric='accuracy_delta_narration_minus_baseline',estimate=float((pb==y).mean()-(pa==y).mean()),
                descriptive_95_percentile_interval=np.quantile(differences,[.025,.975]).tolist(),
                caveat=f'{len(a)} grouped OOF predictions; resampling does not capture model fitting uncertainty or repeated dataset experimentation')


def run(args):
    torch.set_num_threads(args.threads)
    root=Path.cwd().resolve();out=new_output(root,args.output);started=time.monotonic()
    split_policy=validate_prepared_split(root,args.dataset)
    store=FeatureStore(args.dataset,args.features)
    train=[b['group_id'] for b in store.bags if b['split']=='train'];val=[b['group_id'] for b in store.bags if b['split']=='val']
    config=dict(epochs=args.epochs,learning_rates=[.001,.003],outer_folds=4,inner_folds=3,
      seed=7301,threshold=.5,threshold_selection='fixed_before_evaluation',selection_metric='inner_grouped_log_loss',
      methods=['baseline','narration'],narration_weight=.8,action_positive_weight=.25,
      video_loss='whole-source MIL binary cross entropy',action_loss='positive-only prototype cosine loss',
      data_hashes={n:digest(args.dataset/n) for n in ['pairs.jsonl','bags.jsonl','clips.jsonl']},
      feature_hashes={n:digest(args.features/n) for n in ['visual.npz','text.npz']},
      human_source='disabled_pending_schema_agreement',validation_role='reused_diagnostic_not_independent_test',
      split_policy=split_policy,train_source_groups=len(train),diagnostic_source_groups=len(val),independent_test=False)
    write_json(out/'protocol.json',config)
    log=[];oof={method:[] for method in config['methods']};selections=[]
    for k,(fit,evaluate) in enumerate(folds(store,train,4,config['seed'])):
        for method in config['methods']:
            lr,candidates=select(store,fit,method,config,config['seed']+k,log,f'outer_{k}')
            model,usage=train_fit(store,fit,method,lr,args.epochs,config['seed']+1000+k)
            log.append(dict(scope=f'outer_{k}',role='outer_fit',evaluation_groups=evaluate,**usage))
            predictions=predict(store,model,evaluate)
            for row in predictions: row['outer_fold']=k
            oof[method]+=predictions
            selections.append(dict(outer_fold=k,method=method,lr=lr,candidates=candidates))
            print(json.dumps(dict(fold=k,method=method,lr=lr,used_pairs=usage['unique_text_pairs'],metrics=metrics(predictions))),flush=True)
            write_jsonl(out/'data_usage.jsonl',log)
    # Final parameter selection still sees only train. Val is evaluated after all choices are frozen.
    final={};final_selection={};final_models={};final_usage={}
    for method in config['methods']:
        lr,candidates=select(store,train,method,config,config['seed']+4000,log,'final_train_selection')
        model,usage=train_fit(store,train,method,lr,args.epochs,config['seed']+5000)
        log.append(dict(scope='final',role='final_fit',evaluation_groups=val,**usage))
        final_models[method]=model;final_usage[method]=usage
        final_selection[method]=dict(lr=lr,candidates=candidates)
        torch.save(dict(state_dict=model.state_dict(),model_config=dict(dim=store.dim,bottleneck=model.bottleneck),
                        training_protocol=config,data_usage=usage,encoder=store.metadata,
                        action_ids=store.action_ids,action_prototypes=store.prototypes,
                        production_replacement=False),out/f'{method}.pt')
    write_json(out/'selection.json',dict(outer=selections,final=final_selection))
    for method,model in final_models.items():
        final[method]=predict(store,model,val)
        write_jsonl(out/f'{method}_oof.jsonl',oof[method]);write_jsonl(out/f'{method}_reused_val.jsonl',final[method])
    # Diagnostic action rankings are proposals, not calibrated predictions or strong training labels.
    proposals=[];model=final_models['narration'];model.eval()
    with torch.no_grad():
        for clip in store.clips:
            if clip['split']!='train' or clip['kind']!='video_bag_window': continue
            scores=(model.encode(store.visual[clip['clip_id']].unsqueeze(0))@store.prototypes.T)[0]
            top=torch.topk(scores,min(3,len(scores)))
            proposals.append(dict(**clip,action_candidates=[dict(action=store.action_ids[i],cosine_similarity=float(v)) for v,i in zip(top.values,top.indices)],
                review_status='proposed',violation_target=None,score_is_calibrated=False,source='narration_adapter_in_sample_train_diagnostic'))
    write_jsonl(out/'action_proposals_train.jsonl',proposals)
    write_jsonl(out/'data_usage.jsonl',log)
    report=dict(protocol=config,train_oof={m:metrics(r) for m,r in oof.items()},
       paired_accuracy_comparison=paired_bootstrap(oof['baseline'],oof['narration']),
       reused_val_diagnostic={m:metrics(r) for m,r in final.items()},
       final_data_usage=final_usage,event_type_metrics=None,event_boundary_metrics=None,
       unavailable_metrics_reason='No agreed/imported human-confirmed temporal/type reference set. Weak descriptions are not GT.',
       total_elapsed_seconds=time.monotonic()-started,
       conclusion=('No improvement in grouped OOF accuracy.' if metrics(oof['narration'])['accuracy']<=metrics(oof['baseline'])['accuracy']
                   else 'Higher grouped OOF accuracy in this pilot; not independent evidence of generalization.')+
                  ' Reused validation is diagnostic only. No production replacement.')
    write_json(out/'report.json',report)
    print(json.dumps({k:report[k] for k in ['train_oof','paired_accuracy_comparison','reused_val_diagnostic','total_elapsed_seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=60);p.add_argument('--threads',type=int,default=4)
    run(p.parse_args())
