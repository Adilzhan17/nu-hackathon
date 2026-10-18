"""Frozen-backbone baselines with explicit versioned split policy.

With active_split.json, new default runs use a fixed final epoch and evaluate the
diagnostic set once. Explicit --manifest data/training/manifest.csv retains the
legacy train/val behavior. This entrypoint does not alter narration training or
the separate train_x3d.py controller.
"""
import argparse
import csv
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from model_suite import MODELS,Extractor,load_frames
from train_x3d import atomic_json,metrics


def resolve_training_config(root, manifest=None, diagnostic_test=False):
    """An active next-run policy applies only when the CLI omits --manifest."""
    root=Path(root).resolve()
    if manifest is not None:
        return Path(manifest),diagnostic_test,dict(active_policy_used=False)
    active=root/'data/training/active_split.json'
    if not active.exists():
        return root/'data/training/manifest.csv',diagnostic_test,dict(active_policy_used=False)
    from split95 import load_split
    pointer=json.loads(active.read_text(encoding='utf-8'))
    report=load_split(root,version=pointer.get('version',''))
    requested=Path(pointer.get('manifest',''))
    expected=(root/report['manifest_relative_path']).resolve()
    if requested.is_absolute() or '..' in requested.parts or (root/requested).resolve()!=expected:
        raise ValueError('Active split manifest path does not match its verified version')
    if pointer.get('source_manifest_sha256')!=report['source_manifest_sha256']:
        raise ValueError('Active split source hash does not match the verified manifest')
    if pointer.get('requires_diagnostic_test_mode') is not True or pointer.get('independent_test') is not False or pointer.get('diagnostic_only') is not True:
        raise ValueError('Active split evaluation policy is invalid')
    return expected,True,dict(active_policy_used=True,split_version=report['version'],
        source_manifest_sha256=report['source_manifest_sha256'])


def read_training_manifest(manifest, diagnostic_test=False):
    """Keep source groups and SHA aliases indivisible before any feature extraction."""
    with Path(manifest).open(encoding='utf-8-sig') as handle:
        rows=list(csv.DictReader(handle))
    if not rows:raise ValueError('Training manifest is empty')
    partitions={r['split'] for r in rows}
    if diagnostic_test:
        heldout=partitions-{'train'}
        if len(heldout)!=1 or not heldout.issubset({'val','test','diagnostic_test'}):
            raise ValueError('Diagnostic mode requires train and one held-out partition')
        evaluation_split=next(iter(heldout))
    else:
        if partitions!={'train','val'}:raise ValueError('Use --diagnostic-test for a diagnostic/test manifest')
        if Path(manifest).parent.name.startswith('split95_v'):
            raise ValueError('The 95/5 split requires --diagnostic-test; do not select epochs on its held-out videos')
        evaluation_split='val'
    if 'train' not in partitions:raise ValueError('No training partition')
    for field in ('group_id','sha256','video'):
        train_ids={r[field] for r in rows if r['split']=='train'}
        evaluation_ids={r[field] for r in rows if r['split']==evaluation_split}
        if train_ids&evaluation_ids:raise ValueError('Split leakage: '+field)
    for row in rows:
        path=Path(row['video'])
        if path.is_absolute() or '..' in path.parts:raise ValueError('Invalid source video path')
        if row['target'] not in {'0','1'}:raise ValueError('Expected binary source target')
    if {r['target'] for r in rows if r['split']=='train'}!={'0','1'}:
        raise ValueError('Training requires both binary labels')
    return rows,evaluation_split


def fit_fixed_diagnostic_head(head, x_train, y_train, epochs, check_stop=lambda:None, emit=lambda **_:None):
    """Fixed-epoch optimization receives no diagnostic features or labels.

    The final checkpoint and the fixed .5 threshold are chosen without inspecting
    the diagnostic set. Epoch count is a configuration, not tuned on two videos.
    """
    import torch
    if epochs<1:raise ValueError('epochs must be positive')
    optimizer=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.1)
    weights=torch.where(y_train>0,1/y_train.sum(),1/(1-y_train).sum())
    history=[]
    for epoch in range(1,epochs+1):
        check_stop();head.train();optimizer.zero_grad()
        logits=head(torch.nn.functional.dropout(x_train,p=.35,training=True)).flatten()
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y_train,reduction='none')*weights).sum()/2
        loss.backward();optimizer.step()
        result=dict(epoch=epoch,train_loss=loss.item(),checkpoint_selection='fixed_final_epoch_no_diagnostic_access')
        history.append(result)
        emit(stage='training',epoch=epoch,epochs=epochs,metrics=result,message='Фиксированные эпохи; диагностические метки не используются')
    return history


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('runs/suite'))
    baseline_keys=[k for k,v in MODELS.items() if v.get('training_mode')!='full_finetune']
    parser.add_argument('--models',nargs='+',choices=baseline_keys,default=baseline_keys)
    parser.add_argument('--manifest',type=Path,help='Explicit override; otherwise use verified active_split.json or the original manifest')
    parser.add_argument('--diagnostic-test',action='store_true',help='Fixed epochs; evaluate held-out set once after fitting, never select checkpoint on it')
    parser.add_argument('--epochs',type=int,default=120)
    args=parser.parse_args()
    if args.epochs<1:parser.error('--epochs must be positive')
    manifest,args.diagnostic_test,selected_policy=resolve_training_config(Path.cwd(),args.manifest,args.diagnostic_test)
    rows,evaluation_split=read_training_manifest(manifest,args.diagnostic_test)
    if args.diagnostic_test and manifest.parent.name.startswith('split95_v'):
        from split95 import load_split
        frozen=load_split(Path.cwd(),version=manifest.parent.name)
        if manifest.resolve() not in {(Path.cwd()/'data/training'/frozen['version']/name).resolve() for name in ('manifest.csv','manifest_legacy.csv')}:
            raise ValueError('Diagnostic manifest is outside the verified split version')
    if args.diagnostic_test and args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Diagnostic run must use a fresh output directory; existing runs are immutable')
    args.output.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(8)
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
    run_policy=dict(manifest=str(manifest),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        diagnostic_test=args.diagnostic_test,independent_test=False,evaluation_split=evaluation_split,
        checkpoint_selection='fixed_final_epoch' if args.diagnostic_test else 'legacy_validation_balanced_accuracy',
        threshold_selection='fixed_0.5',epochs=args.epochs,training_started=False,**selected_policy)
    atomic_json(args.output/'split_policy.json',run_policy)
    for path in {r['video'] for r in rows}:
        expected=next(r['sha256'] for r in rows if r['video']==path)
        if hashlib.sha256((Path('data/raw/open')/path).read_bytes()).hexdigest()!=expected:raise ValueError('Video checksum mismatch')
    for key in args.models:
        run_policy['training_started']=True
        atomic_json(args.output/'split_policy.json',run_policy)
        folder=args.output/key;folder.mkdir(exist_ok=True)
        state=dict(model_key=key,name=MODELS[key]['name'],status='running',stage='loading',started_at=time.time(),history=[])
        def emit(**updates):
            state.update(updated_at=time.time(),**updates)
            atomic_json(folder/'status.json',state)
            print(json.dumps(dict(model=key,**updates),ensure_ascii=False),flush=True)
        def check_stop():
            if (args.output/'STOP').exists():raise InterruptedError('Остановка запрошена')
        try:
            check_stop();emit(message='Загрузка предобученной модели')
            torch.manual_seed(42)
            extractor=Extractor(key,Path('artifacts/assets'),True,'cuda')
            features=[]
            cfg=MODELS[key]
            for index,row in enumerate(rows):
                check_stop()
                frames=load_frames(Path('data/raw/open')/row['video'],float(row['start_sec']),float(row['end_sec']),cfg['frames'],cfg['size'])
                features.append(extractor.encode(frames.unsqueeze(0)).cpu())
                emit(stage='features',processed=index+1,total=len(rows),message='Извлечение признаков из размеченных окон')
            x=torch.cat(features).float()
            train=torch.tensor([r['split']=='train' for r in rows]);val=~train
            y=torch.tensor([int(r['target']) for r in rows],dtype=torch.float32)
            mean=x[train].mean(0,keepdim=True);std=x[train].std(0,keepdim=True).clamp(min=.1)
            x=((x-mean)/std).to('cuda');y=y.to('cuda');train=train.to('cuda');val=val.to('cuda')
            head=torch.nn.Linear(x.shape[1],1).cuda()
            torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
            if args.diagnostic_test:
                state['history']=fit_fixed_diagnostic_head(head,x[train],y[train],args.epochs,check_stop,emit)
                head.eval()
                with torch.no_grad():
                    probabilities=head(x[val]).flatten().sigmoid()
                    vloss=torch.nn.functional.binary_cross_entropy(probabilities,y[val]).item()
                best_result=dict(state['history'][-1],diagnostic_loss=vloss,independent_test=False,
                    **metrics(y[val].int().tolist(),(probabilities>=.5).int().tolist()))
                best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                best_probs=probabilities.cpu().tolist()
            else:
                optimizer=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.1)
                weights=torch.where(y[train]>0,1/y[train].sum(),1/(1-y[train]).sum())
                best=-1;best_state=None;best_result=None;stale=0
                for epoch in range(1,args.epochs+1):
                    check_stop()
                    head.train();optimizer.zero_grad()
                    logits=head(torch.nn.functional.dropout(x[train],p=.35,training=True)).flatten()
                    loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y[train],reduction='none')*weights).sum()/2
                    loss.backward();optimizer.step()
                    head.eval()
                    with torch.no_grad():
                        probabilities=head(x[val]).flatten().sigmoid()
                        vloss=torch.nn.functional.binary_cross_entropy(probabilities,y[val]).item()
                    result=dict(epoch=epoch,train_loss=loss.item(),val_loss=vloss,
                                **metrics(y[val].int().tolist(),(probabilities>=.5).int().tolist()))
                    state['history'].append(result)
                    score=result['balanced_accuracy']
                    if score>best or (score==best and (best_result is None or vloss<best_result['val_loss'])):
                        best=score;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};best_result=result
                        best_probs=probabilities.cpu().tolist();stale=0
                    else:stale+=1
                    emit(stage='training',epoch=epoch,epochs=args.epochs,metrics=result,best_metrics=best_result,message='Обучение классификатора на замороженных признаках')
                    if stale>=20 and epoch>=30:break
            emit(stage='saving',message='Сохранение весов и проверки')
            if hashlib.sha256(manifest.read_bytes()).hexdigest()!=run_policy['manifest_sha256']:
                raise ValueError('Manifest changed during training; checkpoint not published')
            bundle=dict(schema_version=1,model_key=key,backbone={k:v.detach().cpu() for k,v in extractor.model.state_dict().items()},
                        feature_mean=mean,feature_std=std,weight=best_state['weight'],bias=best_state['bias'],
                        metrics=best_result,preprocessing=dict(cfg,window_sec=8,stride_sec=4,pad=114,resize='letterbox'),
                        manifest_sha256=run_policy['manifest_sha256'],threshold=.5,split_policy=run_policy)
            torch.save(bundle,folder/'model.tmp.pt');os.replace(folder/'model.tmp.pt',folder/'model.pt')
            validation=dict(metrics=best_result,scope='organizer_windows',
                            independent_test=False,diagnostic_only=args.diagnostic_test,evaluation_split=evaluation_split,
                            predictions=[dict(video=r['video'],start_sec=r['start_sec'],target=int(r['target']),probability=p)
                                         for r,p in zip([r for r in rows if r['split']==evaluation_split],best_probs)])
            atomic_json(folder/('diagnostic_test.json' if args.diagnostic_test else 'validation.json'),validation)
            emit(status='completed',stage='completed',message='Модель обучена; ожидает копирования на основной сервер',
                 elapsed_sec=time.time()-state['started_at'],checkpoint_sha256=hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest())
            del extractor,head,x,features
            torch.cuda.empty_cache()
        except InterruptedError as exc:
            emit(status='stopped',stage='stopped',message=str(exc));break
        except Exception as exc:
            emit(status='failed',stage='failed',message=str(exc));traceback.print_exc()
    (args.output/'DONE').write_text(str(time.time()))


if __name__=='__main__':main()
