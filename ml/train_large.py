"""Bounded full-backbone VideoMAE V2 Large fine-tuning on the frozen organizer split."""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path
from train_x3d import atomic_json, metrics
from model_suite import load_frames

KEY='videomaev2_large'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def checked_rows(root,manifest):
    with Path(manifest).open() as f:rows=list(csv.DictReader(f))
    hashes={'train':set(),'val':set()};seen={};base=(Path(root)/'data/raw/open').resolve()
    for row in rows:
        path=(base/row['video']).resolve()
        if not path.is_relative_to(base) or row['split'] not in hashes or row['target'] not in ('0','1'):raise ValueError('Invalid manifest row')
        if not 0<=float(row['start_sec'])<float(row['end_sec']):raise ValueError('Invalid interval')
        if path not in seen:seen[path]=digest(path)
        if seen[path]!=row['sha256']:raise ValueError('Source hash mismatch')
        hashes[row['split']].add(row['sha256'])
    if hashes['train']&hashes['val']:raise ValueError('Train/validation leakage')
    for split in hashes:
        if {r['target'] for r in rows if r['split']==split}!={'0','1'}:raise ValueError('Both classes required')
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model-key',choices=['videomaev2_large','videomaev2_giant'],default=KEY);parser.add_argument('--output',type=Path,default=Path('runs/large_v1'))
    parser.add_argument('--max-seconds',type=int,default=900);parser.add_argument('--epochs',type=int,default=16)
    args=parser.parse_args();key=args.model_key;display='VideoMAE V2 '+('Giant' if key.endswith('giant') else 'Large');run=args.output;run.mkdir(parents=True,exist_ok=True)
    regularized=key.endswith('giant');backbone_lr=5e-6 if regularized else 2e-5;head_lr=3e-5 if regularized else 1e-4;dropout=.5 if regularized else .2;smoothing=.1 if regularized else 0.;warmup_steps=30 if regularized else 100
    began=time.monotonic();stop=False
    def stop_signal(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    state=dict(model_key=key,name=display+' · full fine-tune',status='running',stage='loading',started_at=time.time(),history=[],epochs=args.epochs,training_mode='full_finetune')
    def emit(**updates):
        state.update(updated_at=time.time(),elapsed_sec=time.monotonic()-began,**updates);atomic_json(run/'status.json',state)
        print(json.dumps(updates,ensure_ascii=False),flush=True)
    def check():
        if stop or (run/'STOP').exists():raise InterruptedError('Остановка запрошена')
        if time.monotonic()-began>args.max_seconds:raise TimeoutError('Достигнут лимит времени запуска')
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoConfig,AutoModel
        torch.set_num_threads(8);random.seed(42);torch.manual_seed(42)
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.backends.cuda.matmul.allow_tf32=True
        device='cuda';dtype=torch.bfloat16
        rows=checked_rows(Path.cwd(),Path('data/training/manifest.csv'))
        manifest_hash=digest('data/training/manifest.csv');assets=Path('artifacts/assets')/key
        source=json.loads((assets/'source.json').read_text())
        for name,sha in source['files'].items():
            if digest(assets/name)!=sha:raise ValueError('Pinned model file changed: '+name)
        config=AutoConfig.from_pretrained(str(assets),trust_remote_code=True,local_files_only=True)
        model=AutoModel.from_pretrained(str(assets),config=config,trust_remote_code=True,local_files_only=True).to(device)
        # Memory-saving recomputation preserves the official model's attention and normalization.
        model.model.with_cp=False
        feature_dim=config.model_config['embed_dim'];depth=config.model_config['depth']
        head=torch.nn.Linear(feature_dim,1).to(device);torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
        params=sum(p.numel() for p in model.parameters())+sum(p.numel() for p in head.parameters())
        if params>3_000_000_000:raise ValueError('Model exceeds the user limit of 3B parameters')
        settings=dict(model_key=key,threshold=.5,feature_dim=feature_dim,trainable_parameters=params,manifest_sha256=manifest_hash,
            preprocessing=dict(frames=16,size=224,window_sec=8,stride_sec=4,pad=114,resize='letterbox'),
            training=dict(backbone='full_finetune',head='Linear → sigmoid',optimizer='AdamW',learning_rate=backbone_lr,head_learning_rate=head_lr,
                layer_decay=.85,weight_decay=.05,dropout=dropout,label_smoothing=smoothing,seed=42,max_epochs=args.epochs,early_stopping_patience=5,min_epochs=6,
                batch='4 клипа × 2 шага накопления = 8',loss=f'BCEWithLogits · balanced sampling · label smoothing {smoothing}',selection='Balanced accuracy; при равенстве — меньший val loss',
                feature_normalization='Среднее и std только train; std ≥ 0.1',precision='BF16 training / FP32 validation',gradient_checkpointing=True,warmup_steps=warmup_steps,
                augmentation='Brightness ±10%, offset ±0.03; без перестановки кадров',max_seconds=args.max_seconds),
            provenance=dict(preprocessing='Из конфигурации текущего запуска',training='Сохранено до обучения; полное дообучение всех блоков',source=source))
        atomic_json(run/'settings.json',settings)
        emit(message='Большая модель загружена. Подготовка окон',total_parameters=params,trainable_parameters=params,gpu_name=torch.cuda.get_device_name(),torch_version=torch.__version__)
        tensors=[]
        for i,row in enumerate(rows):
            check();tensors.append((load_frames(Path('data/raw/open')/row['video'],float(row['start_sec']),float(row['end_sec']),16,224)*255).round().byte())
            emit(stage='features',processed=i+1,total=len(rows),message='Декодирование полных кадров без обрезки сцены')
        indices={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ('train','val')}
        targets=torch.tensor([int(r['target']) for r in rows],device=device,dtype=torch.float32)
        image_mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1,1)
        image_std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1,1)
        def frames(ids,augment=False):
            x=torch.stack([tensors[i] for i in ids]).to(device).float()/255
            if augment:x=(x*random.uniform(.9,1.1)+random.uniform(-.03,.03)).clamp(0,1)
            return (x-image_mean)/image_std
        model.eval();features=[]
        with torch.no_grad():
            for a in range(0,len(rows),4):
                check()
                with torch.autocast('cuda',dtype=dtype):features.append(model(pixel_values=frames(list(range(a,min(a+4,len(rows)))))).float())
                emit(stage='features',processed=min(a+4,len(rows)),total=len(rows),message='Признаки для прогрева нового классификатора')
        features=torch.cat(features);mean=features[indices['train']].mean(0,keepdim=True);std=features[indices['train']].std(0,keepdim=True).clamp(min=.1)
        normalized=(features-mean)/std;train_targets=targets[indices['train']]
        class_weights=torch.where(train_targets>0,1/train_targets.sum(),1/(1-train_targets).sum())
        warm=torch.optim.AdamW(head.parameters(),lr=.001 if regularized else .003,weight_decay=.1)
        for step in range(warmup_steps):
            check();warm.zero_grad();logits=head(F.dropout(normalized[indices['train']],p=.5 if regularized else .35,training=True)).flatten()
            loss=(F.binary_cross_entropy_with_logits(logits,train_targets*(1-smoothing)+smoothing/2,reduction='none')*class_weights).sum()/2
            loss.backward();warm.step()
        with torch.no_grad():
            baseline_probs=head(normalized[indices['val']]).flatten().sigmoid().tolist()
        baseline=metrics(targets[indices['val']].int().tolist(),[int(p>=.5) for p in baseline_probs])
        atomic_json(run/'warmup_validation.json',dict(metrics=baseline,scope='organizer_windows',note='Frozen Large after train-only head warmup; not selected as fine-tuned model'))
        del features,normalized,warm
        # Discriminative layer rates: the earliest layers adapt less than the final block.
        groups=[]
        for name,param in model.named_parameters():
            param.requires_grad=True
            if name.startswith('model.blocks.'):
                layer=int(name.split('.')[2])+1
            elif name.startswith('model.patch_embed'):layer=0
            else:layer=depth+1
            groups.append(dict(params=[param],lr=backbone_lr*(.85**(depth+1-layer)),weight_decay=0 if param.ndim<=1 else .05))
        groups.append(dict(params=list(head.parameters()),lr=head_lr,weight_decay=.05))
        optimizer=torch.optim.AdamW(groups);initial_rates=[g['lr'] for g in groups]
        sample_weights=class_weights.cpu();best_score=-1;best_loss=float('inf');stale=0;best_result=None
        model.model.with_cp=True
        for epoch in range(1,args.epochs+1):
            check();model.train();head.train();optimizer.zero_grad(set_to_none=True)
            for g,rate in zip(optimizer.param_groups,initial_rates):g['lr']=rate*(.2+.8*.5*(1+math.cos(math.pi*(epoch-1)/args.epochs)))
            sampled=torch.multinomial(sample_weights,len(indices['train']),replacement=True).tolist();order=[indices['train'][i] for i in sampled]
            batches=[order[i:i+4] for i in range(0,len(order),4)];total_loss=0
            emit(stage='training',epoch=epoch,batch=0,batches=len(batches),message=f'Полное дообучение {depth} блоков {display}')
            for bi,ids in enumerate(batches):
                check()
                # Correct accumulation for the last, potentially shorter group.
                group=batches[(bi//2)*2:min((bi//2)*2+2,len(batches))];group_size=sum(len(b) for b in group)
                with torch.autocast('cuda',dtype=dtype):
                    embedding=model(pixel_values=frames(ids,True)).float();logits=head(F.dropout((embedding-mean)/std,p=dropout,training=True)).flatten()
                    loss=F.binary_cross_entropy_with_logits(logits,targets[ids]*(1-smoothing)+smoothing/2)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                (loss*len(ids)/group_size).backward();total_loss+=loss.item()*len(ids)
                if (bi+1)%2==0 or bi==len(batches)-1:
                    torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),1.0);optimizer.step();optimizer.zero_grad(set_to_none=True)
                emit(batch=bi+1,train_loss=total_loss/sum(len(b) for b in batches[:bi+1]),gpu_peak_gib=round(torch.cuda.max_memory_allocated()/1024**3,2))
            model.eval();head.eval();model.model.with_cp=False;probabilities=[];val_loss=0
            torch.backends.cuda.matmul.allow_tf32=False
            with torch.no_grad():
                for a in range(0,len(indices['val']),4):
                    check();ids=indices['val'][a:a+4]
                    logits=head((model(pixel_values=frames(ids)).float()-mean)/std).flatten()
                    val_loss+=F.binary_cross_entropy_with_logits(logits,targets[ids]).item()*len(ids);probabilities.extend(logits.sigmoid().float().tolist())
            model.model.with_cp=True
            torch.backends.cuda.matmul.allow_tf32=True
            result=dict(epoch=epoch,train_loss=total_loss/len(order),val_loss=val_loss/len(indices['val']),**metrics(targets[indices['val']].int().tolist(),[int(p>=.5) for p in probabilities]))
            state['history'].append(result);score=result['balanced_accuracy']
            if score>best_score or (score==best_score and result['val_loss']<best_loss):
                best_score=score;best_loss=result['val_loss'];stale=0;best_result=result
                emit(stage='saving',message='Сохранение лучшей большой модели')
                bundle=dict(schema_version=1,model_key=key,backbone={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    feature_mean=mean.detach().cpu(),feature_std=std.detach().cpu(),weight=head.weight.detach().cpu(),bias=head.bias.detach().cpu(),
                    metrics=result,preprocessing=settings['preprocessing'],manifest_sha256=manifest_hash,threshold=.5,training_mode='full_finetune',settings=settings)
                torch.save(bundle,run/'model.tmp.pt');os.replace(run/'model.tmp.pt',run/'model.pt');del bundle
                atomic_json(run/'validation.json',dict(metrics=result,scope='organizer_windows',predictions=[dict(video=rows[i]['video'],start_sec=rows[i]['start_sec'],target=int(rows[i]['target']),probability=p) for i,p in zip(indices['val'],probabilities)]))
            else:stale+=1
            emit(stage='training',metrics=result,best_metrics=best_result,best_epoch=best_result['epoch'])
            if stale>=5 and epoch>=6:break
        emit(status='completed',stage='completed',early_stopping=epoch<args.epochs,checkpoint_sha256=digest(run/'model.pt'),message='Полное дообучение завершено; ожидается копирование и проверка CPU')
    except (InterruptedError,TimeoutError) as exc:
        emit(status='stopped',stage='stopped',message=str(exc),checkpoint_sha256=digest(run/'model.pt') if (run/'model.pt').exists() else None)
    except Exception as exc:
        emit(status='failed',stage='failed',message=str(exc));raise

if __name__=='__main__':main()
