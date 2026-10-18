"""X3D-S transfer learning with file-based telemetry and CPU-loadable checkpoints.

Run from project root: python ml/train_x3d.py --run-dir runs/RUN_ID
Validation is on organizer-labelled windows, not whole-video event detection.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import signal
import time
import traceback
from pathlib import Path


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    os.replace(temp, path)


def metrics(targets, predicted):
    tp = sum(y == 1 and p == 1 for y, p in zip(targets, predicted))
    tn = sum(y == 0 and p == 0 for y, p in zip(targets, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(targets, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(targets, predicted))
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return dict(precision=precision, recall=recall, f1=2*precision*recall/max(1e-12,precision+recall),
                balanced_accuracy=(recall + tn/max(1,tn+fp))/2,
                confusion=dict(tp=tp,tn=tn,fp=fp,fn=fn))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--max-seconds', type=int, default=7200)
    args = parser.parse_args()
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    began = time.time()
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    state = dict(status='running', stage='environment', epoch=0, epochs=args.epochs,
                 started_at=began, history=[], model='X3D-S', validation_scope='organizer_windows')
    def emit(stage, **values):
        state.update(stage=stage, updated_at=time.time(), elapsed_sec=time.time()-began, **values)
        atomic_json(run/'status.json', state)
        print(json.dumps(dict(stage=stage, **values), ensure_ascii=False), flush=True)
    def check_stop():
        if stopped or (run/'STOP').exists() or time.time()-began > args.max_seconds:
            raise InterruptedError('Обучение остановлено; доступные веса сохранены. Аренда VM продолжается.')
    try:
        emit('environment', message='Проверка CUDA и загрузка X3D-S')
        import av
        import numpy as np
        import torch
        import torch.nn.functional as F
        from pytorchvideo.models.hub import x3d_s
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.set_num_threads(8)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA недоступна. Обучение на CPU автоматически не запускается.')
        device = torch.device('cuda')
        # Real forward/backward catches unsupported GPU architectures before video preprocessing.
        probe = torch.nn.Conv3d(3, 3, 3, groups=3).to(device)
        probe(torch.randn(1,3,5,16,16,device=device)).sum().backward()
        del probe
        emit('environment', gpu_name=torch.cuda.get_device_name(), torch_version=torch.__version__)
        model = x3d_s(pretrained=True)
        head = model.blocks[-1]
        head.proj = torch.nn.Linear(head.proj.in_features, 2)
        head.activation = None
        model.to(device)
        manifest = Path('data/training/manifest.csv')
        with manifest.open() as f:
            rows = list(csv.DictReader(f))
        # Verify uploaded content and guard against accidentally edited split files.
        seen, split_hashes = {}, {'train':set(), 'val':set()}
        for row in rows:
            path = (Path('data/raw/open')/row['video']).resolve()
            if not path.is_relative_to(Path('data/raw/open').resolve()):
                raise ValueError('Unsafe dataset path')
            if str(path) not in seen:
                seen[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            if seen[str(path)] != row['sha256']:
                raise ValueError('Video checksum mismatch')
            split_hashes[row['split']].add(row['sha256'])
        if split_hashes['train'] & split_hashes['val']:
            raise ValueError('Train/validation leakage')
        tensors = []
        for index, row in enumerate(rows):
            check_stop()
            start, end = float(row['start_sec']), float(row['end_sec'])
            timestamps = np.linspace(start, max(start, end-.08), 13)
            frames = []
            with av.open(str(Path('data/raw/open')/row['video'])) as container:
                stream = container.streams.video[0]
                stream.thread_type = 'AUTO'
                container.seek(int(start/float(stream.time_base)), stream=stream, backward=True)
                for frame in container.decode(stream):
                    if frame.time is None:
                        continue
                    while len(frames) < 13 and frame.time >= timestamps[len(frames)]:
                        # Letterbox the whole cashier scene; center-cropping portrait videos loses hands/desk.
                        scale = 160/max(frame.width,frame.height)
                        image = frame.reformat(width=max(1,round(frame.width*scale)),
                                               height=max(1,round(frame.height*scale)),format='rgb24').to_ndarray()
                        canvas = np.full((160,160,3),114,dtype=np.uint8)
                        y,x=(160-image.shape[0])//2,(160-image.shape[1])//2
                        canvas[y:y+image.shape[0],x:x+image.shape[1]]=image
                        frames.append(canvas)
                    if len(frames)==13:
                        break
            if not frames:
                raise ValueError(f'No decoded frames: {row["video"]}')
            frames.extend([frames[-1]]*(13-len(frames)))
            tensors.append(torch.from_numpy(np.stack(frames)).permute(3,0,1,2).contiguous())
            emit('preprocessing', processed=index+1, total=len(rows), message='Декодирование обучающих окон')

        class Clips(Dataset):
            def __init__(self, split):
                self.indices = [i for i,r in enumerate(rows) if r['split']==split]
                self.training = split=='train'
            def __len__(self): return len(self.indices)
            def __getitem__(self, position):
                i = self.indices[position]
                x = tensors[i].float()/255
                if self.training:
                    x = (x*random.uniform(.9,1.1)+random.uniform(-.03,.03)).clamp(0,1)
                return (x-.45)/.225, int(rows[i]['target'])
        train, val = Clips('train'), Clips('val')
        counts = [sum(int(rows[i]['target'])==label for i in train.indices) for label in (0,1)]
        if min(counts)==0 or {int(rows[i]['target']) for i in val.indices}!={0,1}:
            raise ValueError('Each split must contain both classes')
        sampler = WeightedRandomSampler([1/counts[int(rows[i]['target'])] for i in train.indices],
                                        len(train), replacement=True)
        loader = DataLoader(train,batch_size=8,sampler=sampler,num_workers=0)
        validation = DataLoader(val,batch_size=8,shuffle=False,num_workers=0)
        # Frozen head warm-up, then only the last residual stage and projection head.
        for p in model.parameters(): p.requires_grad=False
        for p in head.proj.parameters(): p.requires_grad=True
        optimizer = torch.optim.AdamW(head.proj.parameters(),lr=1e-3,weight_decay=.01)
        best, best_epoch, stale = -1, 0, 0
        for epoch in range(1,args.epochs+1):
            check_stop()
            if epoch==4:
                for block in (model.blocks[-2],model.blocks[-1]):
                    for p in block.parameters(): p.requires_grad=True
                optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4,weight_decay=.01)
            model.eval()  # Keep pretrained batch-normalization statistics stable on the tiny dataset.
            head.dropout.train()
            total_loss=0
            emit('training',epoch=epoch,batch=0,batches=len(loader),message='Обучение классификатора' if epoch<4 else 'Дообучение последнего блока')
            for batch,(x,y) in enumerate(loader,1):
                check_stop()
                x,y=x.to(device),y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(x),y)
                if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1)
                optimizer.step()
                total_loss+=loss.item()*len(y)
                emit('training',batch=batch,train_loss=total_loss/min(batch*8,len(train)))
            model.eval()
            truth,prediction,probabilities=[],[],[]
            val_loss=0
            emit('validation',message='Проверка на отложенных видео')
            with torch.inference_mode():
                for x,y in validation:
                    check_stop()
                    logits=model(x.to(device))
                    val_loss+=F.cross_entropy(logits,y.to(device)).item()*len(y)
                    truth.extend(y.tolist())
                    prediction.extend(logits.argmax(1).cpu().tolist())
                    probabilities.extend(logits.softmax(1)[:,1].cpu().tolist())
            result=dict(epoch=epoch,train_loss=total_loss/len(train),val_loss=val_loss/len(val),
                        **metrics(truth,prediction))
            state['history'].append(result)
            score=result['balanced_accuracy']
            if score>best:
                best,best_epoch,stale=score,epoch,0
                checkpoint=dict(model='x3d_s',state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                                classes=['normal','violation_unspecified'],epoch=epoch,metrics=result,
                                preprocessing=dict(frames=13,size=160,window_sec=8,resize='letterbox',pad_value=114,mean=.45,std=.225),
                                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),threshold=.5)
                torch.save(checkpoint,run/'best.tmp.pt')
                os.replace(run/'best.tmp.pt',run/'best.pt')
                atomic_json(run/'validation.json',dict(metrics=result,scope='organizer_windows',
                    predictions=[dict(video=rows[i]['video'],start_sec=rows[i]['start_sec'],target=y,probability=p)
                                 for i,y,p in zip(val.indices,truth,probabilities)]))
            else:
                stale+=1
            emit('training',metrics=result,best_epoch=best_epoch,checkpoint='best.pt',checkpoint_epoch=best_epoch)
            if stale>=7 and epoch>=8: break
        emit('completed',status='completed',early_stopping=epoch<args.epochs,
             message=('Ранняя остановка: качество проверки перестало улучшаться. ' if epoch<args.epochs else 'Обучение завершено. ')
                     + 'Требуется проверка на полных видео.')
    except InterruptedError as exc:
        emit('stopped',status='stopped',message=str(exc))
    except Exception as exc:
        emit('failed',status='failed',message=str(exc))
        traceback.print_exc()
        raise


if __name__=='__main__':
    main()
