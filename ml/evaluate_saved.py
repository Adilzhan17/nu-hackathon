"""Evaluate the exact downloaded bundle on CPU, using the unchanged validation windows."""
import argparse,csv,json,time,math
from pathlib import Path
from model_suite import MODELS,load_bundle,load_frames,predict
from train_x3d import atomic_json,metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=list(MODELS),required=True);parser.add_argument('--limit',type=int,default=0);args=parser.parse_args()
    import torch
    torch.set_num_threads(4)
    folder=Path('artifacts/models')/args.model;start=time.monotonic()
    extractor,bundle=load_bundle(folder/'model.pt',Path('artifacts/assets'),'cpu')
    load_seconds=time.monotonic()-start
    with Path('data/training/manifest.csv').open() as f:rows=[r for r in csv.DictReader(f) if r['split']=='val']
    import hashlib
    if hashlib.sha256(Path('data/training/manifest.csv').read_bytes()).hexdigest()!=bundle['manifest_sha256']:raise ValueError('Manifest changed')
    total_windows=len(rows)
    if args.limit>0:rows=rows[:args.limit]
    expected=json.loads((folder/'validation.json').read_text())['predictions'];lookup={(r['video'],float(r['start_sec'])):r['probability'] for r in expected}
    predictions=[];cfg=MODELS[args.model]
    for i,r in enumerate(rows):
        began=time.monotonic();x=load_frames(Path('data/raw/open')/r['video'],float(r['start_sec']),float(r['end_sec']),cfg['frames'],cfg['size'])
        p=predict(extractor,bundle,x.unsqueeze(0))[0]
        predictions.append(dict(video=r['video'],start_sec=r['start_sec'],target=int(r['target']),probability=p,elapsed_sec=time.monotonic()-began,gpu_probability=lookup[(r['video'],float(r['start_sec']))]))
        print(json.dumps(dict(model=args.model,processed=i+1,total=len(rows),elapsed_sec=predictions[-1]['elapsed_sec'])),flush=True)
    result=dict(device='CPU',threads=4,model_key=args.model,scope='organizer_windows' if len(rows)==total_windows else 'cpu_smoke_subset',checked_windows=len(rows),total_validation_windows=total_windows,load_seconds=load_seconds,total_seconds=time.monotonic()-start,
        mean_window_seconds=sum(r['elapsed_sec'] for r in predictions)/len(predictions),
        max_probability_difference=max(abs(r['probability']-r['gpu_probability']) for r in predictions),
        metrics=metrics([r['target'] for r in predictions],[int(r['probability']>=bundle['threshold']) for r in predictions]) if len(rows)==total_windows else None,predictions=predictions)
    from train_large import digest
    result['checkpoint_sha256']=digest(folder/'model.pt')
    if result['metrics'] is not None:
        result['metrics']['val_loss']=-sum(r['target']*math.log(max(1e-12,r['probability']))+(1-r['target'])*math.log(max(1e-12,1-r['probability'])) for r in predictions)/len(predictions)
    atomic_json(folder/'cpu_validation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='predictions'}),flush=True)

if __name__=='__main__':main()
