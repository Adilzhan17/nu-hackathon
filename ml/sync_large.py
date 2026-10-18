"""Read-only telemetry sync plus checksum-verified copies for the large-model run."""
import argparse,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
from suite_sync import SSH,REMOTE,ROOT
from train_x3d import atomic_json
KEY='videomaev2_large'

def copy(remote,target):
    part=target.with_suffix(target.suffix+'.part')
    with part.open('wb') as f:subprocess.run(SSH+['cat '+shlex.quote(str(remote))],stdout=f,stderr=subprocess.PIPE,timeout=300,check=True)
    return part

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model-key',choices=['videomaev2_large','videomaev2_giant'],default=KEY);parser.add_argument('--run',choices=['large_v1','giant_v1'],default='large_v1');args=parser.parse_args();key=args.model_key;remote_run=REMOTE+'/runs/'+args.run
    status_dir=ROOT/'runs/suite'/key;status_dir.mkdir(parents=True,exist_ok=True)
    local=ROOT/'artifacts/models'/key;local.mkdir(parents=True,exist_ok=True)
    began=time.monotonic()
    while time.monotonic()-began<2400:
        try:
            script="import json;from pathlib import Path;p=Path("+repr(remote_run)+");print(json.dumps({n:json.loads((p/n).read_text()) for n in ['status.json','settings.json','warmup_validation.json'] if (p/n).exists()}))"
            r=subprocess.run(SSH+['python3 -c '+shlex.quote(script)],capture_output=True,text=True,timeout=25,check=True)
            data=json.loads(r.stdout);state=data.get('status.json',{})
            if state:
                if data.get('warmup_validation.json'):state['warmup_metrics']=data['warmup_validation.json']['metrics']
                atomic_json(status_dir/'status.json',state)
            if 'settings.json' in data:atomic_json(local/'settings.json',data['settings.json'])
            if state.get('status') in ('completed','stopped','failed'):
                sha=state.get('checkpoint_sha256')
                if sha:
                    modelpart=copy(remote_run+'/model.pt',local/'model.pt')
                    h=hashlib.sha256()
                    with modelpart.open('rb') as f:
                        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
                    if h.hexdigest()!=sha:raise ValueError('Large checkpoint checksum mismatch')
                    validation=copy(remote_run+'/validation.json',local/'validation.json')
                    os.replace(validation,local/'validation.json');os.replace(modelpart,local/'model.pt')
                    atomic_json(local/'receipt.json',dict(sha256=sha,synced_at=time.time()))
                atomic_json(status_dir/'sync.json',dict(done=True,updated_at=time.time(),connected=True))
                break
            atomic_json(status_dir/'sync.json',dict(done=False,updated_at=time.time(),connected=True))
        except Exception as exc:atomic_json(status_dir/'sync.json',dict(updated_at=time.time(),connected=False,error=str(exc)))
        time.sleep(6)

if __name__=='__main__':main()
