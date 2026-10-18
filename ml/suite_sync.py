"""Pull real GPU telemetry and verified model bundles; survives page reloads."""
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from model_suite import MODELS
from train_x3d import atomic_json

ROOT=Path(__file__).resolve().parent.parent
REMOTE='/home/ubuntu/nuhack-training'
SSH=['ssh','-i','/root/.ssh/nuhack_brev','-p','21453','-o','BatchMode=yes','-o','IdentitiesOnly=yes',
     '-o','ConnectTimeout=8','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
     'ubuntu@global.prd.ga.run.brev.nvidia.com']


def main():
    folder=ROOT/'runs/suite';folder.mkdir(parents=True,exist_ok=True)
    while True:
        try:
            script="import json; from pathlib import Path; p=Path("+repr(REMOTE+'/runs/suite')+"); print(json.dumps({'done':(p/'DONE').exists(),'models':{f.parent.name:json.loads(f.read_text()) for f in p.glob('*/status.json')}}))"
            result=subprocess.run(SSH+['python3 -c '+shlex.quote(script)],capture_output=True,text=True,timeout=30,check=True)
            data=json.loads(result.stdout)
            for key,state in data['models'].items():
                if key not in MODELS:continue
                target=folder/key;target.mkdir(exist_ok=True)
                atomic_json(target/'status.json',state)
                artifact=ROOT/'artifacts/models'/key;artifact.mkdir(parents=True,exist_ok=True)
                digest=state.get('checkpoint_sha256')
                receipt=artifact/'receipt.json'
                synced=json.loads(receipt.read_text()).get('sha256') if receipt.exists() else None
                if state['status']=='completed' and synced!=digest:
                    for name in ('model.pt','validation.json'):
                        temp=artifact/(name+'.part')
                        with temp.open('wb') as f:
                            subprocess.run(SSH+['cat '+shlex.quote(REMOTE+'/runs/suite/'+key+'/'+name)],stdout=f,stderr=subprocess.PIPE,timeout=180,check=True)
                    if hashlib.sha256((artifact/'model.pt.part').read_bytes()).hexdigest()!=digest:raise ValueError('Checkpoint checksum mismatch')
                    os.replace(artifact/'validation.json.part',artifact/'validation.json')
                    os.replace(artifact/'model.pt.part',artifact/'model.pt')
                    atomic_json(receipt,dict(sha256=digest,synced_at=time.time()))
            atomic_json(folder/'sync.json',dict(updated_at=time.time(),connected=True,done=data['done']))
            if data['done']:break
        except Exception as exc:
            atomic_json(folder/'sync.json',dict(updated_at=time.time(),connected=False,error=str(exc)))
        time.sleep(8)


if __name__=='__main__':main()
