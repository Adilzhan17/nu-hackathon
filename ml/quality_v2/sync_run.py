"""Bounded synchronization of feature extraction; training heads runs on local CPU."""
import argparse,json,subprocess,sys,time,shlex
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from suite_sync import SSH,REMOTE,ROOT
from train_x3d import atomic_json

parser=argparse.ArgumentParser();parser.add_argument('--semantic',action='store_true');args=parser.parse_args()
run=ROOT/'runs/quality_v2';start=time.time();status_name='semantic_status.json' if args.semantic else 'status.json'
while time.time()-start<2400:
    r=subprocess.run(SSH+['cat '+shlex.quote(REMOTE+'/runs/quality_v2/'+status_name)],capture_output=True,text=True,timeout=30)
    if r.returncode==0:
        state=json.loads(r.stdout);atomic_json(run/status_name,state)
        if state.get('status')=='features_ready':
            for name in (['semantic.npz','semantic_prompts.json'] if args.semantic else ['action.npz','dino.npz','ssl.npz']):
                with (run/name).open('wb') as out:subprocess.run(SSH+['cat '+shlex.quote(REMOTE+'/runs/quality_v2/'+name)],stdout=out,check=True,timeout=60)
            script='semantic.py' if args.semantic else 'experiment.py'
            result=subprocess.run([sys.executable,'-u',str(ROOT/'ml/quality_v2'/script),'train'],cwd=ROOT)
            raise SystemExit(result.returncode)
        if state.get('status')=='failed':raise SystemExit(state)
    time.sleep(8)
raise SystemExit('Feature synchronization timed out')
