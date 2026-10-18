"""Copy immutable completed chains from T4; optionally include partial chains after exit.
Only the configured T4 is contacted. No private key contents are read by this program.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HOST='ubuntu@global.prd.ga.run.brev.nvidia.com'
REMOTE='/home/ubuntu/nuhack-training'
SSH=['ssh','-i','/root/.ssh/nuhack_brev','-p','15171','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=10',HOST]


def capture(output,run='runs/narration_training/organizer_t4/blind',include_partial=False):
    out=Path(output).resolve()
    if out.exists():raise ValueError('Snapshot output must be new')
    if not run.startswith('runs/narration_training/') or '..' in Path(run).parts:raise ValueError('Invalid remote run')
    # A complete marker is written last and skipped on resume. Hash both before and after copying.
    script='''import pathlib,json,hashlib,subprocess
root=pathlib.Path(%r);run=root/%r
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
workers=[x for x in processes if ('python -u ml/migration/t4_organizer.py' in x or 'python -u -m ml.narration_training.organizer.resume' in x) and not ('bash -c' in x or 'timeout ' in x or 'python -c' in x)]
partial=%r
if partial and workers:raise RuntimeError('Worker still active; partial snapshot refused')
folders=[p for p in run.glob('video_*') if p.is_dir() and (partial or (p/'complete.json').exists())]
files=[p for p in run.iterdir() if p.is_file() and p.suffix=='.json']
files += [p for d in folders for p in d.rglob('*') if p.is_file()]
files += list((run/'runtime').glob('*.json'))
code=list((root/'ml/narration_training/organizer').glob('*.py'))
code += [root/'ml/autolabel/sampling.py',root/'ml/autolabel/dataset.py',root/'ml/narration_training/prepare.py']
code += [root/'ml/migration/t4_organizer.py']
files += code
if not workers:files += list(run.parent.glob('resume*.log'))
result={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
print(json.dumps(dict(files=result,active_workers=workers,groups=[p.name for p in folders],partial_included=partial)))
''' % (REMOTE,run,include_partial)
    def inventory():
        cmd='cd '+shlex.quote(REMOTE)+' && .venv-scenario/bin/python -c '+shlex.quote(script)
        return json.loads(subprocess.check_output(SSH+[cmd],text=True))
    before=inventory();out.mkdir(parents=True)
    names=out/'transfer_files.txt';names.write_text(''.join(p+'\n' for p in before['files']))
    subprocess.run(['rsync','-az','--files-from='+str(names),'-e',shlex.join(SSH[:-1]),HOST+':'+REMOTE+'/',str(out)+'/'],check=True)
    after=inventory();mismatches=[]
    for name,sha in before['files'].items():
        if hashlib.sha256((out/name).read_bytes()).hexdigest()!=sha or after['files'].get(name)!=sha:mismatches.append(name)
    result=dict(before,captured_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),remote_run=run,
                server='outdoor-orange-whippet',port=15171,verified_file_count=len(before['files']),changed_during_copy=mismatches)
    (out/'snapshot_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if mismatches:raise RuntimeError('Snapshot changed during copy; excluded from reconciliation')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--run',default='runs/narration_training/organizer_t4/blind');p.add_argument('--include-partial',action='store_true')
    a=p.parse_args();capture(a.output,a.run,a.include_partial)
