"""One bounded read-only sample. Never starts GPU work or reads credentials."""
import json
import os
from pathlib import Path
import subprocess
import time

SSH = ['ssh', '-i', '/root/.ssh/nuhack_brev', '-p', '15171', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=4', 'ubuntu@global.prd.ga.run.brev.nvidia.com']

def host_sample():
    def ticks():
        values=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
        return sum(values),values[3]+values[4]
    a=ticks();time.sleep(.15);b=ticks()
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    disk=os.statvfs('.')
    return dict(sampled_at=time.time(),cpu_percent=round(100*(1-(b[1]-a[1])/max(1,b[0]-a[0])),1),
        cpu_count=os.cpu_count(),load_average=list(os.getloadavg()),ram_total=mem['MemTotal'],
        ram_used=mem['MemTotal']-mem['MemAvailable'],swap_used=mem['SwapTotal']-mem['SwapFree'],
        disk_free=disk.f_bavail*disk.f_frsize,disk_total=disk.f_blocks*disk.f_frsize)

def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default

def remote_sample():
    # Only numeric resource values and explicitly selected progress fields cross SSH.
    import inspect,shlex
    code='import json,os,time,subprocess\nfrom pathlib import Path\n'+inspect.getsource(host_sample)+'''
result=host_sample()
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
result['gpu']=[]
for line in gpu.stdout.splitlines():
    v=[s.strip() for s in line.split(',')]
    if len(v)==6:result['gpu'].append(dict(name=v[0],utilization=float(v[1]),memory_used_mib=float(v[2]),memory_total_mib=float(v[3]),temperature=float(v[4]),power_watts=float(v[5])))
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
result['tasks']=[]
for line in processes.splitlines():
    if 'python -c' in line or 'bash -c' in line or 'timeout ' in line:continue
    for needle,kind in [('ml.narration_training.organizer.resume','context_annotation'),('ml.context_pipeline.train_detector','detector_training'),('ml.context_pipeline.worker','uploaded_video_analysis'),('ml.human_training','human_training'),('ml/train_suite.py','model_training')]:
        if needle in line:result['tasks'].append(dict(pid=int(line.split()[0]),kind=kind));break
run=Path('runs/narration_training/organizer_reconciled_v1/blind')
result['organizer_completed']=len(list(run.glob('video_*/complete.json')))
status=Path('runs/context_pipeline/detector_training_v2/status.json')
if status.exists():
    try:
        value=json.loads(status.read_text());result['detector_training']={k:value[k] for k in ['status','stage','epoch','epochs','elapsed_sec','updated_at','error'] if k in value}
    except ValueError:pass
log=run.parent/'resume.log';result['last_progress']=None
if log.exists():
    for line in reversed(log.read_text()[-30000:].splitlines()):
        try:
            row=json.loads(line)
            if 'stage' in row:result['last_progress']={k:row[k] for k in ['video','stage','window','total_windows','elapsed'] if k in row};break
        except ValueError:pass
print(json.dumps(result))
'''
    p=subprocess.run(SSH+['cd /home/ubuntu/nuhack-training && .venv-scenario/bin/python -c '+shlex.quote(code)],
        capture_output=True,text=True,timeout=12)
    if p.returncode:raise RuntimeError('T4 не ответил на запрос мониторинга')
    return json.loads(p.stdout)

def collect(root):
    root=Path(root);out=dict(sampled_at=time.time(),main=host_sample())
    try:out['t4']=dict(status='online',**remote_sample())
    except (OSError,ValueError,subprocess.SubprocessError,RuntimeError):out['t4']=dict(status='unavailable',message='Нет свежего ответа T4; значения не подменены нулями.')
    run=root/'runs/narration_training/organizer_reconciled_v1'
    out['organizer']=read(run/'completion_status.json',{})
    out['organizer']={k:out['organizer'].get(k) for k in ['completed_total','remaining']}
    reports=sorted(run.glob('report*/summary.json'),key=lambda p:p.stat().st_mtime)
    out['report']=read(reports[-1],{}) if reports else {}
    batches=sorted((root/'runs/human_training').glob('*/status.json'),key=lambda p:p.stat().st_mtime)
    batch=read(batches[-1],{}) if batches else {}
    out['training']={k:batch.get(k) for k in ['run_id','status','summary','message']}
    out['videomae']={k:read(root/'artifacts/models/videomaev2/settings.json',{}).get(k) for k in ['training','trainable_parameters','manifest_sha256']}
    action=read(root/'runs/context_pipeline/action_training_v1/status.json',{})
    out['action_training']={k:action.get(k) for k in ['status','stage','processed','total','completed_groups','total_groups','used','groups','baseline_top1','trained_top1','trainable_parameters','error','updated_at','elapsed_sec']}
    out['detector_dataset']=read(root/'runs/context_pipeline/detector_dataset_v3/summary.json',{})
    out['detector_registry']=read(root/'runs/context_pipeline/detector_active.json',{})
    decision=read(root/'runs/context_pipeline/deployment_decision.json',{})
    out['detector_decision']={k:decision.get(k) for k in ['candidate_activated','reason','candidate_sha256']}
    out['detector_evaluation']=read(root/'runs/context_pipeline/detector_evaluation_v3/report.json',{})
    test=read(root/'runs/forte_x/production_browser_test.json',{})
    out['forte_x_test']={k:test.get(k) for k in ['status','job_id','checked_at','wall_seconds','observations','evidence_images','assertions','error']}
    questions=root/'runs/context_pipeline/train_review_questions.jsonl'
    out['detector_questions']=[json.loads(line) for line in questions.read_text().splitlines()] if questions.exists() else []
    out['cases']=[]
    if reports:
        scenarios=reports[-1].parent/'worked_scenarios.jsonl'
        for line in scenarios.read_text().splitlines() if scenarios.exists() else []:
            s=json.loads(line)
            out['cases'].append(dict(video=s['video'],split=s['split'],observations=s['visual_proposals'],rules=s['rule_assessments']))
    return out

def main():
    import argparse
    from .publish import publish
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());a=p.parse_args()
    publish(a.root,collect(a.root))

if __name__=='__main__':main()
