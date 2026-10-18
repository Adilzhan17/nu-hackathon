"""Import only verified complete chains from a local immutable T4 snapshot.
Never overwrite a completed chain or change an existing raw response. CPU only.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil

from ml.autolabel.dataset import digest
from ml.narration_training.prepare import write_json,write_jsonl
from .reconcile import audit_chain,validate_plan,verify_source


def import_completed(root,run,source):
    root=Path(root).resolve();run=Path(run).resolve();source=Path(source).resolve();blind=run/'blind'
    proof=verify_source(source);plan=json.loads((blind/'plan.json').read_text());validate_plan(root,plan)
    if json.loads((source/'plan.json').read_text())!=plan:raise ValueError('Changed inference plan')
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=run/'imports'/stamp;out.mkdir(parents=True)
    imported=[];audits=[]
    for v in plan['videos']:
        src=source/v['group_id'];dest=blind/v['group_id']
        if not (src/'complete.json').exists():continue
        if (dest/'complete.json').exists():
            # Original selected complete chains are immutable. The remote run must have skipped them.
            for p in dest.rglob('*'):
                if not p.is_file():continue
                if p.name=='reconciled_origin.json':continue
                q=src/p.relative_to(dest)
                if not q.exists() or digest(p)!=digest(q):raise ValueError('Completed local chain changed remotely: '+v['group_id']+'/'+str(p.relative_to(dest)))
            continue
        a=audit_chain(root,source,v,plan);audits.append(a)
        if not a['complete_valid']:continue
        if dest.exists():
            for p in dest.iterdir():
                if not p.is_file() or p.suffix!='.json' or p.name=='reconciled_origin.json':continue
                r=json.loads(p.read_text())
                if isinstance(r,dict) and 'raw_response' in r:
                    if not (src/p.name).exists() or digest(p)!=digest(src/p.name):raise ValueError('Cached response was replaced remotely')
        dest.mkdir(exist_ok=True)
        # Copy marker last, so interruption cannot expose a half-imported chain as complete.
        for p in src.iterdir():
            if p.name=='complete.json':continue
            if p.is_dir():shutil.copytree(p,dest/p.name,dirs_exist_ok=True)
            else:shutil.copy2(p,dest/p.name)
        shutil.copy2(src/'complete.json',dest/'complete.json');imported.append(v['video'])
    runtime=source/'runtime'
    if runtime.exists():
        target=blind/'runtime';target.mkdir(exist_ok=True)
        for p in runtime.glob('*.json'):
            if (target/p.name).exists() and digest(target/p.name)!=digest(p):raise ValueError('Runtime provenance changed')
            shutil.copy2(p,target/p.name)
    for name in ['model_provenance.json','general_rules.json','execution_status.json','batch_status.json']:
        if (source/name).exists():shutil.copy2(source/name,out/name)
    write_jsonl(out/'chain_audit.jsonl',audits)
    summary=dict(source=str(source),source_integrity=proof,imported=imported,imported_count=len(imported),
       completed_total=len(list(blind.glob('video_*/complete.json'))),remaining=len(plan['videos'])-len(list(blind.glob('video_*/complete.json'))),
       invalid_complete_chains=[a['video'] for a in audits if not a['complete_valid']],old_complete_chains_unchanged=True)
    write_json(out/'summary.json',summary);write_json(run/'completion_status.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='source_integrity'},ensure_ascii=False));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    a=p.parse_args();import_completed(a.root,a.run,a.source)
