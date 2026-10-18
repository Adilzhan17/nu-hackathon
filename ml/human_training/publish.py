"""Atomically attach retraining status to the live, read-only readiness report."""
import json,os,tempfile
from pathlib import Path

def publish(root, status):
    dest=root/'runs/readiness/report.json'
    data=json.loads(dest.read_text());data['human_retraining']=status
    fd,path=tempfile.mkstemp(prefix='.human-status-',dir=dest.parent)
    with os.fdopen(fd,'w') as f:
        json.dump(data,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(path,dest)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('status',type=Path);a=p.parse_args();publish(Path.cwd(),json.loads(a.status.read_text()))
