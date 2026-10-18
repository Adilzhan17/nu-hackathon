"""Atomic HTML publication: no restart of the annotation server or Next.js."""
import json
import os
from pathlib import Path
import tempfile

def atomic(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name)
    try:
        with os.fdopen(fd,'w') as f:f.write(text)
        os.chmod(temp,0o644);os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)

def publish(root,state):
    root=Path(root)
    payload=json.dumps(state,ensure_ascii=False).replace('<','\\u003c')
    template=(root/'ml/live_console/page.html').read_text()
    atomic(root/'runs/live_console/status.json',payload+'\n')
    atomic(root/'ml/scenario.html',template.replace('STATE_JSON',payload))
