"""Exercise the real local upload and polling API without external AI services."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

def submit(source,models):
    base='http://127.0.0.1:8765'
    def call(path,method='GET',body=None,headers=None):
        request=urllib.request.Request(base+path,data=body,method=method,headers={'Origin':base,**(headers or {})})
        with urllib.request.urlopen(request,timeout=40) as response:return json.load(response)
    def post(path,value):return call(path,'POST',json.dumps(value).encode(),{'Content-Type':'application/json'})
    state=post('/api/uploads',dict(name=source.name,size=source.stat().st_size,models=models));jid=state['id']
    with source.open('rb') as f:
        offset=0
        while chunk:=f.read(8*1024*1024):
            call('/api/uploads/'+jid,'PUT',chunk,{'Content-Type':'application/octet-stream','X-Upload-Offset':str(offset)});offset+=len(chunk)
    post('/api/uploads/'+jid+'/complete',{});return jid,call

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--models',nargs='+',default=['videomaev2','yolo11']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    jid,call=submit(a.source,a.models);print(json.dumps(dict(job_id=jid,url='/analyze?job='+jid)),flush=True);began=time.monotonic()
    while time.monotonic()-began<310:
        s=call('/api/uploads/'+jid)
        if s['status'] in ['completed','failed']:
            a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(s,ensure_ascii=False,indent=2));print(json.dumps(dict(job_id=jid,status=s['status'],message=s.get('message'),pipeline_status=s.get('pipeline',{}).get('status')),ensure_ascii=False));break
        time.sleep(2)
    else:raise TimeoutError('Site polling budget exhausted')
