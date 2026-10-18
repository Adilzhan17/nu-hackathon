"""Loopback-only independent Gemma lab. No mutations in human-review or Forte X data."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit,parse_qs
import uuid

from forte_vision.infrastructure.gpu_transport import atomic,process,probe
from forte_vision.application.analysis import MAX_DURATION
from forte_vision.domain.regulations import attach_regulatory_sources

MAX_BYTES=64*1024*1024


def make_handler(root):
    directory=root/'runs/gemma_lab/uploads';directory.mkdir(parents=True,exist_ok=True)
    active=threading.Lock()
    # A service restart cannot silently resume a paid GPU job.
    for file in directory.glob('*/job.json'):
        old=json.loads(file.read_text())
        if old['status'] in ('processing','uploading'):
            old.update(status='interrupted',message='Сервис перезапущен; проверьте GPU перед повтором');atomic(file,old)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def json(self,value,status=200):
            raw=json.dumps(value,ensure_ascii=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
        def file(self,path,content_type):
            if not path.is_file():return self.json({'error':'Файл пока не готов'},404)
            size=path.stat().st_size;start=0;end=size-1
            header=self.headers.get('Range','')
            if header:
                match=re.fullmatch(r'bytes=(\d+)-(\d*)',header)
                if not match:return self.json({'error':'Invalid range'},416)
                start=int(match[1]);end=min(end,int(match[2])) if match[2] else end
                if start>end:return self.json({'error':'Invalid range'},416)
            self.send_response(206 if header else 200);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes')
            self.send_header('Cache-Control','private, no-store')
            if header:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            with path.open('rb') as f:
                f.seek(start);left=end-start+1
                while left:
                    chunk=f.read(min(left,256*1024))
                    if not chunk:break
                    self.wfile.write(chunk);left-=len(chunk)
        def do_GET(self):
            path=urlsplit(self.path).path
            if path=='/':return self.file(root/'frontend/analysis/index.html','text/html; charset=utf-8')
            if path=='/status':
                recent=[]
                for p in sorted(directory.glob('*/job.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:10]:
                    j=json.loads(p.read_text())
                    if j.get('profile')=='jury_api':continue
                    recent.append({k:j.get(k) for k in ('id','name','status','created_at')})
                return self.json(dict(busy=active.locked(),mode='local_gemma',sample_step_sec=.5,max_duration_sec=MAX_DURATION,
                    max_bytes=MAX_BYTES,max_seconds=900,model='Gemma 4 12B QAT Q4_0',recent=recent))
            match=re.fullmatch(r'/jobs/([a-f0-9]{32})(?:/(media|frames/[fd]\d{5}\.jpg))?',path)
            if not match:return self.json({'error':'Not found'},404)
            folder=directory/match[1]
            if not (folder/'job.json').exists():return self.json({'error':'Not found'},404)
            state=json.loads((folder/'job.json').read_text())
            # Jury footage is only available through the authenticated jury API.
            if state.get('profile')=='jury_api':return self.json({'error':'Not found'},404)
            if match[2]=='media':return self.file(folder/'preview.mp4','video/mp4')
            if match[2]:return self.file(folder/match[2],'image/jpeg')
            return self.json(attach_regulatory_sources(state))
        def do_POST(self):
            if urlsplit(self.path).path!='/jobs':return self.json({'error':'Not found'},404)
            # Only a same-origin checked Next route can initiate the loopback worker.
            if self.headers.get('X-Gemma-Local-Proxy')!='1':return self.json({'error':'Proxy required'},403)
            length=self.headers.get('Content-Length','')
            if not length.isdecimal() or not 0<int(length)<=MAX_BYTES:return self.json({'error':'Максимум 64 MB'},413)
            name=parse_qs(urlsplit(self.path).query).get('name',['video.mp4'])[0]
            if len(name)>180 or Path(name).suffix.lower() not in ('.mp4','.mov'):return self.json({'error':'Нужен MP4 или MOV'},422)
            if not active.acquire(blocking=False):return self.json({'error':'Gemma уже обрабатывает видео; дождитесь завершения'},409)
            folder=directory/uuid.uuid4().hex;folder.mkdir(mode=0o700)
            state=dict(id=folder.name,name=Path(name).name,status='uploading',created_at=time.time(),
                mode='gemma_lab',result=None,review_status='proposed',ground_truth=False)
            jury_profile=self.headers.get('X-Gemma-Profile')=='jury'
            seconds=270 if jury_profile else 900
            overview_step=None if jury_profile else .5
            state.update(profile='jury_api' if jury_profile else 'web_full',max_seconds=seconds,
                         sampling_profile='adaptive_max_18_overview_frames' if jury_profile else 'full_half_second')
            def emit(**values):
                state.update(updated_at=time.time(),**values);atomic(folder/'job.json',state)
            try:
                self.connection.settimeout(60)
                with (folder/'source').open('xb') as stream:
                    left=int(length)
                    while left:
                        chunk=self.rfile.read(min(left,256*1024))
                        if not chunk:raise ValueError('Upload interrupted')
                        stream.write(chunk);left-=len(chunk)
                state.update(probe(folder/'source'))
                emit(status='processing',stage='queued',message='Видео получено')
            except Exception as e:
                emit(status='failed',stage='validation',message=str(e));active.release();return self.json({'error':str(e),'job_id':folder.name,'analysis_started':False},422)
            def work():
                try:process(folder,emit,seconds=seconds,overview_step=overview_step)
                except Exception as e:emit(status='failed',message=str(e))
                finally:active.release()
            threading.Thread(target=work,daemon=True).start()
            return self.json(state,202)
    return Handler


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8771);p.add_argument('--root',type=Path,default=Path.cwd())
    a=p.parse_args();ThreadingHTTPServer(('127.0.0.1',a.port),make_handler(a.root.resolve())).serve_forever()


if __name__ == '__main__':
    main()
