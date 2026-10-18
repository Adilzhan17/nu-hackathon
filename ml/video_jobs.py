"""Bounded chunk uploads and serial CPU jobs; separate from the review dataset."""
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from model_suite import MODELS
from train_x3d import atomic_json

MAX_SIZE=512*1024*1024
CHUNK_SIZE=8*1024*1024
TOTAL_SIZE=3*1024*1024*1024
# All workers in this process share the CPU budget, including private jury jobs.
INFERENCE_LOCK=threading.Lock()
JOB_TIMEOUT=300


class VideoJobs:
    def __init__(self,root,start_worker=True,directory=None,job_timeout=JOB_TIMEOUT,private=False):
        self.root=Path(root);self.directory=Path(directory) if directory is not None else self.root/'data/uploads'
        self.directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.job_timeout=job_timeout;self.private=private
        if private:self.directory.chmod(0o700)
        self.lock=threading.RLock();self.tasks=queue.Queue()
        for path in self.directory.glob('*/job.json'):
            try:
                state=json.loads(path.read_text())
                if state['status'] in ('queued','processing'):
                    if self.remaining(state)<=0:
                        state.update(status='failed',stage='failed',error_code='deadline_exceeded',message='Истёк общий лимит 300 секунд',updated_at=time.time())
                        atomic_json(path,state)
                    else:
                        state.update(status='queued',message='Возобновление после перезапуска сервера')
                        atomic_json(path,state);self.tasks.put(path.parent.name)
            except (ValueError,KeyError):continue
        if start_worker:threading.Thread(target=self.worker,daemon=True).start()

    def models(self):
        items=[]
        for key,cfg in MODELS.items():
            status=self.root/'runs/suite'/key/'status.json'
            data=json.loads(status.read_text()) if status.exists() else dict(status='queued',stage='queued',message='Ожидает обучения')
            ready=(self.root/'artifacts/models'/key/'model.pt').is_file()
            item=dict(cfg,key=key,training=data,ready=ready)
            validation=self.root/'artifacts/models'/key/'validation.json'
            if ready and validation.exists():
                saved=json.loads(validation.read_text())
                item['metrics']=saved['metrics'];item['metrics_device']='GPU'
                cpu_path=validation.with_name('cpu_validation.json')
                cpu=json.loads(cpu_path.read_text()) if cpu_path.exists() else {}
                expected={(p['video'],float(p['start_sec']),p['target']) for p in saved.get('predictions',[])}
                checked={(p['video'],float(p['start_sec']),p['target']) for p in cpu.get('predictions',[])}
                if cpu.get('scope')=='organizer_windows' and cpu.get('metrics') and expected and checked==expected and cpu.get('checkpoint_sha256')==data.get('checkpoint_sha256') and cpu.get('checkpoint_sha256'):
                    item['gpu_metrics']=saved['metrics']
                    item['metrics']=dict(cpu['metrics'],epoch=saved['metrics'].get('epoch'))
                    item['metrics_device']='CPU'
            items.append(item)
        return items

    def folder(self,jid):
        if not isinstance(jid,str) or not re.fullmatch('[a-f0-9]{32}',jid):raise ValueError('Неизвестное видео')
        path=self.directory/jid
        if not (path/'job.json').is_file():raise FileNotFoundError('Видео не найдено')
        return path

    def get(self,jid):
        folder=self.folder(jid)
        state=json.loads((folder/'job.json').read_text())
        if state['status']=='uploading' and (folder/'source.part').exists():
            state['uploaded_bytes']=(folder/'source.part').stat().st_size
        return state

    def remaining(self,state):
        return float(state.get('deadline_at',state['created_at']+self.job_timeout))-time.time()

    def create(self,payload,*,deadline_at=None):
        name=payload.get('name','');size=payload.get('size');models=payload.get('models',[])
        if not isinstance(name,str) or len(name)>240 or Path(name).suffix.lower() not in ('.mp4','.mov'):
            raise ValueError('Выберите MP4 или MOV')
        if type(size) is not int or not 0<size<=MAX_SIZE:raise ValueError('Размер видео: от 1 байта до 512 MB')
        ready={m['key'] for m in self.models() if m['ready']}
        if not isinstance(models,list) or not models or any(not isinstance(k,str) or k not in ready for k in models):
            raise ValueError('Выберите готовую модель')
        models=list(dict.fromkeys(models))
        with self.lock:
            states=[json.loads(p.read_text()) for p in self.directory.glob('*/job.json')]
            if sum(s['size'] for s in states)+size>TOTAL_SIZE:raise ValueError('Хранилище загруженных видео заполнено')
            if sum(s['status'] in ('uploading','queued','processing') for s in states)>=4:
                raise ValueError('Очередь заполнена; дождитесь завершения обработки')
            if shutil.disk_usage(self.directory).free<size*3:raise ValueError('Недостаточно свободного места')
            jid=uuid.uuid4().hex;folder=self.directory/jid;folder.mkdir(mode=0o700)
            created=time.time()
            state=dict(id=jid,name=name,size=size,models=models,status='uploading',stage='uploading',
                       uploaded_bytes=0,created_at=created,updated_at=created,results={},
                       deadline_at=min(created+self.job_timeout,deadline_at) if deadline_at is not None else created+self.job_timeout,
                       private=self.private)
            atomic_json(folder/'job.json',state);(folder/'source.part').touch()
            return state

    def chunk(self,jid,offset,length,stream):
        if type(offset) is not int or type(length) is not int or not 0<length<=CHUNK_SIZE:
            raise ValueError('Некорректный фрагмент файла')
        with self.lock:
            folder=self.folder(jid);state=self.get(jid);partial=folder/'source.part'
            if state['status']!='uploading':raise ValueError('Загрузка уже завершена')
            if self.remaining(state)<=0:raise TimeoutError('Истёк общий лимит 300 секунд')
            actual=partial.stat().st_size
            if offset!=actual:raise FileExistsError('Смещение загрузки не совпадает: '+str(actual))
            if offset+length>state['size']:raise ValueError('Размер превышает заявленный')
            try:
                with partial.open('ab') as f:
                    remaining=length
                    while remaining:
                        chunk=stream.read(min(1024*1024,remaining))
                        if not chunk:raise ValueError('Передача прервана')
                        f.write(chunk);remaining-=len(chunk)
            except Exception:
                with partial.open('r+b') as f:f.truncate(offset)
                raise
            state.update(uploaded_bytes=offset+length,updated_at=time.time())
            atomic_json(folder/'job.json',state)
            return state

    def submit(self,jid):
        with self.lock:
            folder=self.folder(jid);state=self.get(jid)
            if state['status']!='uploading':return state
            if self.remaining(state)<=0:raise TimeoutError('Истёк общий лимит 300 секунд')
            if (folder/'source.part').stat().st_size!=state['size']:raise ValueError('Файл ещё не загружен полностью')
            os.replace(folder/'source.part',folder/'source')
            state.update(status='queued',stage='queued',message='В очереди на обработку',updated_at=time.time())
            atomic_json(folder/'job.json',state);self.tasks.put(jid)
            return state

    def delete(self,jid):
        with self.lock:
            folder=self.folder(jid);state=self.get(jid)
            if state['status'] in ('queued','processing'):
                raise ValueError('Дождитесь завершения обработки')
            shutil.rmtree(folder)
            return dict(deleted=True,id=jid)

    def worker(self):
        while True:
            jid=self.tasks.get()
            try:
                self.run_job(jid)
            finally:self.tasks.task_done()

    def run_job(self,jid):
        """A single wall-clock deadline covers upload, queue, decoding and all models."""
        folder=self.folder(jid);acquired=False
        try:
            state=self.get(jid)
            if state['status'] not in ('queued','processing'):return
            remaining=self.remaining(state)
            if remaining<=0:raise TimeoutError('Истёк общий лимит 300 секунд')
            acquired=INFERENCE_LOCK.acquire(timeout=remaining)
            if not acquired:raise TimeoutError('Истёк общий лимит 300 секунд в очереди')
            remaining=self.remaining(self.get(jid))
            if remaining<=0:raise TimeoutError('Истёк общий лимит 300 секунд в очереди')
            with (folder/'worker.log').open('w') as log:
                result=subprocess.run([str(self.root/'.venv/bin/python'),str(self.root/'ml/infer_video.py'),'--job',str(folder)],
                                      cwd=self.root,stdout=log,stderr=subprocess.STDOUT,timeout=remaining,
                                      env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4'))
            if result.returncode and self.get(jid)['status'] not in ('failed','completed'):
                raise RuntimeError('Процесс обработки завершился с ошибкой')
        except Exception as exc:
            state=self.get(jid)
            timeout=isinstance(exc,(TimeoutError,subprocess.TimeoutExpired))
            state.update(status='failed',stage='failed',message='Истёк общий лимит 300 секунд' if timeout else str(exc),
                         error_code='deadline_exceeded' if timeout else 'inference_failed',updated_at=time.time())
            atomic_json(folder/'job.json',state)
        finally:
            if acquired:INFERENCE_LOCK.release()
