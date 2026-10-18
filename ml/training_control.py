"""Fixed-command SSH training runner. No arbitrary remote commands from HTTP requests."""
import csv
import json
import os
import re
import shlex
import subprocess
import tarfile
import threading
import time
import uuid
from pathlib import Path

from prepare_training import prepare

HOST = 'global.prd.ga.run.brev.nvidia.com'
KEY = '/root/.ssh/nuhack_brev'


class TrainingControl:
    def __init__(self, root):
        self.root = Path(root)
        self.directory = self.root/'runs'
        self.directory.mkdir(exist_ok=True)
        self.path = self.directory/'control.json'
        self.lock = threading.RLock()
        self.state = json.loads(self.path.read_text()) if self.path.exists() else dict(
            status='awaiting_access',stage='access',message='Ожидается SSH-доступ к GPU',connected=False)
        self.busy = False
        self.checking = False
        self.polling = False
        if self.state.get('status') == 'running' and not self.state.get('remote_launched'):
            self.update(status='failed',stage='failed',message='Подготовка прервана перезапуском панели; удалённое обучение не запускалось.')
        if self.state.get('remote_launched') and self.state.get('status') not in ('completed','failed','stopped'):
            self.busy = True
            threading.Thread(target=self.monitor,daemon=True).start()

    def update(self, **values):
        with self.lock:
            self.state.update(updated_at=time.time(), **values)
            temp = self.path.with_suffix('.tmp')
            temp.write_text(json.dumps(self.state,ensure_ascii=False))
            os.replace(temp,self.path)

    def user(self):
        config = self.directory/'gpu_config.json'
        user = json.loads(config.read_text()).get('user','ubuntu') if config.exists() else 'ubuntu'
        if not re.fullmatch('[a-z_][a-z0-9_-]{0,31}',user):
            raise ValueError('Invalid configured SSH user')
        return user

    def set_user(self, user):
        with self.lock:
            if self.busy or self.checking:
                raise FileExistsError('Нельзя менять подключение во время работы')
            if not isinstance(user,str) or not re.fullmatch('[a-z_][a-z0-9_-]{0,31}',user):
                raise ValueError('Укажите имя пользователя Linux из whoami')
            (self.directory/'gpu_config.json').write_text(json.dumps(dict(user=user)))
            self.update(connected=False)

    def ssh(self):
        return ['ssh','-i',KEY,'-p','21453','-o','BatchMode=yes','-o','IdentitiesOnly=yes',
                '-o','ConnectTimeout=8','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
                '-o','StrictHostKeyChecking=yes',self.user()+'@'+HOST]

    def remote(self, command, timeout=25):
        result = subprocess.run(self.ssh()+[command],text=True,capture_output=True,timeout=timeout)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout or 'SSH command failed')[-1800:])
        return result.stdout

    def probe(self):
        try:
            output = self.remote("python3 -c 'import json,os; print(json.dumps(dict(home=os.path.expanduser(\"~\"))))' && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
            home = json.loads(output.splitlines()[0])['home']
            if not home.startswith('/') or '\n' in home:
                raise ValueError('Invalid remote home')
            values=dict(connected=True,remote_root=home+'/nuhack-training',gpu_info='\n'.join(output.splitlines()[1:]),connection_error=None)
            if not self.busy and self.state.get('status') in ('awaiting_access','ready'):
                values.update(status='ready',stage='ready',message='GPU подключён. Можно запускать обучение.')
            self.update(**values)
        except Exception as exc:
            self.update(connected=False,connection_error=str(exc))
        finally:
            self.checking=False

    def snapshot(self):
        with self.lock:
            data=json.loads(json.dumps(self.state))
            data.update(busy=self.busy,checking=self.checking,ssh_user=self.user(),hourly_rate=6.97,
                        budget_note='Тариф из панели Brev: $6,97/ч. Баланс и время включения VM здесь не доступны. Остановка обучения не останавливает аренду.')
        for key,file in [('dataset','data/training/summary.json'),('narration','data/transcripts/summary.json')]:
            path=self.root/file
            if path.exists(): data[key]=json.loads(path.read_text())
        active=self.root/'data/training/active_split.json'
        if active.exists():
            data['active_split_policy']=json.loads(active.read_text())
            data['legacy_start_supported']=False
            data['next_training_note']='Активно 95/5. Новый запуск использует ml/train_suite.py; архивный X3D-контроллер не меняет закреплённые исходники.'
        rid = data.get('run_id', '')
        if re.fullmatch(r'\d{8}-\d{6}-[a-f0-9]{6}', rid):
            validation = self.directory/rid/'validation.json'
            if validation.exists():
                data['saved_metrics'] = json.loads(validation.read_text())['metrics']
        return data

    def action(self, action):
        with self.lock:
            if action=='check':
                if self.checking: return self.snapshot()
                self.checking=True
                threading.Thread(target=self.probe,daemon=True).start()
            elif action=='start':
                if self.busy: raise FileExistsError('Процесс уже запущен')
                if (self.root/'data/training/active_split.json').exists():
                    raise ValueError('Активно разбиение 95/5. Используйте ml/train_suite.py с активной конфигурацией; архивный запуск X3D остановлен до изменения исходников.')
                if not self.state.get('connected'): raise ValueError('Сначала подключите GPU')
                self.busy=True
                self.update(status='running',stage='preparing',history=[],metrics=None,epoch=0,
                            checkpoint_epoch=None,synced_checkpoint_epoch=None,artifacts=[],remote_launched=False,
                            run_id=time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6],
                            requested_at=time.time(),message='Подготовка данных',log='',stop_requested=False)
                threading.Thread(target=self.launch,daemon=True).start()
            elif action=='stop':
                if not self.busy: raise ValueError('Нет активного обучения')
                self.update(stop_requested=True,message='Запрошена остановка обучения. VM продолжает тарифицироваться.')
                threading.Thread(target=self.request_stop,daemon=True).start()
            else:
                raise ValueError('Unknown action')
            return self.snapshot()

    def request_stop(self):
        try:
            if self.state.get('remote_launched'):
                self.remote('touch '+shlex.quote(self.state['remote_root']+'/runs/'+self.state['run_id']+'/STOP'))
        except Exception as exc:
            self.update(connection_error='Не удалось передать остановку: '+str(exc))

    def launch(self):
        try:
            prepare(self.root)
            rid=self.state['run_id']
            local=self.directory/rid
            local.mkdir()
            archive=local/'payload.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                for name in ('ml/train_x3d.py','ml/gpu_bootstrap.sh','data/training/manifest.csv','data/training/summary.json'):
                    tar.add(self.root/name,arcname=name)
                with (self.root/'data/training/manifest.csv').open() as f:
                    videos=sorted({r['video'] for r in csv.DictReader(f)})
                for video in videos:
                    tar.add(self.root/'data/raw/open'/video,arcname='data/raw/open/'+video)
            if self.state.get('stop_requested'):
                self.update(status='stopped',stage='stopped',message='Остановлено до отправки данных')
                return
            remote=self.state['remote_root']
            q=shlex.quote
            self.update(stage='uploading',message='Передача видео и меток на GPU',upload_bytes=archive.stat().st_size)
            self.remote('mkdir -p '+q(remote+'/runs/'+rid))
            with archive.open('rb') as f:
                result=subprocess.run(self.ssh()+['tar -xzf - -C '+q(remote)],stdin=f,capture_output=True,timeout=1200)
                if result.returncode: raise RuntimeError(result.stderr.decode(errors='replace')[-1800:])
            archive.unlink()
            if self.state.get('stop_requested'):
                self.update(status='stopped',stage='stopped',message='Остановлено до запуска GPU-задачи')
                return
            relative='runs/'+rid
            initial=json.dumps(dict(status='running',stage='environment',message='Установка зависимостей',history=[],updated_at=time.time()))
            # Persist before launch: on a lost response, monitor this run instead of launching another.
            self.update(remote_launched=True,stage='environment',message='Подготовка GPU-окружения')
            command='cd '+q(remote)+' && printf %s '+q(initial)+' > '+q(relative+'/status.json')
            command+=' && { nohup bash ml/gpu_bootstrap.sh '+q(relative)+' > '+q(relative+'/train.log')+' 2>&1 < /dev/null & echo $! > '+q(relative+'/pid')+'; }'
            # A separate shell owns the background process; no inherited SSH output descriptors.
            self.remote('bash -c '+q(command),timeout=30)
            if self.state.get('stop_requested'): self.request_stop()
            self.monitor()
        except Exception as exc:
            if self.state.get('remote_launched'):
                self.update(connection_error=str(exc),message='Связь прервана; проверяем состояние удалённого запуска.')
                self.monitor()
            else:
                self.update(status='failed',stage='failed',message=str(exc))
        finally:
            self.busy=False

    def sync_checkpoint(self, remote, rid, epoch):
        local=self.directory/rid
        local.mkdir(exist_ok=True)
        for name in ('best.pt','validation.json','environment.txt'):
            target=local/name
            temp=target.with_suffix('.download')
            with temp.open('wb') as f:
                result=subprocess.run(self.ssh()+['cat '+shlex.quote(remote+'/runs/'+rid+'/'+name)],stdout=f,stderr=subprocess.PIPE,timeout=120)
            if result.returncode:
                temp.unlink(missing_ok=True)
                raise RuntimeError('Не удалось скопировать '+name)
            os.replace(temp,target)
        self.update(synced_checkpoint_epoch=epoch,artifacts=[dict(name=n,url='/api/training/artifact/'+rid+'/'+n,
                    bytes=(local/n).stat().st_size) for n in ('best.pt','validation.json','environment.txt')])

    def monitor(self):
        self.polling=True
        rid=self.state['run_id']
        remote=self.state['remote_root']
        script='''import json,subprocess,time,os
from pathlib import Path
p=Path(%s)
state=json.loads((p/'status.json').read_text())
if state.get('status')=='running' and (p/'pid').exists():
 try: os.kill(int((p/'pid').read_text()),0)
 except ProcessLookupError: state.update(status='failed',stage='failed',message='Удалённый процесс завершился без итогового статуса; проверьте журнал.')
with (p/'train.log').open('rb') as f:
 f.seek(0,2); f.seek(max(0,f.tell()-20000)); log=f.read().decode(errors='replace')
r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
print(json.dumps(dict(training=state,log=log,gpu_live=r.stdout.strip(),observed_at=time.time())))
''' % repr(remote+'/runs/'+rid)
        try:
            while True:
                try:
                    result=json.loads(self.remote('python3 -c '+shlex.quote(script)))
                    state=result.pop('training')
                    local = self.directory/rid
                    local.mkdir(exist_ok=True)
                    (local/'status.json').write_text(json.dumps(state,ensure_ascii=False))
                    (local/'log.txt').write_text(result['log'])
                    # Only trainer-owned telemetry can override these fields.
                    allowed=('status','stage','message','history','metrics','epoch','epochs','batch','batches','processed','total',
                             'train_loss','best_epoch','checkpoint_epoch','elapsed_sec','started_at','gpu_name','torch_version')
                    self.update(**{k:v for k,v in state.items() if k in allowed},**result,connected=True,connection_error=None)
                    epoch=state.get('checkpoint_epoch')
                    if epoch is not None and epoch!=self.state.get('synced_checkpoint_epoch'):
                        self.sync_checkpoint(remote,rid,epoch)
                    if state.get('status') in ('completed','stopped','failed'): break
                except Exception as exc:
                    self.update(connected=False,connection_error=str(exc))
                time.sleep(8)
        finally:
            self.busy=False
            self.polling=False
