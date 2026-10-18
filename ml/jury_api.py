"""Authenticated local jury API. Uploaded jury videos never enter training or public assets."""
import base64
import hashlib
import hmac
import json
import math
import os
import re
import socket
import stat
import time
from email.parser import BytesHeaderParser
from pathlib import Path
from urllib.parse import urlparse

from infer_video import file_sha256
from train_x3d import atomic_json
from video_jobs import VideoJobs

MAX_UPLOAD=64*1024*1024
MAX_HEADERS=16*1024
READ_CHUNK=64*1024
MODEL_KEY='x3d'
WARNING='Экспериментальная бинарная модель: тип действия и нормативное основание не установлены. Результат требует просмотра человеком; качество не подтверждено на независимом тесте.'


class APIError(Exception):
    def __init__(self,status,code,message):
        super().__init__(message);self.status=status;self.code=code


class RequestReader:
    """Never buffer the whole request; protect slow uploads and total request time."""
    def __init__(self,stream,length,deadline,connection=None):
        self.stream=stream;self.remaining=length;self.deadline=deadline;self.connection=connection

    def read(self,length):
        remaining_time=self.deadline-time.time()
        if remaining_time<=0:raise APIError(408,'deadline_exceeded','Истёк общий лимит 300 секунд')
        if self.connection is not None:self.connection.settimeout(min(15,remaining_time))
        length=min(length,self.remaining,READ_CHUNK)
        if length<=0:return b''
        value=self.stream.read(length)
        if not value:raise APIError(400,'incomplete_upload','Передача файла прервана')
        self.remaining-=len(value)
        return value

    def readline(self,limit):
        line=bytearray()
        while self.remaining and len(line)<limit:
            line+=self.read(1)
            if line.endswith(b'\r\n'):return bytes(line)
        raise APIError(400,'invalid_multipart','Некорректные заголовки multipart')


def multipart_header(reader,boundary):
    if reader.readline(MAX_HEADERS)!=b'--'+boundary+b'\r\n':
        raise APIError(400,'invalid_multipart','Ожидалось поле file в multipart/form-data')
    raw=bytearray()
    while True:
        line=reader.readline(MAX_HEADERS-len(raw))
        if line==b'\r\n':break
        raw+=line
        if len(raw)>=MAX_HEADERS:raise APIError(413,'headers_too_large','Слишком длинные заголовки multipart')
    headers=BytesHeaderParser().parsebytes(bytes(raw)+b'\r\n')
    if len(headers.get_all('Content-Disposition',[]))!=1 or headers.get_content_disposition()!='form-data':
        raise APIError(400,'invalid_multipart','Некорректный Content-Disposition')
    if headers.get_param('name',header='content-disposition')!='file':
        raise APIError(400,'invalid_field','Загрузите одно видео в поле file')
    filename=headers.get_filename()
    if not isinstance(filename,str):raise APIError(400,'missing_filename','Отсутствует имя видео')
    filename=filename.replace('\\','/').split('/')[-1]
    if not filename or len(filename)>240 or '\x00' in filename or Path(filename).suffix.lower()!='.mov':
        raise APIError(415,'unsupported_media','Требуется видео в формате .mov')
    return filename


def multipart_file(reader,boundary,destination):
    """Stream a single binary part; detect boundaries split at any byte offset."""
    delimiter=b'\r\n--'+boundary
    pending=b'';written=0;digest=hashlib.sha256()
    with Path(destination).open('wb') as output:
        os.chmod(destination,0o600)
        def write(value):
            nonlocal written
            written+=len(value)
            if written>MAX_UPLOAD:raise APIError(413,'upload_too_large','Максимальный размер MOV — 64 MiB')
            digest.update(value);output.write(value)
        while reader.remaining:
            pending+=reader.read(READ_CHUNK)
            while True:
                index=pending.find(delimiter)
                if index<0:
                    keep=len(delimiter)+4
                    if len(pending)>keep:write(pending[:-keep]);pending=pending[-keep:]
                    break
                suffix=index+len(delimiter)
                if len(pending)<suffix+2:
                    write(pending[:index]);pending=pending[index:];break
                marker=pending[suffix:suffix+2]
                if marker not in (b'--',b'\r\n'):
                    write(pending[:index+2]);pending=pending[index+2:];continue
                if marker==b'\r\n':
                    raise APIError(400,'multiple_parts','Допускается только одно поле file')
                write(pending[:index]);tail=pending[suffix+2:]
                if len(tail)+reader.remaining>2:
                    raise APIError(400,'trailing_multipart','Некорректное завершение multipart')
                while reader.remaining:tail+=reader.read(reader.remaining)
                if tail not in (b'',b'\r\n'):
                    raise APIError(400,'invalid_multipart','Некорректное завершение multipart')
                if not written:raise APIError(400,'empty_upload','Пустой файл')
                return written,digest.hexdigest()
    raise APIError(400,'incomplete_upload','Отсутствует завершающая граница multipart')


def configured_key(root):
    key=os.environ.get('NUHACK_API_KEY')
    if key:return key.encode('utf-8')
    path=Path(os.environ.get('NUHACK_API_KEY_FILE',str(Path(root)/'.secrets/jury_api_key')))
    try:
        metadata=path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode)&0o077:
            return None
        key=path.read_bytes().strip()
        return key if 24<=len(key)<=256 else None
    except OSError:return None


def finite(value):
    return type(value) in (float,int) and math.isfinite(value)


def completed_result(state,folder):
    """Convert a complete binary result without inventing action classes or legal facts."""
    if state.get('models')!=[MODEL_KEY]:raise ValueError('Недопустимая модель API')
    result=state.get('results',{}).get(MODEL_KEY,{})
    duration=state.get('duration_sec')
    if not finite(duration) or not 0<duration<=180:raise ValueError('Не подтверждена длительность видео')
    windows=result.get('windows',[]);threshold=result.get('threshold')
    if result.get('status')!='completed' or not windows or result.get('processed')!=len(windows) or result.get('total')!=len(windows):
        raise ValueError('Обработаны не все окна видео')
    if not finite(threshold) or not 0<=threshold<=1:raise ValueError('Отсутствует порог модели')
    covered=0.;candidates=False
    for row in windows:
        start,end,score=(row.get(k) for k in ('start_sec','end_sec','score'))
        if not all(finite(v) for v in (start,end,score)) or not 0<=start<end<=duration+.1 or not 0<=score<=1 or start>covered+.1:
            raise ValueError('Не подтверждено покрытие всего видео')
        covered=max(covered,end);candidates|=score>=threshold
    if covered<duration-.1:raise ValueError('Обработка завершилась до конца видео')
    events=result.get('events',[])
    if bool(events)!=candidates or len(events)>64:raise ValueError('Кандидаты и интервалы модели не согласованы')
    source_hash=state.get('source_sha256','');model_hash=result.get('model_sha256','')
    if not re.fullmatch('[0-9a-f]{64}',source_hash) or not re.fullmatch('[0-9a-f]{64}',model_hash):
        raise ValueError('Отсутствует provenance источника или модели')
    violations=[];evidence_bytes=0
    for event in events:
        start,end,stamp=(event.get(k) for k in ('start_sec','end_sec','evidence_timestamp_sec'))
        if not all(finite(v) for v in (start,end,stamp)) or not 0<=start<=stamp<=end+.15 or not start<end<=duration+.1:
            raise ValueError('Нет подтверждённого таймкода кадра')
        name=event.get('evidence','')
        if not isinstance(name,str) or not re.fullmatch(r'x3d-[0-9]+\.jpg',name):raise ValueError('Нет кадра-доказательства')
        path=Path(folder)/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size>2*1024*1024:
            raise ValueError('Кадр-доказательство недоступен')
        raw=path.read_bytes();evidence_bytes+=len(raw)
        if evidence_bytes>12*1024*1024:raise ValueError('Превышен размер кадров-доказательств')
        if not raw.startswith(b'\xff\xd8') or not raw.endswith(b'\xff\xd9') or hashlib.sha256(raw).hexdigest()!=event.get('evidence_sha256'):
            raise ValueError('Не подтверждена целостность кадра')
        from PIL import Image
        with Image.open(path) as picture:
            if picture.format!='JPEG' or picture.width*picture.height>1280*1280:raise ValueError('Недопустимый формат кадра')
            picture.verify()
        violations.append(dict(type='Неклассифицированный признак возможного нарушения',
            timecode_sec=stamp,start_sec=start,end_sec=end,
            description=f'Бинарная модель выделила интервал {start:.2f}–{end:.2f} с как подозрительный. Конкретное действие и нормативное основание не установлены; кадр приведён для проверки человеком.',
            certainty='suspected',classification_status='unclassified',regulatory_source=None,
            evidence_frames=[dict(timestamp_sec=stamp,image_base64=base64.b64encode(raw).decode('ascii'),
                provenance=dict(source_sha256=source_hash,jpeg_sha256=event['evidence_sha256'],frame_timestamp_sec=stamp,resize_max_edge=1280))]))
    warnings=[WARNING,'Используется исходный порог модели и объединение положительных окон; это не проверенная калибровка вердикта на целом видео.']
    if not violations:warnings.append('Кандидаты не обнаружены; это не подтверждение соблюдения всех кассовых правил.')
    response=dict(job_id=state['id'],status='done',verdict='violation' if violations else 'no_violation',
        violations=violations,warnings=warnings,duration_sec=duration,
        provenance=dict(source_sha256=source_hash,model_key=MODEL_KEY,model_sha256=model_hash,threshold=threshold,
                        aggregation='any_positive_window',local_inference=True,classification_supported=False,legal_mapping_supported=False))
    if violations:response['certainty']='suspected'
    if state.get('preview_ready') and (Path(folder)/'preview.mp4').is_file():
        response['preview_url']=f'/api/v1/jobs/{state["id"]}/preview'
    return response


class JuryAPI:
    def __init__(self,root,jobs=None,key=None):
        self.root=Path(root)
        self.jobs=jobs if jobs is not None else VideoJobs(self.root,directory=self.root/'data/jury_jobs',private=True)
        self.key=key.encode('utf-8') if isinstance(key,str) else key if key is not None else configured_key(self.root)

    def authorize(self,handler):
        supplied=handler.headers.get_all('X-API-Key',[])
        if not self.key:raise APIError(503,'api_not_configured','API-ключ не настроен на сервере')
        if len(supplied)!=1 or not hmac.compare_digest(supplied[0].encode('utf-8'),self.key):
            raise APIError(401,'unauthorized','Требуется верный X-API-Key')

    def failure(self,handler,exc):
        handler.close_connection=True
        handler.json_response(dict(error=dict(code=exc.code,message=str(exc))),exc.status)

    def handle_post(self,handler):
        path=urlparse(handler.path).path
        if not path.startswith('/api/v1/'):return False
        jid=None;original_timeout=handler.connection.gettimeout();started=time.time()
        try:
            self.authorize(handler)
            if path!='/api/v1/analyze':raise APIError(404,'not_found','Маршрут не найден')
            if handler.headers.get('Transfer-Encoding') or len(handler.headers.get_all('Content-Length',[]))!=1:
                raise APIError(411,'length_required','Требуется Content-Length без Transfer-Encoding')
            try:length=int(handler.headers['Content-Length'])
            except (ValueError,TypeError):raise APIError(400,'invalid_length','Некорректный Content-Length')
            if not 0<length<=MAX_UPLOAD+MAX_HEADERS+256:
                raise APIError(413,'upload_too_large','Максимальный размер MOV — 64 MiB')
            content=handler.headers.get('Content-Type','')
            parsed=BytesHeaderParser().parsebytes(('Content-Type: '+content+'\r\n\r\n').encode('utf-8'))
            boundary=parsed.get_boundary()
            if parsed.get_content_type()!='multipart/form-data' or not isinstance(boundary,str) or not re.fullmatch(r"[0-9A-Za-z'()+_,./:=? -]{1,70}",boundary) or boundary.endswith(' '):
                raise APIError(415,'invalid_multipart','Требуется multipart/form-data с boundary')
            reader=RequestReader(handler.rfile,length,started+300,handler.connection)
            name=multipart_header(reader,boundary.encode('ascii'))
            try:state=self.jobs.create(dict(name=name,size=length,models=[MODEL_KEY]),deadline_at=started+300)
            except ValueError as exc:raise APIError(503,'capacity_unavailable',str(exc))
            jid=state['id'];folder=self.jobs.folder(jid)
            size,digest=multipart_file(reader,boundary.encode('ascii'),folder/'source.part')
            with self.jobs.lock:
                state=self.jobs.get(jid);state.update(size=size,uploaded_bytes=size,source_sha256=digest,updated_at=time.time(),purpose='jury')
                atomic_json(folder/'job.json',state)
            self.jobs.submit(jid)
            handler.json_response(dict(job_id=jid,status='processing'),202)
        except (APIError,TimeoutError,socket.timeout) as exc:
            if jid is not None:
                try:self.jobs.delete(jid)
                except (OSError,ValueError):pass
            self.failure(handler,exc if isinstance(exc,APIError) else APIError(408,'upload_timeout','Истёк срок загрузки видео'))
        except OSError:
            if jid is not None:
                try:self.jobs.delete(jid)
                except (OSError,ValueError):pass
            self.failure(handler,APIError(503,'storage_error','Не удалось сохранить загрузку'))
        finally:handler.connection.settimeout(original_timeout)
        return True

    def job_result(self,jid):
        state=self.jobs.get(jid);folder=self.jobs.folder(jid)
        if state['status'] in ('uploading','queued','processing'):
            if self.jobs.remaining(state)<=0:
                return dict(job_id=jid,status='failed',error=dict(code='deadline_exceeded',message='Истёк общий лимит 300 секунд, включая очередь'))
            result=state.get('results',{}).get(MODEL_KEY,{})
            return dict(job_id=jid,status='processing',stage=state.get('stage','queued'),
                        progress=dict(processed=result.get('processed',0),total=result.get('total',0)))
        if state['status']=='failed':
            return dict(job_id=jid,status='failed',error=dict(code=state.get('error_code','inference_failed'),message=state.get('message','Не удалось обработать видео')))
        if state['status']!='completed':
            return dict(job_id=jid,status='failed',error=dict(code='invalid_state',message='Неизвестное состояние обработки'))
        if state['updated_at']>state['deadline_at']:
            return dict(job_id=jid,status='failed',error=dict(code='deadline_exceeded',message='Обработка превысила общий лимит 300 секунд'))
        try:return completed_result(state,folder)
        except (ValueError,KeyError,OSError,TypeError) as exc:
            return dict(job_id=jid,status='failed',error=dict(code='incomplete_result',message=str(exc)))

    def handle_get(self,handler):
        path=urlparse(handler.path).path
        if not path.startswith('/api/v1/'):return False
        original_timeout=handler.connection.gettimeout()
        handler.connection.settimeout(15)
        try:
            self.authorize(handler)
            match=re.fullmatch(r'/api/v1/jobs/([0-9a-f]{32})(/preview)?',path)
            if not match:raise APIError(404,'not_found','Задание не найдено')
            jid=match[1]
            if match[2]:
                state=self.jobs.get(jid)
                if state['status']!='completed' or not state.get('preview_ready'):
                    raise APIError(404,'preview_unavailable','Предпросмотр пока недоступен')
                self.send_preview(handler,self.jobs.folder(jid)/'preview.mp4')
            else:handler.json_response(self.job_result(jid))
        except FileNotFoundError:self.failure(handler,APIError(404,'not_found','Задание не найдено'))
        except APIError as exc:self.failure(handler,exc)
        except (BrokenPipeError,ConnectionResetError,socket.timeout):handler.close_connection=True
        finally:handler.connection.settimeout(original_timeout)
        return True

    @staticmethod
    def send_preview(handler,path):
        if path.is_symlink() or not path.is_file():raise APIError(404,'preview_unavailable','Предпросмотр недоступен')
        size=path.stat().st_size;start=0;end=size-1;status=200
        value=handler.headers.get('Range')
        if value:
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',value)
            if not match or not any(match.groups()):raise APIError(416,'invalid_range','Некорректный Range')
            if match[1]:start=int(match[1]);end=min(end,int(match[2])) if match[2] else end
            else:start=max(0,size-int(match[2]))
            if start>end or start>=size:raise APIError(416,'invalid_range','Range вне файла')
            status=206
        handler.send_response(status);handler.send_header('Content-Type','video/mp4')
        handler.send_header('Content-Length',str(end-start+1));handler.send_header('Accept-Ranges','bytes')
        handler.send_header('Cache-Control','private, no-store');handler.send_header('X-Content-Type-Options','nosniff')
        if status==206:handler.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        handler.end_headers()
        with path.open('rb') as stream:
            stream.seek(start);remaining=end-start+1
            while remaining:
                chunk=stream.read(min(READ_CHUNK,remaining))
                if not chunk:break
                handler.wfile.write(chunk);remaining-=len(chunk)
