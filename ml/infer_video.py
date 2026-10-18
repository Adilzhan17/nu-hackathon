"""Sliding-window inference; explicit two-model experiment also uses our private T4."""
import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from model_suite import MODELS,load_bundle,load_frames,predict
from train_x3d import atomic_json


def windows(duration, length=8, stride=4):
    if duration<=length:return [(0.,duration)]
    starts=[float(x) for x in range(0,int(duration-length)+1,stride)]
    last=round(duration-length,3)
    if last-starts[-1]>.1:starts.append(last)
    return [(s,min(duration,s+length)) for s in starts]


def merge_events(scores, threshold=.5):
    events=[]
    for row in scores:
        if row['score']<threshold:continue
        if events and row['start_sec']<=events[-1]['end_sec']:
            event=events[-1];event['end_sec']=max(event['end_sec'],row['end_sec'])
            if row['score']>event['score']:
                event.update(score=row['score'],timecode_sec=(row['start_sec']+row['end_sec'])/2)
        else:
            events.append(dict(start_sec=row['start_sec'],end_sec=row['end_sec'],score=row['score'],
                               timecode_sec=(row['start_sec']+row['end_sec'])/2))
    return events


def preview(source,destination):
    import av
    with av.open(str(source)) as incoming:
        track=incoming.streams.video[0]
        if track.codec_context.name!='h264':return False
        partial=destination.with_suffix('.part.mp4')
        with av.open(str(partial),'w',options={'movflags':'+faststart'}) as outgoing:
            output=outgoing.add_stream_from_template(track)
            for packet in incoming.demux(track):
                if packet.dts is None:continue
                packet.stream=output;outgoing.mux(packet)
        os.replace(partial,destination)
    return True


def evidence(source, timestamp, destination):
    import av
    with av.open(str(source)) as container:
        stream=container.streams.video[0]
        stream.codec_context.thread_count=1;stream.thread_type='SLICE'
        origin=float((stream.start_time or 0)*stream.time_base)
        container.seek(int((timestamp+origin)/float(stream.time_base)),stream=stream,backward=True)
        for frame in container.decode(stream):
            if frame.time is not None and frame.time-origin>=timestamp:
                picture=frame.to_image();picture.thumbnail((1280,1280))
                picture.save(destination,format='JPEG',quality=85)
                return float(frame.time-origin)
    return None


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    args=parser.parse_args();folder=args.job
    state=json.loads((folder/'job.json').read_text())
    def emit(**values):
        state.update(updated_at=time.time(),**values);atomic_json(folder/'job.json',state)
    def check_deadline():
        if time.time()>=state.get('deadline_at',state['created_at']+300):
            raise TimeoutError('Истёк общий лимит 300 секунд на весь запрос, включая очередь')
    try:
        check_deadline()
        import av
        import torch
        torch.set_num_threads(4)
        source=folder/'source'
        with av.open(str(source)) as container:
            if state.get('private') and not set(container.format.name.split(','))&{'mov','mp4'}:
                raise ValueError('Файл не является декодируемым MOV-видео')
            if not container.streams.video:raise ValueError('В файле нет видеопотока')
            track=container.streams.video[0]
            track.codec_context.thread_count=1;track.thread_type='SLICE'
            duration=float(track.duration*track.time_base) if track.duration is not None else float(container.duration/av.time_base)
            duration_limit=180 if state.get('private') else 600
            if not 0<duration<=duration_limit:raise ValueError(f'Допустимая длительность: до {duration_limit} секунд')
            if track.width*track.height>3840*2160:raise ValueError('Разрешение выше 4K')
            next(container.decode(track))
        digest=file_sha256(source);check_deadline()
        known=[]
        import csv
        manifest=Path('data/training/manifest.csv')
        if not state.get('private') and manifest.is_file():
            with manifest.open() as f:
                known=sorted({r['split'] for r in csv.DictReader(f) if r['sha256']==digest})
        emit(status='processing',stage='preparing',duration_sec=duration,known_splits=known,
             source_sha256=digest,message='Подготовка видео',results={},device='CPU · основной сервер')
        ready=preview(source,folder/'preview.mp4')
        check_deadline()
        emit(preview_ready=ready)
        # Dedicated experiment selected on /analyze. The production jury API
        # remains private=True/X3D and cannot enter this branch.
        if not state.get('private') and set(state['models']) == {'videomaev2','yolo11'}:
            import sys
            sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
            from ml.context_pipeline.bridge import run as contextual_run
            contextual_run(folder,state,emit)
            return
        spans=windows(duration)
        for key in state['models']:
            check_deadline()
            began=time.monotonic();results=state['results']
            results[key]=dict(status='processing',windows=[],events=[],processed=0,total=len(spans))
            emit(stage='analyzing',current_model=key,message='Анализ: '+MODELS[key]['name'])
            try:
                path=Path('artifacts/models')/key/'model.pt'
                if not path.is_file():raise ValueError('Модель ещё не готова')
                extractor,bundle=load_bundle(path,Path('artifacts/assets'),'cpu')
                check_deadline()
                cfg=MODELS[key];scores=[]
                for index,(start,end) in enumerate(spans):
                    check_deadline()
                    frames=load_frames(source,start,end,cfg['frames'],cfg['size'])
                    score=predict(extractor,bundle,frames.unsqueeze(0))[0]
                    check_deadline()
                    scores.append(dict(start_sec=start,end_sec=end,score=score))
                    results[key].update(windows=scores,processed=index+1)
                    emit()
                events=merge_events(scores,bundle['threshold'])
                for i,event in enumerate(events):
                    check_deadline()
                    filename=f'{key}-{i}.jpg'
                    actual_timestamp=evidence(source,event['timecode_sec'],folder/filename)
                    if actual_timestamp is not None:
                        event.update(evidence=filename,evidence_timestamp_sec=actual_timestamp,
                                     evidence_sha256=file_sha256(folder/filename))
                results[key].update(status='completed',events=events,score=max(r['score'] for r in scores),
                                    verdict='needs_review' if events else 'no_candidates',threshold=bundle['threshold'],
                                    model_sha256=file_sha256(path),
                                    elapsed_sec=time.monotonic()-began)
                del extractor,bundle
                gc.collect()
                if key in ('yolov8','yolo11'):
                    from detect_objects import detect
                    emit(message='Рамки объектов: отдельный детектор; его checkpoint и классы указаны под видео.')
                    def object_progress(count,timestamp):
                        results[key]['object_progress']=dict(frames=count,time_sec=timestamp)
                        emit()
                    try:
                        objects=detect(source,key,object_progress)
                        atomic_json(folder/(key+'-objects.json'),objects)
                        results[key].update(objects_file=key+'-objects.json',object_frames=len(objects['frames']),
                                            objects_partial=objects['partial'])
                    except Exception as exc:results[key]['objects_error']=str(exc)
                    results[key]['elapsed_sec']=time.monotonic()-began
            except Exception as exc:
                results[key].update(status='failed',error=str(exc),elapsed_sec=time.monotonic()-began)
                if isinstance(exc,TimeoutError):raise
            emit()
        done=sum(r['status']=='completed' for r in state['results'].values())
        check_deadline()
        emit(status='completed' if done==len(state['models']) else 'failed',stage='completed',
             message=f'Проверено моделей: {done}/{len(state["models"])}. Подозрительные интервалы требуют просмотра.')
    except Exception as exc:
        if state.get('pipeline'):
            state['pipeline'].update(status='failed',error=str(exc))
        emit(status='failed',stage='failed',message=str(exc),
             error_code='deadline_exceeded' if isinstance(exc,TimeoutError) else 'inference_failed')
        raise


if __name__=='__main__':main()
