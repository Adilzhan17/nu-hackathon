"""CPU frame preparation and private T4 transport, isolated from existing upload jobs."""
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import time

from forte_vision.application.analysis import MAX_DURATION
from forte_vision.config import GPUTransport

_transport = GPUTransport.from_env()
SSH = _transport.ssh_command()
REMOTE = _transport.root


def atomic(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False));temporary.replace(path)


def probe(source):
    """Validate video metadata before accepting a GPU job; no audio decoding."""
    import av
    import math
    with av.open(str(source)) as container:
        if not container.streams.video:raise ValueError('В файле нет видео')
        stream=container.streams.video[0]
        duration=float(stream.duration*stream.time_base) if stream.duration is not None else (
            float(container.duration/av.time_base) if container.duration is not None else 0)
        if not math.isfinite(duration) or duration<=0:raise ValueError('Не удалось определить длительность видео')
        if duration>MAX_DURATION:
            raise ValueError(f'Видео длится {duration:.1f} с; предел Gemma — {MAX_DURATION} с. Анализ не запускался.')
        if stream.width*stream.height>3840*2160:raise ValueError('Разрешение выше 4K; анализ не запускался')
        return dict(duration_sec=duration,width=stream.width,height=stream.height)


def prepare(folder, overview_step=.5):
    import av
    if not isinstance(overview_step,(int,float)) or not .5<=overview_step<=12 or overview_step*4!=round(overview_step*4):
        raise ValueError('Unsupported overview sampling step')
    source=folder/'source';metadata=probe(source);sha=hashlib.sha256(source.read_bytes()).hexdigest()
    frames=[];dense=[];index=0
    (folder/'frames').mkdir(exist_ok=True)
    with av.open(str(source)) as container:
        if not container.streams.video:raise ValueError('В файле нет видео')
        stream=container.streams.video[0];stream.thread_count=2;stream.thread_type='SLICE'
        duration=metadata['duration_sec']
        origin=None
        for frame in container.decode(stream):
            if frame.time is None:continue
            if origin is None:origin=float(frame.time)
            timestamp=float(frame.time)-origin
            if timestamp>MAX_DURATION:
                raise ValueError(f'Фактическая длительность видео превышает {MAX_DURATION} секунд')
            if timestamp+1e-7<index*.25:continue
            frame_id=('f'+f'{index//2:05d}') if index%2==0 else ('d'+f'{index:05d}')
            image=frame.to_image();path=folder/'frames'/(frame_id+'.jpg');image.save(path,quality=93,subsampling=0)
            row=dict(id=frame_id,timestamp=timestamp,requested_sec=index*.25,path='frames/'+path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),width=image.width,height=image.height)
            dense.append(row)
            if index % round(overview_step / .25) == 0:frames.append(row)
            index+=1
    if len(frames)<2:raise ValueError('Видео слишком короткое для последовательности')
    plan=dict(schema_version=1,duration_sec=duration,source_sha256=sha,frames=frames,detail_frames=dense)
    atomic(folder/'plan.json',plan)
    # Copy only the video stream when possible; no audio is read or published.
    from fractions import Fraction
    with av.open(str(source)) as incoming, av.open(str(folder/'preview.mp4'),'w',options={'movflags':'+faststart'}) as out:
        track=incoming.streams.video[0]
        if track.codec_context.name=='h264':
            output=out.add_stream_from_template(track)
            for packet in incoming.demux(track):
                if packet.dts is not None:packet.stream=output;out.mux(packet)
        else:
            output=out.add_stream('libx264',rate=track.average_rate or 30)
            scale=min(1,960/max(track.width,track.height));output.width=int(track.width*scale)//2*2;output.height=int(track.height*scale)//2*2
            output.pix_fmt='yuv420p';output.options={'preset':'ultrafast','crf':'25'};output.thread_count=2
            first=None
            for f in incoming.decode(track):
                if f.time is None:continue
                if first is None:first=float(f.time)
                converted=f.reformat(width=output.width,height=output.height,format='yuv420p')
                converted.pts=round((float(f.time)-first)*90000);converted.time_base=Fraction(1,90000)
                for packet in output.encode(converted):out.mux(packet)
            for packet in output.encode():out.mux(packet)
    return plan


def process(folder,emit,seconds=900,overview_step=.5):
    if overview_step is None:
        duration=probe(folder/'source')['duration_sec']
        overview_step=max(2.0,math.ceil(duration/17*4)/4)
    emit(status='processing',stage='sampling',message=f'Извлекаем кадры через {overview_step:g} с и контекст для уточнения')
    plan=prepare(folder,overview_step=overview_step)
    actual_step=(plan['frames'][1]['timestamp']-plan['frames'][0]['timestamp']) if len(plan['frames'])>1 else overview_step
    emit(duration_sec=plan['duration_sec'],sampled_frames=len(plan['frames']),sampling_step_sec=round(actual_step,6),
         preview_ready=True,stage='private_transfer',
         message='Передаём кадры на собственный T4; внешние AI API не используются')
    remote=REMOTE+'/runs/gemma_lab/'+folder.name
    subprocess.run(SSH+['mkdir -m 700 -p '+shlex.quote(remote)],check=True,capture_output=True,timeout=15)
    rsync=['rsync','-az','--protect-args','-e',shlex.join(SSH[:-1])]
    subprocess.run(rsync+['--include=plan.json','--include=frames/','--include=frames/**','--exclude=*',
        str(folder)+'/',SSH[-1]+':'+remote+'/'],check=True,capture_output=True,timeout=90)
    result_dir=folder/'remote';result_dir.mkdir(exist_ok=True)
    command=(
        'cd '+shlex.quote(REMOTE)+' && timeout -k 10 '+str(seconds+25)+
        ' .venv-gemma-lab/bin/python -u -m forte_vision.inference.gemma --job '+shlex.quote(remote)+
        ' --seconds '+str(seconds)+
        ' --instructions forte_vision/inference/prompts/scenario_v4.txt'
    )
    emit(stage='gemma',message='Gemma: последовательный просмотр и память событий')
    with (folder/'transport.log').open('w') as log:
        proc=subprocess.Popen(SSH+[command],stdout=log,stderr=subprocess.STDOUT)
    deadline=time.monotonic()+seconds+60;last_sync=0
    try:
        while True:
            ended=proc.poll() is not None
            if time.monotonic()-last_sync>8 or ended:
                sync=subprocess.run(rsync+['--include=*.json','--include=llama.log','--exclude=*',
                    SSH[-1]+':'+remote+'/',str(result_dir)+'/'],capture_output=True,timeout=20)
                last_sync=time.monotonic()
                path=result_dir/'result.json'
                if path.exists():
                    result=json.loads(path.read_text())
                    emit(result=result,stage=result['stage'],message='Gemma · '+result['stage'])
                if ended:break
            if time.monotonic()>deadline:raise TimeoutError('Истёк лимит ожидания Gemma; сохранены частичные результаты')
            time.sleep(1)
    finally:
        if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
    path=result_dir/'result.json'
    if not path.exists():raise RuntimeError('Gemma не вернула результат; проверьте runtime и доступность T4')
    result=json.loads(path.read_text())
    emit(result=result,status=result['status'],stage=result['stage'],message=result.get('error') or 'Проход завершён; выводы требуют проверки')
