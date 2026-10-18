"""Existing upload worker adapter; no web-server restart or production replacement."""
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

SSH=['ssh','-i','/root/.ssh/nuhack_brev','-p','15171','-o','BatchMode=yes','-o','ConnectTimeout=5','ubuntu@global.prd.ga.run.brev.nvidia.com']
REMOTE='/home/ubuntu/nuhack-training'
DETECTOR='runs/human_training/batch_20261017_211756/detector/fit/weights/last.pt'
DETECTOR_SHA='66392416b0d2add78157768bcd5fdb7bc630e5019c45bcb33a30e400a654f464'

def iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0]));h=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=w*h;union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0

def assign_tracks(frames,max_gap=1.25):
    active={};counter=0
    for frame in frames:
        used=set();t=frame['time_sec']
        for box in sorted(frame['boxes'],key=lambda b:-b['score']):
            options=[(iou(box['xyxy'],old['xyxy']),tid) for tid,(last,old) in active.items() if tid not in used and 0<t-last<=max_gap and old['class_id']==box['class_id']]
            score,tid=max(options,default=(0,None))
            if score<.3:counter+=1;tid='object_'+str(counter)
            box.update(track_id=tid,tracking_status='geometric_proposal',ownership='unknown')
            active[tid]=(t,dict(box));used.add(tid)
        active={k:v for k,v in active.items() if t-v[0]<=max_gap}
    return frames

def gpu_available():
    command='cd '+REMOTE+" && flock -n .gpu-work.lock sh -c 'test -z \"$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)\"'"
    p=subprocess.run(SSH+[command],capture_output=True,text=True,timeout=10)
    return p.returncode==0

def run(folder,state,emit):
    from model_suite import load_bundle,load_frames,predict,MODELS
    from infer_video import windows,merge_events,file_sha256
    from train_x3d import atomic_json
    from ml.autolabel.sampling import frames
    import torch
    from ultralytics import YOLO,settings
    root=Path.cwd();folder=Path(folder).resolve();source=folder/'source'
    if state.get('private'):raise ValueError('Experimental pipeline is not enabled in jury API')
    if not re.fullmatch('[a-f0-9]{32}',state['id']):raise ValueError('Invalid job ID')
    if state['duration_sec']>60:raise ValueError('Контекстный пилот: загрузите видео до 60 секунд')
    deadline=state['deadline_at']-5
    def remaining():
        left=deadline-time.time()
        if left<5:raise TimeoutError('Общий лимит обработки исчерпан; частичный результат не является нормой')
        return left
    pipe=dict(name='Forte X',release='1.0',production_interface=True,version=1,status='preflight',experimental=True,review_status='proposed',ground_truth=False,
              stages=[],whole_video_verdict='insufficient_data',observations=[],rules=[],device='CPU + private T4')
    emit(pipeline=pipe,message='Проверка доступности T4',stage='pipeline_preflight')
    if not gpu_available():
        pipe['status']='gpu_busy';emit(pipeline=pipe)
        raise RuntimeError('T4 занят текущей задачей или недоступен. Повторите после её завершения; процессы не остановлены.')
    torch.set_num_threads(4);results=state['results'];start=time.monotonic()
    pipe['status']='videomae';emit(stage='videomae',message='VideoMAE: интервалы для проверки')
    extractor,bundle=load_bundle(root/'artifacts/models/videomaev2/model.pt',root/'artifacts/assets','cpu')
    scores=[];spans=windows(state['duration_sec'])
    for a,b in spans:
        remaining();x=load_frames(source,a,b,16,224)
        scores.append(dict(start_sec=a,end_sec=b,score=predict(extractor,bundle,x.unsqueeze(0))[0]))
        results['videomaev2']=dict(status='processing',windows=scores,events=[],processed=len(scores),total=len(spans));emit()
    results['videomaev2'].update(status='completed',events=merge_events(scores,bundle['threshold']),score=max(r['score'] for r in scores),threshold=bundle['threshold'],elapsed_sec=time.monotonic()-start,model_sha256=file_sha256(root/'artifacts/models/videomaev2/model.pt'),semantic_scope='binary_candidate_only_not_action_types')
    del extractor,bundle;gc.collect();remaining()
    pipe['stages'].append('videomae');pipe['status']='detector';emit(stage='detector',message='YOLO11m: деньги, документы и печати; рамки требуют проверки')
    from .objects import detect_video
    def progress(count,timestamp):
        remaining();emit(message=f'Детектор: {count} кадров, {timestamp:.1f} с; исходное разрешение + перекрывающиеся фрагменты')
    objects=detect_video(source,root,progress,max_seconds=min(100,remaining()-75))
    atomic_json(folder/'yolo11-objects.json',objects)
    detections=objects['frames']
    results['yolo11']=dict(status='completed',kind='object_detection',objects_file='yolo11-objects.json',object_frames=len(detections),windows=[],events=[],elapsed_sec=objects['elapsed_sec'],objects_partial=objects['partial'],detector=objects['model'])
    if objects['partial']:pipe['detector_partial']=True
    gc.collect();pipe['stages'].append('detector');pipe['status']='qwen';emit(stage='qwen',message='Qwen: обзор всего видео, переходы и память')
    left=remaining()
    if left<70:raise TimeoutError('Недостаточно времени для полного контекстного прохода; результаты детектора сохранены')
    job=REMOTE+'/runs/context_pipeline/'+state['id'];remote_output=folder/'pipeline';remote_output.mkdir(exist_ok=True)
    plan=dict(duration_sec=state['duration_sec'],sha256=state['source_sha256'],candidate_windows=scores,detector_observations=detections)
    atomic_json(remote_output/'input.json',plan)
    subprocess.run(SSH+['mkdir -m 700 -p '+shlex.quote(job)],check=True,capture_output=True,timeout=min(10,remaining()))
    for name,path in [('source',source),('input.json',remote_output/'input.json')]:
        with path.open('rb') as stream:
            subprocess.run(SSH+['umask 077; cat > '+shlex.quote(job+'/'+name)],stdin=stream,capture_output=True,check=True,timeout=min(20,remaining()))
    seconds=min(190,int(remaining()-15))
    command='cd '+REMOTE+' && env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=4 timeout '+str(seconds+5)+' .venv-scenario/bin/python -u -m ml.context_pipeline.worker --job '+shlex.quote(job)+' --max-seconds '+str(seconds)
    with (remote_output/'worker.log').open('w') as log:
        completed=subprocess.run(SSH+[command],stdout=log,stderr=subprocess.STDOUT,timeout=min(seconds+10,remaining()))
    subprocess.run(['rsync','-az','--exclude=source','-e',shlex.join(SSH[:-1]),SSH[-1]+':'+job+'/',str(remote_output)+'/'],check=True,capture_output=True,timeout=min(10,remaining()))
    if completed.returncode or not (remote_output/'result.json').exists():raise RuntimeError('Контекстный проход не завершился. Частичные результаты сохранены; нарушения не объявлены.')
    result=json.loads((remote_output/'result.json').read_text())
    import shutil
    for obs in result['observations']:
        for f in obs['evidence']:
            actual=(remote_output/f['path']).resolve()
            if not actual.is_relative_to(remote_output) or not actual.is_file() or file_sha256(actual)!=f['sha256']:raise ValueError('Invalid evidence file')
            name='yolo11-'+str(1000+len(list(folder.glob('yolo11-*.jpg'))))+'.jpg'
            shutil.copy2(actual,folder/name);f['served_file']=name
    from .rules import whole_video_verdict
    pipe.update(result);pipe['stages']=['videomae','detector','qwen','rules'];pipe['whole_video_verdict']=whole_video_verdict(result.get('rules',[]))
    emit(pipeline=pipe,status='completed' if result['status']=='completed' else 'failed',stage='completed' if result['status']=='completed' else 'partial',message='Связка завершена: проверьте интервалы и доказательства.' if result['status']=='completed' else 'Получен частичный контекстный результат; требуется проверка.')
