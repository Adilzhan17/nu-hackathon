"""Native-resolution overlapping tiles, class-aware NMS, immutable detector registry."""
import hashlib
import json
from pathlib import Path
from .bridge import iou

def tile_origins(length,size,overlap=.25):
    if length<=size:return [0]
    step=max(1,round(size*(1-overlap)));starts=list(range(0,length-size+1,step))
    if starts[-1]!=length-size:starts.append(length-size)
    return starts

def nms(boxes,threshold=.5):
    kept=[]
    for box in sorted(boxes,key=lambda b:-b['score']):
        if any(k['class_id']==box['class_id'] and iou(k['xyxy'],box['xyxy'])>threshold for k in kept):continue
        kept.append(box)
    return kept

def detect_image(model,image,device='cpu',conf=.25,tiled=True,size=960,native_tile_size=384):
    w,h=image.size;regions=[(0,0,w,h,'full')]
    if tiled and (w>native_tile_size or h>native_tile_size):
        regions.extend((x,y,min(w,x+native_tile_size),min(h,y+native_tile_size),'tile') for y in tile_origins(h,native_tile_size) for x in tile_origins(w,native_tile_size))
    boxes=[]
    for x,y,right,bottom,kind in regions:
        crop=image if kind=='full' else image.crop((x,y,right,bottom))
        result=model.predict(crop,imgsz=size,conf=conf,iou=.5,device=device,verbose=False,max_det=100)[0]
        for coords,score,cls in zip(result.boxes.xyxy.tolist(),result.boxes.conf.tolist(),result.boxes.cls.tolist()):
            a,b,c,d=coords
            # Suppress fragments touching an internal tile edge; overlapping
            # neighbors and the full frame remain available for the same object.
            if kind=='tile' and ((x>0 and a<3) or (y>0 and b<3) or (right<w and c>right-x-3) or (bottom<h and d>bottom-y-3)):continue
            boxes.append(dict(xyxy=[max(0,(x+a)/w),max(0,(y+b)/h),min(1,(x+c)/w),min(1,(y+d)/h)],
                score=float(score),class_id=int(cls),label=model.names[int(cls)],view=kind,source_tile=[x,y,right,bottom],review_status='proposed'))
    return nms(boxes),len(regions)

def active_detector(root):
    root=Path(root).resolve();config=json.loads((root/'runs/context_pipeline/detector_active.json').read_text())
    path=(root/config['checkpoint']).resolve()
    if not path.is_relative_to(root/'runs') or path.suffix!='.pt':raise ValueError('Invalid detector checkpoint')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=config['sha256']:raise ValueError('Detector checksum mismatch')
    return path,config

def detect_video(path,root,progress=None,max_seconds=120):
    import av,time
    from ultralytics import YOLO,settings
    from .bridge import assign_tracks
    settings.update({'sync':False});checkpoint,config=active_detector(root);model=YOLO(str(checkpoint),task='detect')
    began=time.monotonic();next_time=0;rows=[];partial=False;views=0
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO';w,h=stream.width,stream.height
        duration=float(stream.duration*stream.time_base) if stream.duration else float(container.duration/av.time_base)
        interval=max(2,duration/12)
        for frame in container.decode(stream):
            if frame.time is None or frame.time<next_time:continue
            if time.monotonic()-began>max_seconds:partial=True;break
            boxes,count=detect_image(model,frame.to_image(),conf=config['confidence'],tiled=config['tiled']);views+=count
            rows.append(dict(time_sec=frame.time,boxes=boxes));next_time=frame.time+interval
            if progress:progress(len(rows),frame.time)
    return dict(model=config['name'],checkpoint=config['checkpoint'],checkpoint_sha256=config['sha256'],classes=model.names,
        confidence=config['confidence'],nms_iou=.5,tile_size=384,model_input_size=960,tile_overlap=.25,native_resolution_tiles=True,
        frames=assign_tracks(rows),width=w,height=h,partial=partial,sampling_interval_sec=interval,views_processed=views,
        elapsed_sec=time.monotonic()-began,scope='Human-finetuned target objects; not identity verification, ownership or legal violations',
        score_semantics='uncalibrated detector score',evaluation=config.get('evaluation'),limitations=config.get('limitations',[]))
