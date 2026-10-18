"""Versioned target detector overlay; legacy COCO fallback is explicitly labelled."""
import hashlib
import time
from pathlib import Path

CLASSES={0:'Человек',39:'Бутылка',41:'Чашка',56:'Стул',60:'Стол',62:'Экран',63:'Ноутбук',
         64:'Мышь',66:'Клавиатура',67:'Телефон',73:'Книга'}


def detect(path,key,progress=None,max_seconds=120):
    # A new subprocess reads the registry for each analysis. Switching versions
    # is atomic and does not restart the site or modify manual annotations.
    root=Path(__file__).resolve().parent.parent
    if (root/'runs/context_pipeline/detector_active.json').exists():
        import sys
        sys.path.insert(0,str(root))
        from ml.context_pipeline.objects import detect_video
        return detect_video(path,root,progress,max_seconds)
    import av
    from ultralytics import YOLO,settings
    settings.update({'sync':False})
    name='yolov8n.pt' if key=='yolov8' else 'yolo11n.pt'
    checkpoint=Path('artifacts/assets')/name
    model=YOLO(str(checkpoint),task='detect')
    frames=[];next_time=0.;began=time.monotonic();limited=False
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO'
        width,height=stream.width,stream.height
        for frame in container.decode(stream):
            if frame.time is None or frame.time<next_time:continue
            if time.monotonic()-began>max_seconds:limited=True;break
            image=frame.to_ndarray(format='bgr24')
            result=model.predict(image,imgsz=640,conf=.35,classes=list(CLASSES),device='cpu',verbose=False)[0]
            boxes=[]
            for xyxy,conf,cls in zip(result.boxes.xyxyn.tolist(),result.boxes.conf.tolist(),result.boxes.cls.tolist()):
                boxes.append(dict(xyxy=xyxy,score=conf,class_id=int(cls),label=CLASSES[int(cls)]))
            frames.append(dict(time_sec=round(frame.time,3),boxes=boxes))
            next_time=frame.time+.5
            if progress and len(frames)%10==0:progress(len(frames),frame.time)
    return dict(model=name,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                classes=model.names,displayed_class_ids=list(CLASSES),confidence=.35,
                limitations=['COCO: деньги, документы/ID, ключи и печати этим режимом не распознаются.'],
                score_semantics='uncalibrated detector score',sampling_fps=2,width=width,height=height,frames=frames,
                elapsed_sec=time.monotonic()-began,partial=limited,
                scope='COCO objects; not cash, keys, safe states or legal violations')
