"""Shared, offline-capable feature extractors for five model families."""
import json
from pathlib import Path

MODELS = {
    'x3d': dict(name='X3D-S', frames=13, size=160, license='Apache-2.0', kind='Видеомодель'),
    'slowfast': dict(name='SlowFast R50', frames=32, size=256, license='Apache-2.0', kind='Видеомодель'),
    'videomaev2': dict(name='VideoMAE V2 Base', frames=16, size=224, license='CC-BY-NC-4.0 · эксперимент', kind='Видеомодель', default_selected=False),
    'videomaev2_large': dict(name='VideoMAE V2 Large', frames=16, size=224, license='CC-BY-NC-4.0 · эксперимент', kind='Полное дообучение видеомодели', training_mode='full_finetune', default_selected=False),
    'videomaev2_giant': dict(name='VideoMAE V2 Giant', frames=16, size=224, license='CC-BY-NC-4.0 · эксперимент', kind='Полное дообучение видеомодели', training_mode='full_finetune', default_selected=False),
    'yolov8': dict(name='YOLOv8n-cls', frames=8, size=224, license='AGPL-3.0 / Enterprise', kind='Кадры + временная агрегация'),
    'yolo11': dict(name='YOLO11n-cls', frames=8, size=224, license='AGPL-3.0 / Enterprise', kind='Кадры + временная агрегация'),
}
REVISION = '78c337a418cc4adaf7f1ff6c7ba343418393966b'


def load_frames(path, start, end, frames, size):
    import av
    import numpy as np
    import torch
    timestamps = np.linspace(start, max(start, end-.08), frames)
    images=[]
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        # Bound FFmpeg decoder threads: AUTO frame threading can stall on seeks
        # between many short windows in the same MOV on the CPU deployment.
        stream.codec_context.thread_count=1
        stream.thread_type='SLICE'
        origin=float((stream.start_time or 0)*stream.time_base)
        container.seek(int((start+origin)/float(stream.time_base)),stream=stream,backward=True)
        for frame in container.decode(stream):
            if frame.time is None: continue
            relative_time=frame.time-origin
            while len(images)<frames and relative_time>=timestamps[len(images)]:
                scale=size/max(frame.width,frame.height)
                array=frame.reformat(width=max(1,round(frame.width*scale)),height=max(1,round(frame.height*scale)),format='rgb24').to_ndarray()
                canvas=np.full((size,size,3),114,dtype=np.uint8)
                y,x=(size-array.shape[0])//2,(size-array.shape[1])//2
                canvas[y:y+array.shape[0],x:x+array.shape[1]]=array
                images.append(canvas)
            if len(images)==frames:break
    if not images:raise ValueError('Не удалось декодировать кадры видео')
    images.extend([images[-1]]*(frames-len(images)))
    return torch.from_numpy(np.stack(images)).permute(3,0,1,2).float()/255


class Extractor:
    def __init__(self, key, assets, pretrained=False, device='cpu'):
        import torch
        self.key,self.device=key,device
        assets=Path(assets)
        if key in ('x3d','slowfast'):
            from pytorchvideo.models.hub import x3d_s,slowfast_r50
            self.model=(x3d_s if key=='x3d' else slowfast_r50)(pretrained=pretrained)
            self.model.blocks[-1].proj=torch.nn.Identity()
            self.model.blocks[-1].activation=None
        elif key in ('videomaev2','videomaev2_large','videomaev2_giant'):
            from transformers import AutoConfig,AutoModel
            folder=assets/key
            config=AutoConfig.from_pretrained(str(folder),trust_remote_code=True,local_files_only=True)
            if pretrained:
                self.model=AutoModel.from_pretrained(str(folder),config=config,trust_remote_code=True,local_files_only=True)
            else:
                self.model=AutoModel.from_config(config,trust_remote_code=True)
        elif key in ('yolov8','yolo11'):
            from ultralytics import YOLO,settings
            settings.update({'sync':False})
            name='yolov8n-cls' if key=='yolov8' else 'yolo11n-cls'
            self.model=YOLO(str(assets/(name+'.pt')) if pretrained else name+'.yaml',task='classify').model
        else:raise ValueError('Unknown model')
        self.model.to(device).eval()
        for p in self.model.parameters():p.requires_grad=False

    def encode(self, x):
        import torch
        x=x.to(self.device)
        with torch.inference_mode():
            if self.key in ('x3d','slowfast'):
                x=(x-.45)/.225
                if self.key=='slowfast':
                    indices=torch.linspace(0,x.shape[2]-1,x.shape[2]//4,device=x.device).long()
                    return self.model([x.index_select(2,indices),x])
                return self.model(x)
            if self.key in ('videomaev2','videomaev2_large','videomaev2_giant'):
                mean=x.new_tensor([.485,.456,.406]).view(1,3,1,1,1)
                std=x.new_tensor([.229,.224,.225]).view(1,3,1,1,1)
                return self.model(pixel_values=(x-mean)/std)
            b,c,t,h,w=x.shape
            captured=[]
            handle=self.model.model[-1].linear.register_forward_pre_hook(lambda module,args:captured.append(args[0]))
            try:self.model(x.permute(0,2,1,3,4).reshape(b*t,c,h,w))
            finally:handle.remove()
            features=captured[0].reshape(b,t,-1)
            # Preserve variation as well as average appearance, without claiming object detection.
            return torch.cat([features.mean(1),features.std(1,unbiased=False)],dim=1)


def load_bundle(path, assets, device='cpu'):
    import torch
    bundle=torch.load(path,map_location='cpu',weights_only=True)
    extractor=Extractor(bundle['model_key'],assets,False,device)
    extractor.model.load_state_dict(bundle['backbone'],strict=True)
    return extractor,bundle


def predict(extractor, bundle, frames):
    import torch
    feature=extractor.encode(frames)
    mean=bundle['feature_mean'].to(feature.device)
    std=bundle['feature_std'].to(feature.device)
    weight=bundle['weight'].to(feature.device)
    bias=bundle['bias'].to(feature.device)
    return torch.sigmoid(((feature-mean)/std)@weight.T+bias).flatten().cpu().tolist()
