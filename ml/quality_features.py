"""Offline full-frame + native center-crop features for weakly labeled videos."""
from pathlib import Path


def load_views(path, start, end, frames=16, size=224):
    import av
    import numpy as np
    import torch
    from PIL import Image
    stamps=np.linspace(start,max(start,end-.08),frames);images=[]
    with av.open(str(path)) as video:
        stream=video.streams.video[0];stream.thread_type='AUTO'
        video.seek(max(0,int(start/float(stream.time_base))),stream=stream,backward=True)
        for frame in video.decode(stream):
            if frame.time is None:continue
            if len(images)>=frames:break
            if frame.time<stamps[len(images)]:continue
            im=frame.to_image();w,h=im.size;scale=size/max(w,h)
            fit=im.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.BILINEAR)
            canvas=Image.new('RGB',(size,size),(114,114,114));canvas.paste(fit,((size-fit.width)//2,(size-fit.height)//2))
            side=min(w,h);crop=im.crop(((w-side)//2,(h-side)//2,(w+side)//2,(h+side)//2)).resize((size,size),Image.Resampling.BICUBIC)
            pair=np.stack([np.array(canvas),np.array(crop)])
            while len(images)<frames and frame.time>=stamps[len(images)]:images.append(pair)
    if not images:raise ValueError('No decoded frames')
    images.extend([images[-1]]*(frames-len(images)))
    return torch.from_numpy(np.stack(images)).permute(1,4,0,2,3).float()/255


class QualityEncoder:
    def __init__(self, family, assets, pretrained=True, device='cpu'):
        import torch
        from transformers import AutoConfig,VideoMAEForVideoClassification,Dinov2Model
        self.family=family;self.device=device;self.assets=Path(assets)
        folder=self.assets/('quality_'+family)
        config=AutoConfig.from_pretrained(folder,local_files_only=True)
        cls=VideoMAEForVideoClassification if family=='action' else Dinov2Model
        self.model=cls.from_pretrained(folder,local_files_only=True) if pretrained else cls(config)
        self.model.to(device).eval()
        for p in self.model.parameters():p.requires_grad=False
        if sum(p.numel() for p in self.model.parameters())>3_000_000_000:raise ValueError('3B parameter limit exceeded')

    def encode(self,x):
        import torch
        x=x.to(self.device);b,v,c,t,h,w=x.shape
        mean=x.new_tensor([.485,.456,.406]).view(1,1,3,1,1,1)
        std=x.new_tensor([.229,.224,.225]).view(1,1,3,1,1,1)
        x=(x-mean)/std
        with torch.inference_mode():
            if self.family=='action':
                x=x.permute(0,1,3,2,4,5).reshape(b*v,t,c,h,w)
                features=self.model.videomae(pixel_values=x).last_hidden_state.mean(1)
                return self.model.fc_norm(features).reshape(b,-1)
            x=x[:,:,:, [2,6,10,14]].permute(0,1,3,2,4,5).reshape(b*v*4,c,h,w)
            output=self.model(pixel_values=x).last_hidden_state
            # CLS and pooled spatial patches retain complementary information.
            f=torch.cat([output[:,0],output[:,1:].mean(1)],1).reshape(b,v,4,-1)
            return torch.cat([f.mean(2),f.std(2,unbiased=False)],-1).reshape(b,-1)
