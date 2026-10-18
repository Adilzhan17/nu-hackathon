"""Small temporal residual adapter; binary supervision belongs to a whole video bag."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TemporalNarrationModel(nn.Module):
    def __init__(self,dim,bottleneck=48):
        super().__init__();self.dim=dim;self.bottleneck=bottleneck
        self.adapter=nn.Sequential(nn.Linear(3*dim,bottleneck),nn.GELU(),nn.Linear(bottleneck,dim))
        nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
        self.verdict=nn.Linear(dim,1)
    def encode(self,frames):
        mean=frames.mean(dim=1)
        positions=torch.linspace(-1,1,frames.shape[1],device=frames.device).view(1,-1,1)
        trend=(frames*positions).mean(dim=1)
        change=frames[:,-1]-frames[:,0]
        residual=self.adapter(torch.cat([mean,trend,change],dim=-1))
        return F.normalize(mean+.2*residual,dim=-1)
    def video_logits(self,frames,sizes):
        scores=self.verdict(self.encode(frames)).squeeze(-1)*10
        # Smooth MIL pooling: no positive or negative window labels are created.
        return torch.stack([.5*(torch.logsumexp(part/.5,dim=0)-math.log(len(part))) for part in scores.split(sizes)])


def narration_loss(model,batch,prototypes):
    z=model.encode(batch['video'])
    per_pair=1-(z*batch['text']).sum(-1)
    text=(per_pair*batch['weight']).sum()/batch['weight'].sum().clamp_min(1)
    positive=batch['positive_action_mask']
    # Positive-only prototype matching: unnamed actions are UNKNOWN, never absent/negative.
    action=((1-z@prototypes.T)*positive).sum()/positive.sum().clamp_min(1)
    return text+.25*action,dict(text=float(text.detach()),action=float(action.detach()))
