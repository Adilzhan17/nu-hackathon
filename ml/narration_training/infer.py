"""Load an experimental checkpoint and infer whole-video scores from frozen local features."""
import argparse
from pathlib import Path
import torch
from .data import FeatureStore
from .model import TemporalNarrationModel
from .prepare import write_jsonl
from .train import predict

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--features',type=Path,required=True)
    p.add_argument('--split',choices=['train','val'],default='val');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    if not a.output.resolve().is_relative_to(Path('runs/narration_training').resolve()): raise ValueError('Wrong output root')
    if a.output.exists(): raise FileExistsError(a.output)
    store=FeatureStore(a.dataset,a.features);checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint['encoder']['model_source']!=store.metadata['model_source']: raise ValueError('Encoder mismatch')
    model=TemporalNarrationModel(**checkpoint['model_config']);model.load_state_dict(checkpoint['state_dict'])
    rows=predict(store,model,[b['group_id'] for b in store.bags if b['split']==a.split])
    write_jsonl(a.output,rows);print(dict(video_groups=len(rows),scope='whole_video',calibrated=False))

if __name__=='__main__': main()
