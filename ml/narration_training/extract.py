"""Frozen, entirely local SigLIP2 encoding. Bounded GPU lease; no audio or network."""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, SiglipModel
from ml.autolabel.dataset import read_jsonl, digest
from ml.autolabel.sampling import frames
from .prepare import MODEL_ID, MODEL_REVISION, write_json, validate_prepared_split


@contextlib.contextmanager
def gpu_lock(path, wait_sec=30):
    with open(path,'a') as f:
        deadline=time.monotonic()+wait_sec
        while True:
            try:
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>deadline: raise TimeoutError('Shared GPU busy; retry later, no process interrupted')
                time.sleep(.5)
        try: yield
        finally: fcntl.flock(f,fcntl.LOCK_UN)


def extract(args):
    root=args.root.resolve(); data=args.dataset.resolve(); out=args.output.resolve()
    split_policy=validate_prepared_split(root,data)
    if not out.is_relative_to(root/'runs/narration_training'): raise ValueError('Invalid output root')
    out.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    torch.set_num_threads(4)
    source=json.loads((args.model/'source.json').read_text())
    if (source['repo'],source['revision']) != (MODEL_ID,MODEL_REVISION): raise ValueError('Wrong model revision')
    for name,expected in source['files'].items():
        if digest(args.model/name)!=expected: raise ValueError('Model file hash mismatch: '+name)
    if 'apache-2.0' not in (args.model/'README.md').read_text().lower(): raise ValueError('License not verified')
    clips=read_jsonl(data/'clips.jsonl'); pairs=read_jsonl(data/'pairs.jsonl')
    if args.limit_groups:
        selected=sorted({p['group_id'] for p in pairs})[:args.limit_groups]
        clips=[c for c in clips if c['group_id'] in selected]
    bags={b['group_id']:b for b in read_jsonl(data/'bags.jsonl')}
    resolved={}
    for group in {c['group_id'] for c in clips}:
        bag=bags[group]
        available=[root/'data/raw/open'/name for name in [bag['video'],*bag['aliases']]
                   if (root/'data/raw/open'/name).is_file()]
        if not available or digest(available[0])!=bag['sha256']:
            raise ValueError('Missing or changed source bytes: '+group)
        resolved[group]=available[0]
    cached={}; evidence={}
    if (out/'visual.npz').exists():
        old=np.load(out/'visual.npz',allow_pickle=False)
        oldmeta=json.loads((out/'extraction.json').read_text())
        if oldmeta['clips_sha256']!=digest(data/'clips.jsonl'): raise ValueError('Stale cache')
        cached=dict(zip(old['clip_ids'].tolist(),old['features']))
        evidence=json.loads((out/'frame_timestamps.json').read_text())
    started=time.monotonic()
    # The lease covers model allocation, both towers and all GPU work.
    with gpu_lock(args.lock):
        device='cuda' if torch.cuda.is_available() else 'cpu'
        dtype=torch.bfloat16 if device=='cuda' else torch.float32
        model=SiglipModel.from_pretrained(args.model,local_files_only=True,torch_dtype=dtype).eval().to(device)
        count=sum(p.numel() for p in model.parameters())
        if count>3_000_000_000: raise ValueError('Parameter cap exceeded')
        tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        processor=AutoImageProcessor.from_pretrained(args.model,local_files_only=True,use_fast=False)
        max_tokens=model.config.text_config.max_position_embeddings
        vocabulary=json.loads((data/'action_vocabulary.json').read_text())
        captions=[p['caption'] for p in pairs]+vocabulary['prompts']
        token_counts=[len(tokenizer(t,add_special_tokens=True)['input_ids']) for t in captions]
        encoded=[]
        with torch.inference_mode():
            for i in range(0,len(captions),32):
                batch=tokenizer(captions[i:i+32],padding='max_length',max_length=max_tokens,truncation=True,return_tensors='pt')
                feat=model.get_text_features(input_ids=batch['input_ids'].to(device)).float()
                encoded.append(torch.nn.functional.normalize(feat,dim=-1).cpu().numpy())
        encoded=np.concatenate(encoded)
        np.savez_compressed(out/'text.npz',pair_ids=np.array([p['pair_id'] for p in pairs]),
                            features=encoded[:len(pairs)],action_ids=np.array(vocabulary['actions']),
                            action_features=encoded[len(pairs):])
        write_json(out/'text_token_audit.json',dict(max_tokens=max_tokens,
          pairs=[dict(pair_id=p['pair_id'],tokens=n,truncated=n>max_tokens) for p,n in zip(pairs,token_counts)],
          truncated_pairs=sum(n>max_tokens for n in token_counts[:len(pairs)])))
        def save():
            ids=sorted(cached)
            np.savez_compressed(out/'visual.npz',clip_ids=np.array(ids),features=np.stack([cached[k] for k in ids]))
            write_json(out/'frame_timestamps.json',evidence)
            write_json(out/'extraction.json',dict(model_source=source,parameters=count,license='Apache-2.0',
                split_version=split_policy['version'],independent_test=False,
                clips_sha256=digest(data/'clips.jsonl'),pairs_sha256=digest(data/'pairs.jsonl'),
                frames_per_interval=args.frames,visual_clips=len(ids),feature_dim=encoded.shape[1],
                elapsed_seconds=time.monotonic()-started,device=device,precision=str(dtype),
                temporal_order_preserved=True,audio_decoded=False,local_only=True))
        completed=0
        for clip in clips:
            if clip['clip_id'] in cached: continue
            if time.monotonic()-started>args.max_seconds: save();raise TimeoutError('Bounded extraction budget exhausted; cache can resume')
            start,end=clip['start_sec'],clip['end_sec']
            sampled=list(frames(resolved[clip['group_id']],start,end-1e-4,args.frames/(end-start),max_side=768))[:args.frames]
            if not sampled: raise ValueError('No frames: '+clip['clip_id'])
            real_pts=[t for t,_ in sampled]
            images=[im for _,im in sampled]
            while len(images)<args.frames: images.append(images[-1])
            with torch.inference_mode():
                pixels=processor(images=images,return_tensors='pt')['pixel_values'].to(device,dtype=dtype)
                feat=torch.nn.functional.normalize(model.get_image_features(pixel_values=pixels).float(),dim=-1)
            cached[clip['clip_id']]=feat.cpu().numpy().astype(np.float32)
            evidence[clip['clip_id']]=dict(actual_timestamps=real_pts,repeated_last_frame=args.frames-len(real_pts),
                                         start_sec=start,end_sec=end,decoded_video=str(resolved[clip['group_id']].relative_to(root)))
            completed+=1
            if completed%16==0: save();print(json.dumps(dict(encoded=len(cached),new=completed,elapsed=round(time.monotonic()-started,1))),flush=True)
        save()
        del model
        if device=='cuda': torch.cuda.empty_cache()
    print(json.dumps(dict(encoded=len(cached),elapsed_seconds=round(time.monotonic()-started,2),parameters=count)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--limit-groups',type=int,default=0);p.add_argument('--frames',type=int,default=8)
    p.add_argument('--max-seconds',type=int,default=600)
    extract(p.parse_args())
