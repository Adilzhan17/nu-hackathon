"""Training artifacts from original organizer labels; no automatic pseudo-label promotion."""
import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from .dataset import registry, snapshot, csv_rows, digest
from .pipeline import inside, save, jsonl


def export_clip(source,destination,start,end,fps=25,max_side=960):
    import av
    import numpy as np
    from .sampling import frames
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    first=None;count=0
    with av.open(str(destination),'w',options={'movflags':'+faststart'}) as out:
        stream=out.add_stream('libx264',rate=fps);stream.pix_fmt='yuv420p';stream.options={'crf':'20','preset':'veryfast'}
        stream.codec_context.thread_count=2
        stream.time_base=Fraction(1,90000);stream.codec_context.time_base=Fraction(1,90000)
        for ts,im in frames(source,start,end,fps,max_side=max_side):
            if first is None:
                first=ts;stream.width=im.width//2*2;stream.height=im.height//2*2
            frame=av.VideoFrame.from_ndarray(np.asarray(im),format='rgb24').reformat(width=stream.width,height=stream.height,format='yuv420p')
            frame.pts=round((ts-first)*90000);frame.time_base=Fraction(1,90000)
            for packet in stream.encode(frame):out.mux(packet)
            count+=1
        for packet in stream.encode():out.mux(packet)
    if count<2:raise ValueError('No temporal sequence in clip')
    return dict(clip_first_source_sec=first,clip_frames=count,clip_sha256=digest(destination),clip_fps=fps)


def export(args):
    root=args.root.resolve();dest=inside(args.output,root/'data/autolabel')
    if dest.exists():raise FileExistsError('Use new training artifact directory')
    before=snapshot(root);reg,reviews,observations=registry(root)
    originals=csv_rows(root/'data/training/manifest.csv')
    rows=[r for r in originals if r['split']=='train']
    dest.mkdir(parents=True);(dest/'clips').mkdir()
    group_counts=Counter(r['group_id'] for r in rows)
    class_groups=Counter({r['group_id']:r['target'] for r in rows}.values())
    def process(item):
        i,r=item;v=reg[r['video']]
        if v['split']!='train' or v['group_id']!=r['group_id']:raise ValueError('Split mismatch')
        name=f'clips/source_{i:04d}.mp4'
        meta=export_clip(root/'data/raw/open'/r['video'],dest/name,float(r['start_sec']),float(r['end_sec']))
        return dict(r,target='unknown' if int(r['target']) else '0',label='unknown' if int(r['target']) else 'normal',
                    organizer_video_target=r['target'],organizer_video_label=r['label'],clip=name,**meta,source_manifest_row=originals.index(r),
                    label_origin='organizer',supervision='video_label_only_positive_interval_unknown',
                    action_type='unresolved',eligible_for_binary_training=False if int(r['target']) else True,eligible_for_action_training=False,
                    interval_target='unknown' if int(r['target']) else '0',
                    group_balanced_weight=1/(group_counts[r['group_id']]*class_groups[r['target']]))
    done=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(process,enumerate(rows)):
            done.append(row)
            if len(done)%10==0:print(json.dumps(dict(exported_source_clips=len(done),total=len(rows))),flush=True)
    with (dest/'organizer_train.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(done[0]));w.writeheader();w.writerows(done)
    source_events=[];quarantine=[]
    for g,obs in observations.items():
        if not any(r['group_id']==g for r in rows):continue
        for o in obs:
            record=dict(o,canonical_group_id=g,split='train',label_origin='user_transcript',
                        eligible_for_action_training=False,reason_not_training='Transcript timing and action require visual confirmation.')
            (quarantine if o['status']=='quarantined' else source_events).append(record)
    jsonl(dest/'user_event_supervision.jsonl',source_events);jsonl(dest/'quarantined_sources.jsonl',quarantine)
    if snapshot(root)!=before:raise RuntimeError('Protected input files changed')
    summary=dict(source_train_clips=len(done),independent_train_groups=len(group_counts),
                 interval_classes=dict(Counter(r['label'] for r in done)),organizer_video_group_classes=dict(class_groups),
                 user_event_candidates=len(source_events),quarantined_sources=len(quarantine),
                 validation_manifest='data/training/manifest.csv',validation_policy='Read frozen val rows from original manifest; no new labels or augmented val copies.',
                 original_manifest_sha256=digest(root/'data/training/manifest.csv'),protected_inputs=before,
                 new_independent_videos=0,pseudo_labels_promoted=0,
                 limitations=['Organizer violation labels do not identify the exact rule or action type.',
                              'Positive source windows have unknown interval targets and are not interval ground truth.',
                              'More clips from the same source are correlated examples, not new independent scenes.'])
    save(dest/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))


class OrganizerClipDataset:
    """Torch-compatible Dataset. Train uses exported clips; val reads frozen original windows."""
    def __init__(self,root,artifact_dir,split='train',frames=16,size=224):
        self.root=Path(root);self.artifact_dir=Path(artifact_dir);self.split=split;self.num_frames=frames;self.size=size
        if split not in {'train','val'} or frames<2 or size<16:raise ValueError('Invalid dataset parameters')
        meta=json.loads((self.artifact_dir/'summary.json').read_text())
        if digest(self.root/'data/training/manifest.csv')!=meta['original_manifest_sha256']:
            raise ValueError('Frozen manifest changed')
        if split=='train' and (self.artifact_dir/'SUPERSEDED.json').exists():
            raise ValueError('Positive source windows are unknown, not interval ground truth. Use VideoBagDataset with training_v2.')
        self.rows=csv_rows(self.artifact_dir/'organizer_train.csv') if split=='train' else [r for r in csv_rows(self.root/'data/training/manifest.csv') if r['split']=='val']
        if any(r['split']!=split for r in self.rows):raise ValueError('Split leakage')
        if split=='train' and any(r['target'] not in {'0',0} for r in self.rows):
            raise ValueError('Source-positive windows are unknown; use VideoBagDataset for video-level supervision')
        self.sample_weights=[float(r['group_balanced_weight']) for r in self.rows] if split=='train' else None

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        import av
        import numpy as np
        import torch
        r=self.rows[index]
        if self.split=='train':
            path=inside(self.artifact_dir/r['clip'],self.artifact_dir/'clips')
            start=0.;end=float(r['end_sec'])-float(r['clip_first_source_sec'])
        else:path=self.root/'data/raw/open'/r['video'];start=float(r['start_sec']);end=float(r['end_sec'])
        targets=np.linspace(start,max(start,end-.08),self.num_frames);images=[]
        with av.open(str(path)) as c:
            s=c.streams.video[0];origin=float((s.start_time or 0)*s.time_base)
            c.seek(int((start+origin)/float(s.time_base)),stream=s,backward=True)
            for f in c.decode(video=0):
                if f.pts is None:continue
                ts=float(f.pts*s.time_base)-origin
                while len(images)<self.num_frames and ts>=targets[len(images)]:
                    scale=self.size/max(f.width,f.height)
                    image=f.reformat(width=max(1,round(f.width*scale)),height=max(1,round(f.height*scale)),format='rgb24').to_ndarray()
                    canvas=np.full((self.size,self.size,3),114,dtype=np.uint8)
                    y=(self.size-image.shape[0])//2;x=(self.size-image.shape[1])//2
                    canvas[y:y+image.shape[0],x:x+image.shape[1]]=image;images.append(canvas)
                if len(images)==self.num_frames:break
        if not images:raise ValueError('Undecodable training clip')
        images.extend([images[-1]]*(self.num_frames-len(images)))
        tensor=torch.from_numpy(np.stack(images)).permute(3,0,1,2).float()/255
        # Normalize or augment in the selected model's transform, not silently here.
        return tensor,int(r['target'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True)
    export(p.parse_args())


if __name__=='__main__':main()
