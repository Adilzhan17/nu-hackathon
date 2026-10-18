"""Video-level training with unknown positive-window labels (multiple-instance supervision)."""
import argparse
from collections import defaultdict,Counter
import csv
import json
from pathlib import Path
from .dataset import csv_rows,registry,snapshot,digest
from .deep import coverage_windows
from .pipeline import inside,save,jsonl
from .sampling import metadata


def export(root,legacy,output):
    root=Path(root).resolve();legacy=inside(legacy,root/'data/autolabel');output=inside(output,root/'data/autolabel')
    if output.exists():raise FileExistsError('New artifact version required')
    reg,_,_=registry(root);original=csv_rows(root/'data/training/manifest.csv');groups=defaultdict(list)
    for row in original:
        if row['split']=='train':groups[row['group_id']].append(row)
    bags=[]
    for group,rows in sorted(groups.items()):
        targets={int(r['target']) for r in rows}
        if len(targets)!=1:raise ValueError('Contradictory source video labels')
        target=targets.pop();video=rows[0]['video'];source=reg[video]
        if source['split']!='train' or source['group_id']!=group:raise ValueError('Group leakage')
        duration=metadata(root/'data/raw/open'/video)['duration_sec']
        windows=[dict(start_sec=c['start_sec'],end_sec=c['end_sec'],interval_target=None if target else 0,
                      interval_review_status='unknown' if target else 'organizer_video_negative') for c in coverage_windows(duration)]
        bags.append(dict(video=video,group_id=group,split='train',source_sha256=source['sha256'],duration_sec=duration,
                         organizer_video_target=target,label_scope='whole_source_video',label_origin='organizer',
                         interval_localization_ground_truth=False,windows=windows,
                         organizer_anchors=sorted({float(r['anchor_sec']) for r in rows if target and r.get('anchor_sec')})))
    output.mkdir(parents=True);jsonl(output/'video_bags_train.jsonl',bags)
    old=csv_rows(legacy/'organizer_train.csv');new=[]
    for r in old:
        target=int(r.get('organizer_video_target',r['target']))
        new.append(dict(video=r['video'],group_id=r['group_id'],split='train',start_sec=r['start_sec'],end_sec=r['end_sec'],
                        clip=str((legacy/r['clip']).relative_to(root)),clip_sha256=r['clip_sha256'],
                        organizer_video_target=target,organizer_video_label=r.get('organizer_video_label',r['label']),
                        interval_target='unknown' if target else '0',
                        interval_review_status='unknown' if target else 'organizer_video_negative',
                        eligible_for_supervised_interval_training=False if target else True,
                        localization_scope='organizer_anchor_context_not_confirmed_bounds' if target else 'source_video_negative'))
    with (output/'source_windows.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(new[0]));w.writeheader();w.writerows(new)
    class_groups=Counter(b['organizer_video_target'] for b in bags)
    save(output/'summary.json',dict(policy_version=2,train_video_bags=len(bags),video_classes=dict(class_groups),
        coverage_windows=sum(len(b['windows']) for b in bags),
        unknown_positive_windows=sum(len(b['windows']) for b in bags if b['organizer_video_target']==1),
        original_exported_clips=len(new),unknown_anchor_clips=sum(r['interval_target']=='unknown' for r in new),
        interval_positive_ground_truth=0,source_manifest_sha256=digest(root/'data/training/manifest.csv'),
        protected_inputs=snapshot(root),label_policy='Only whole-video positive labels. Every unconfirmed interval of a positive video is unknown.',
        validation_policy='Frozen original manifest, unchanged.',independent_train_groups=len(bags)))
    save(legacy/'SUPERSEDED.json',dict(replacement=str(output.relative_to(root)),
         reason='Version 1 exposed weak organizer anchor windows with binary targets. Positive intervals are now unknown; use video-level bags.',
         positive_window_labels_must_not_be_used_as_ground_truth=True))
    print(json.dumps(json.loads((output/'summary.json').read_text()),ensure_ascii=False,indent=2))


class VideoBagDataset:
    """Return all covering windows in a train video and ONE label for the entire bag.

    Use batch_size=1 (variable bag length); encode windows in micro-batches and pool
    at video level. Do not broadcast video_target onto its window predictions.
    """
    def __init__(self,root,artifact_dir,frames=16,size=224):
        self.root=Path(root);self.artifact_dir=Path(artifact_dir);self.num_frames=frames;self.size=size
        if frames<2 or size<16:raise ValueError('Invalid sampling parameters')
        meta=json.loads((self.artifact_dir/'summary.json').read_text())
        if meta['policy_version']!=2 or digest(self.root/'data/training/manifest.csv')!=meta['source_manifest_sha256']:
            raise ValueError('Dataset policy or frozen manifest mismatch')
        self.rows=read_rows(self.artifact_dir/'video_bags_train.jsonl')
        manifest=csv_rows(self.root/'data/training/manifest.csv')
        identities={(r['video'],r['group_id'],r['sha256'],int(r['target'])) for r in manifest if r['split']=='train'}
        if len({r['group_id'] for r in self.rows})!=len(self.rows):raise ValueError('Repeated source group')
        if any((r['video'],r['group_id'],r['source_sha256'],r['organizer_video_target']) not in identities for r in self.rows):
            raise ValueError('Frozen source/group/label mismatch')
        counts=Counter(r['organizer_video_target'] for r in self.rows)
        self.sample_weights=[1/counts[r['organizer_video_target']] for r in self.rows]
        for r in self.rows:
            if r['split']!='train' or (r['organizer_video_target']==1 and any(w['interval_target'] is not None for w in r['windows'])):
                raise ValueError('Invalid split or propagated positive interval label')
            for w in r['windows']:
                if not 0<=w['start_sec']<w['end_sec']<=r['duration_sec']:raise ValueError('Invalid temporal bounds')
                if r['organizer_video_target']==0 and w['interval_target']!=0:raise ValueError('Negative source label mismatch')

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        import numpy as np
        import torch
        from .sampling import frames
        r=self.rows[index];bag=[]
        for w in r['windows']:
            # Decode selected PTS from the original, then sample T frames without dropping scene edges.
            samples=list(frames(self.root/'data/raw/open'/r['video'],w['start_sec'],w['end_sec'],
                                max(2.,self.num_frames/(w['end_sec']-w['start_sec'])),max_side=self.size))
            if len(samples)<2:raise ValueError('Undecodable bag interval')
            picked=[samples[round(i*(len(samples)-1)/(self.num_frames-1))][1] for i in range(self.num_frames)]
            canvases=[]
            for im in picked:
                image=np.asarray(im);canvas=np.full((self.size,self.size,3),114,dtype=np.uint8)
                y=(self.size-image.shape[0])//2;x=(self.size-image.shape[1])//2
                canvas[y:y+image.shape[0],x:x+image.shape[1]]=image;canvases.append(canvas)
            bag.append(torch.from_numpy(np.stack(canvases)).permute(3,0,1,2).float()/255)
        return dict(frames=torch.stack(bag),video_target=r['organizer_video_target'],group_id=r['group_id'],
                    interval_targets=torch.tensor([-1 if w['interval_target'] is None else w['interval_target'] for w in r['windows']]),
                    interval_label_mask=torch.tensor([w['interval_target'] is not None for w in r['windows']]))


def read_rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--legacy',type=Path,default=Path('data/autolabel/training_v1'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();export(a.root,a.legacy,a.output)
