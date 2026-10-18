"""Feature-backed PyTorch datasets; text and video keep the same pair identity."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from ml.autolabel.dataset import read_jsonl, digest


class FeatureStore:
    def __init__(self, dataset, features):
        dataset=Path(dataset);features=Path(features)
        self.bags=read_jsonl(dataset/'bags.jsonl'); self.pairs=read_jsonl(dataset/'pairs.jsonl')
        self.clips=read_jsonl(dataset/'clips.jsonl')
        self.metadata=json.loads((features/'extraction.json').read_text())
        if self.metadata['clips_sha256']!=digest(dataset/'clips.jsonl') or self.metadata['pairs_sha256']!=digest(dataset/'pairs.jsonl'):
            raise ValueError('Features and audited data disagree')
        visual=np.load(features/'visual.npz',allow_pickle=False)
        text=np.load(features/'text.npz',allow_pickle=False)
        self.visual={k:torch.from_numpy(v.copy()) for k,v in zip(visual['clip_ids'].tolist(),visual['features'])}
        self.text={k:torch.from_numpy(v.copy()) for k,v in zip(text['pair_ids'].tolist(),text['features'])}
        if not all(c['clip_id'] in self.visual for c in self.clips): raise ValueError('Incomplete visual cache')
        self.action_ids=text['action_ids'].tolist()
        self.prototypes=torch.from_numpy(text['action_features'].copy())
        self.dim=self.prototypes.shape[-1]
        if any(not torch.isfinite(v).all() for v in [*self.visual.values(),*self.text.values(),self.prototypes]):
            raise ValueError('Non-finite feature')
        self.group_split={b['group_id']:b['split'] for b in self.bags}
        if len(self.group_split)!=len(self.bags): raise ValueError('Duplicate source bag')
        for p in self.pairs:
            if self.group_split[p['group_id']]!='train': raise ValueError('Narration leakage')


class VideoBagDataset(Dataset):
    def __init__(self, store, groups):
        self.store=store; self.rows=[b for b in store.bags if b['group_id'] in set(groups)]
        self.windows={b['group_id']:[c['clip_id'] for c in store.clips if c['kind']=='video_bag_window' and c['group_id']==b['group_id']] for b in self.rows}
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        return dict(video=torch.stack([self.store.visual[k] for k in self.windows[row['group_id']]]),
                    target=torch.tensor(float(row['video_target'])),group_id=row['group_id'])


class NarrationDataset(Dataset):
    def __init__(self, store, groups):
        groups=set(groups)
        if any(store.group_split[g]!='train' for g in groups): raise ValueError('Weak supervision can only use train groups')
        self.store=store
        self.rows=[p for p in store.pairs if p['group_id'] in groups and p['eligible_text_loss']]
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]; mask=torch.zeros(len(self.store.action_ids))
        if row['eligible_action_positive_loss']:
            for action in row['action_candidates']: mask[self.store.action_ids.index(action)]=1
        return dict(video=self.store.visual[row['pair_id']],text=self.store.text[row['pair_id']],
                    positive_action_mask=mask,weight=torch.tensor(row['text_loss_weight']),
                    pair_id=row['pair_id'],group_id=row['group_id'])


def collate_bags(rows):
    sizes=[len(r['video']) for r in rows]
    return dict(video=torch.cat([r['video'] for r in rows]),sizes=sizes,
                target=torch.stack([r['target'] for r in rows]),group_ids=[r['group_id'] for r in rows])


def loaders(store,groups):
    bags=VideoBagDataset(store,groups); pairs=NarrationDataset(store,groups)
    # Full batches are appropriate for 32 source groups and 99 weak pairs. No replacement sampling.
    return (DataLoader(bags,batch_size=max(1,len(bags)),collate_fn=collate_bags,shuffle=False),
            DataLoader(pairs,batch_size=max(1,len(pairs)),shuffle=False))
