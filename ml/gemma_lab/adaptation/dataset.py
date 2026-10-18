"""Local PIL frame Dataset/DataLoader for the audited Gemma adapter experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset


class GemmaAdaptationDataset(Dataset):
    """Never inserts references, captions or organizer answers into the prompt.

    __getitem__ returns frames, prompt, target, loss_weight and metadata. Pass
    frames/prompt/target directly to Backend.encode; apply loss_weight to loss.
    Metadata contains references strictly for audit/evaluation outside the model.
    """
    def __init__(self, manifest, run_root=None, *, allow_evaluation=False, verify_frames=True):
        self.manifest = Path(manifest).resolve()
        self.run_root = Path(run_root).resolve() if run_root else self.manifest.parent.parent
        self.verify_frames = verify_frames
        protocol = json.loads((self.run_root / 'frozen_group_protocol.json').read_text())
        self.rows = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        seen = set()
        for row in self.rows:
            if row['sample_id'] in seen:
                raise ValueError('Duplicate prepared sample ID')
            seen.add(row['sample_id'])
            group = row['group_id']
            part = protocol['group_membership'].get(group)
            if row.get('partition') != part:
                raise ValueError('Sample partition differs from frozen source group')
            if row['video_sha256'] != protocol['source_identity'][group]['video_sha256']:
                raise ValueError('Prepared source video hash differs from frozen protocol')
            if not allow_evaluation and (part != 'fit' or row.get('split') != 'train'):
                raise ValueError('Evaluation source entered training Dataset')
            if not allow_evaluation and row.get('kind') not in {'human_action', 'weak_caption'}:
                raise ValueError('Unsupported training supervision type')
            if not allow_evaluation and (not isinstance(row.get('target'), str) or not row['target'].strip()):
                raise ValueError('Training requires an explicit nonempty assistant target')
            if row.get('loss_weight') not in {0., .2, 1.}:
                raise ValueError('Unexpected supervision weight')
            max_frames = 16 if allow_evaluation and row.get('kind') == 'whole_video_eval' else 4
            if not 2 <= len(row['frames']) <= max_frames or row['frame_paths'] != [frame['path'] for frame in row['frames']]:
                raise ValueError('Prepared frame list mismatch')
            if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
                raise ValueError('A generic prompt is required')
            for frame in row['frames']:
                self._frame_path(frame)

    def _frame_path(self, frame):
        path = (self.run_root / frame['path']).resolve()
        if not path.is_relative_to((self.run_root / 'prepared/frames').resolve()):
            raise ValueError('Prepared frame escapes its dedicated frame directory')
        return path

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        images = []
        for frame in row['frames']:
            path = self._frame_path(frame)
            if self.verify_frames and hashlib.sha256(path.read_bytes()).hexdigest() != frame['sha256']:
                raise ValueError('Prepared frame SHA256 changed')
            with Image.open(path) as image:
                if image.size != (frame['width'], frame['height']):
                    raise ValueError('Prepared frame dimensions changed')
                images.append(image.convert('RGB'))
        return dict(frames=images, prompt=row['prompt'], target=row.get('target'),
                    loss_weight=row['loss_weight'], metadata=row)


def collate_one(batch):
    if len(batch) != 1:
        raise ValueError('This memory-bounded experiment requires batch_size=1')
    return batch[0]


def make_loader(manifest, run_root=None, *, shuffle=False, seed=7301, allow_evaluation=False):
    import torch
    dataset = GemmaAdaptationDataset(manifest, run_root, allow_evaluation=allow_evaluation)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, collate_fn=collate_one,
                      num_workers=0, generator=generator, pin_memory=False)
