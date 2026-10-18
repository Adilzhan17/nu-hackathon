"""Explicit consumer of train-only weak scenario records, never a clip-target dataset."""
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from ml.autolabel.dataset import read_jsonl,csv_rows

class OrganizerScenarioDataset(Dataset):
    def __init__(self,root,export,groups=None):
        self.root=Path(root).resolve();rows=read_jsonl(export)
        registry={r['video']:r for r in csv_rows(self.root/'data/training/manifest.csv')}
        self.rows=[]
        for row in rows:
            source=registry.get(row['video'])
            if not source or source['split']!='train' or row['split']!='train':raise ValueError('Non-train organizer record')
            if (source['group_id'],source['sha256'])!=(row['group_id'],row['video_sha256']):raise ValueError('Organizer source identity mismatch')
            if row['source_type']!='organizer_provided' or row['supervision']!='weak_whole_video_scenario':raise ValueError('Wrong supervision scope')
            if row['event_targets'] or row['binary_window_target'] is not None or row['eligible_for_strong_event_training']:
                raise ValueError('Scenario claim was incorrectly promoted to a strong target')
            if not row['eligible_for_weak_scenario_training']:continue
            if groups is None or row['group_id'] in set(groups):self.rows.append(row)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        # No full original-message read: that message also contains disclosed validation answers.
        return dict(video_path=str(self.root/'data/raw/open'/row['video']),group_id=row['group_id'],
          scenario_text=row['original_text'],expected_sequence=row['expected_sequence'],trust=row['trust'],
          source_claim_id=row['source_claim_id'],source_type='organizer_provided',
          supervision='weak_whole_video_scenario',event_targets=[],binary_window_target=None)

def scenario_loader(root,export,groups=None,batch_size=4):
    return DataLoader(OrganizerScenarioDataset(root,export,groups),batch_size=batch_size,collate_fn=list,shuffle=False)
