"""Synthetic source/provenance tests for the opt-in and active 38/2 preparation."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ml.autolabel.dataset import digest, read_jsonl
from ml.split95 import activate_split, prepare as prepare_split
from .data import FeatureStore
from .prepare import PAIR_PATH, prepare, validate_prepared_split
from .train import folds


class NarrationSplitPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.manifest=self.root/'data/training/manifest.csv';self.manifest.parent.mkdir(parents=True)
        rows=[]
        for i in range(40):
            video=f'videos/synthetic_{i}.mov';path=self.root/'data/raw/open'/video
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(f'synthetic source {i}'.encode())
            rows.append(dict(video=video,group_id=f'group_{i}',sha256=digest(path),split='train' if i<32 else 'val',target=str(i%2)))
        with self.manifest.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        report=prepare_split(self.root);activate_split(self.root)
        promoted=next(e for e in report['entities'] if e['source_split']=='val' and e['split']=='train')
        heldout=next(e for e in report['entities'] if e['split']=='diagnostic_test')
        chosen=[rows[0],next(r for r in rows if r['sha256']==promoted['sha256']),next(r for r in rows if r['sha256']==heldout['sha256'])]
        captions=[f'Synthetic source sentence {i}.' for i in range(3)]
        (self.root/'audio.txt').write_text('\n'.join(captions)+'\n')
        sections=[];observations=[];pairs=[]
        for i,(row,caption) in enumerate(zip(chosen,captions)):
            section=f'section_{i}';lines=[i+1,i+1]
            sections.append(dict(section=section,video=row['video'],start_line=i+1,end_line=i+1,text=caption,mapping='explicit_heading'))
            observations.append(dict(video=row['video'],group_id=row['group_id'],section=section,start_sec=2,end_sec=3,
                action='cash_received',observation=caption,status='transcribed',source='audio.txt',source_lines=lines))
            pairs.append(dict(pair_id=f'pair_{i}',observation_id=f'observation_{i:04d}',video=row['video'],group_id=row['group_id'],
                split=row['split'],start_sec=.5,end_sec=4,observation_start_sec=2,observation_end_sec=3,
                action_from_source='cash_received',caption=caption,source='audio.txt',source_lines=lines,
                source_mapping='explicit_heading',annotation_status='transcribed',timing_precision='approximate'))
        self.write('data/transcripts/observations.jsonl',''.join(json.dumps(r)+'\n' for r in observations))
        self.write('data/transcripts/sources.json',json.dumps(dict(source_sha256=digest(self.root/'audio.txt'),sections=sections)))
        self.write('data/review/event_review.csv','video,group_id\n')
        self.write(PAIR_PATH,''.join(json.dumps(r)+'\n' for r in pairs))
        self.output=self.root/'runs/narration_training/dataset95'

    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)

    def prepare(self,output=None,**kwargs):
        output=output or self.output
        with mock.patch('ml.narration_training.prepare.metadata',return_value={'duration_sec':10.}),contextlib.redirect_stdout(io.StringIO()):
            return prepare(self.root,output,**kwargs)

    def test_default_active_policy_prepares_38_2_without_promoted_or_test_captions(self):
        original=self.manifest.read_bytes()
        summary=self.prepare()
        self.assertEqual(summary['bag_splits'],{'train':38,'val':2})
        self.assertEqual(summary['canonical_bag_splits'],{'train':38,'diagnostic_test':2})
        self.assertFalse(summary['independent_test'])
        self.assertEqual(summary['valid_pairs'],1)
        self.assertEqual(summary['excluded_pairs'],2)
        pair,=read_jsonl(self.output/'pairs.jsonl')
        self.assertEqual(pair['group_id'],'group_0')
        self.assertTrue(pair['provenance_checks']['train_only'])
        self.assertTrue(pair['provenance_checks']['effective_train_only'])
        self.assertFalse(pair['human_confirmed'])
        self.assertIsNone(pair['window_violation_target'])
        self.assertEqual(self.manifest.read_bytes(),original)
        protected=json.loads((self.output/'protected_inputs.json').read_text())
        self.assertIn('data/training/manifest.csv',protected)
        self.assertIn('data/training/split95_v1/report.json',protected)
        self.assertTrue(all(digest(self.root/name)==sha for name,sha in protected.items()))

    def test_original_split_explicitly_reproduces_32_8_even_with_active_95(self):
        summary=self.prepare(original_split=True)
        self.assertEqual(summary['bag_splits'],{'train':32,'val':8})
        pair,=read_jsonl(self.output/'pairs.jsonl')
        self.assertNotIn('effective_train_only',pair['provenance_checks'])
        self.assertEqual(validate_prepared_split(self.root,self.output)['version'],'original_32_8')
        # Historical datasets did not contain this metadata file.
        (self.output/'split_policy.json').unlink()
        self.assertEqual(validate_prepared_split(self.root,self.output)['version'],'original_32_8')

    def test_explicit_version_is_same_policy_as_active_default(self):
        self.prepare()
        other=self.root/'runs/narration_training/explicit95'
        self.prepare(other,split_version='split95_v1')
        for name in ('bags.jsonl','clips.jsonl','pairs.jsonl','split_policy.json'):
            self.assertEqual((self.output/name).read_bytes(),(other/name).read_bytes())

    def test_fresh_directory_required_and_source_outputs_not_overwritten(self):
        self.prepare()
        content=(self.output/'bags.jsonl').read_bytes()
        with self.assertRaises(FileExistsError):self.prepare()
        self.assertEqual((self.output/'bags.jsonl').read_bytes(),content)
        with self.assertRaises(ValueError):self.prepare(self.root/'runs/narration_training/conflicting',original_split=True,split_version='split95_v1')

    def test_wrong_bag_assignment_and_heldout_caption_are_rejected(self):
        self.prepare();bags=read_jsonl(self.output/'bags.jsonl');pairs=read_jsonl(self.output/'pairs.jsonl')
        held=next(b for b in bags if b['split']=='val')
        altered=[dict(b,split='train') if b is held else b for b in bags]
        with self.assertRaisesRegex(ValueError,'wrong partition'):
            validate_prepared_split(self.root,self.output,bags=altered)
        injected=[dict(pairs[0],group_id=held['group_id'],video_sha256=held['sha256'])]
        with self.assertRaisesRegex(ValueError,'Narration caption'):
            validate_prepared_split(self.root,self.output,pairs=injected)

    def test_old_feature_cache_is_rejected_for_new_dataset(self):
        old=self.root/'runs/narration_training/original'
        self.prepare(old,original_split=True);self.prepare()
        features=self.root/'runs/narration_training/old_features';features.mkdir()
        (features/'extraction.json').write_text(json.dumps(dict(clips_sha256=digest(old/'clips.jsonl'),pairs_sha256=digest(old/'pairs.jsonl'))))
        with self.assertRaisesRegex(ValueError,'Features and audited data disagree'):
            FeatureStore(self.output,features)

    def test_nested_fold_candidates_remain_inside_38_train_groups(self):
        from types import SimpleNamespace
        self.prepare();bags=read_jsonl(self.output/'bags.jsonl')
        store=SimpleNamespace(bags=bags)
        groups=[b['group_id'] for b in bags if b['split']=='train'];diagnostic={b['group_id'] for b in bags if b['split']=='val'}
        seen=[]
        for fit,held in folds(store,groups,4,7301):
            self.assertFalse((set(fit)|set(held))&diagnostic)
            self.assertFalse(set(fit)&set(held));seen.extend(held)
            for inner_fit,inner_held in folds(store,fit,3,7302):
                self.assertFalse((set(inner_fit)|set(inner_held))&(diagnostic|set(held)))
        self.assertEqual(sorted(seen),sorted(groups))
        self.assertEqual(len(groups),38)


if __name__=='__main__':unittest.main()
