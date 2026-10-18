import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from .dataset import GemmaAdaptationDataset, make_loader
from .prepare import checked_target, generic_prompt, weak_exclusion, WEAK_EXTRA_CODES


class PreparationTests(unittest.TestCase):
    def test_weak_source_corrected_by_human_is_excluded(self):
        pair = dict(adaptation_partition='fit', split='train', action_candidates=['cash_counting'],
                    eligible_for_weak_text_alignment=True, group_id='g', start_sec=0., end_sec=3., observation_id='o')
        human = [dict(group_id='g', start_sec=1., end_sec=2., observed_action='cash_counter_loading',
                      task_id='h', provenance=dict(source_kind='user_reference', source_id='o'))]
        self.assertTrue(weak_exclusion(pair, human).startswith('human_corrected'))
        human[0]['provenance']['source_id'] = 'different_source'
        self.assertIsNone(weak_exclusion(pair, human))

    def test_unconfirmed_and_long_targets_not_silently_promoted_or_truncated(self):
        pair = dict(adaptation_partition='fit', split='train', action_candidates=['cash_counting'],
                    eligible_for_weak_text_alignment=True, ambiguous_object_or_action=True)
        self.assertEqual(weak_exclusion(pair, []), 'ambiguous_object_or_action')
        with self.assertRaisesRegex(ValueError, 'not_truncated'):
            checked_target(dict(description=' '.join(['word'] * 101)))
        for kind, n in [('action', 2), ('weak', 2), ('video', 16)]:
            prompt = generic_prompt(kind, list(range(n)))
            self.assertLessEqual(len(prompt.split()), 180)
            self.assertNotIn('Х-10', prompt)
        weak_prompt = generic_prompt('weak', [0., 1.])
        for code in WEAK_EXTRA_CODES:
            self.assertIn(code, weak_prompt)

    def _dataset(self, root):
        folder = root / 'prepared'
        (folder / 'frames').mkdir(parents=True)
        frames = []
        for i in range(2):
            path = folder / 'frames' / f'f{i}.jpg'
            Image.new('RGB', (32, 24), (i * 60, 0, 0)).save(path)
            frames.append(dict(path=path.relative_to(root).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               width=32, height=24, timestamp=float(i), actual_sec=float(i)))
        protocol = dict(group_membership={'g': 'fit'}, source_identity={'g': {'video_sha256': 'a' * 64}})
        (root / 'frozen_group_protocol.json').write_text(json.dumps(protocol))
        row = dict(sample_id='s', kind='human_action', group_id='g', partition='fit', split='train',
                   video_sha256='a' * 64, frames=frames, frame_paths=[f['path'] for f in frames],
                   prompt='Generic timestamps:0,1.', target='{"observed_action":"safe_opening"}', loss_weight=1.)
        manifest = folder / 'training_samples.jsonl'
        manifest.write_text(json.dumps(row) + '\n')
        return manifest, row

    def test_loader_yields_pil_and_detects_frame_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, row = self._dataset(root)
            item = next(iter(make_loader(manifest)))
            self.assertEqual(item['frames'][0].size, (32, 24))
            self.assertEqual(item['metadata']['sample_id'], 's')
            self.assertEqual(item['loss_weight'], 1.)
            (root / row['frame_paths'][0]).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                GemmaAdaptationDataset(manifest)[0]

    def test_loader_refuses_non_fit_or_escape_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, row = self._dataset(root)
            row['partition'] = 'inner_holdout'
            manifest.write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'partition'):
                GemmaAdaptationDataset(manifest)
            row['partition'] = 'fit'
            row['frames'][0]['path'] = '../outside.jpg'
            row['frame_paths'][0] = '../outside.jpg'
            manifest.write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'escapes'):
                GemmaAdaptationDataset(manifest)


if __name__ == '__main__':
    unittest.main()
