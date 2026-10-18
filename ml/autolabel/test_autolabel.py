import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from .dataset import registry, digest
from .review import reconcile
from .sampling import candidates
from .schema import validate_proposal
from .vlm import parse_events


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.reg = {'videos/a.mov': dict(split='train', group_id='g', aliases=['videos/a.mov']),
                    'videos/v.mov': dict(split='val', group_id='v', aliases=['videos/v.mov'])}
        self.p = dict(video='videos/a.mov', group_id='g', start_sec=1., end_sec=2.,
                      observed_action='safe_opening', objects=['safe'], evidence_timestamps=[1., 2.],
                      confidence=.9, uncertainty_reason='', model_id='Qwen/test', model_revision='a'*40,
                      review_status='proposed', proposal_id='p')

    def test_valid_sequence(self):
        validate_proposal(self.p, self.reg, 3., [1., 1.5, 2.])

    def test_reject_leakage_bad_time_and_fake_boxes(self):
        bad = [dict(video='videos/v.mov', group_id='v'), dict(group_id='wrong'), dict(end_sec=4.),
               dict(start_sec=-1.), dict(start_sec=2.), dict(confidence=float('nan')),
               dict(confidence=True), dict(review_status='accepted'), dict(bbox=[1,2,3,4]),
               dict(evidence_timestamps=[1.]), dict(evidence_timestamps=[1., 1.]),
               dict(evidence_timestamps=[1., 1.75]), dict(observed_action='violation'),
               dict(objects=['invented_object']), dict(model_revision='main')]
        for fields in bad:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate_proposal(dict(self.p, **fields), self.reg, 3., [1., 2.])

    def test_strict_json_not_salvaged(self):
        for response in ['some prose {"events": []}', '{"events": [], "verdict":"normal"}',
                         '{"events":[{"bbox":[0,0,1,1]}]}']:
            with self.assertRaises(ValueError):
                parse_events(response)
        self.assertEqual(parse_events('```json\n{"events": []}\n```'), [])
        self.assertEqual(parse_events('[]'), [])

    def test_normal_is_not_conflict_with_action(self):
        conflict, queue = reconcile([self.p], [], {'g': []}, {'g': [{'source_verdict':'без нарушения'}]}, {'g':3.}, self.reg)
        self.assertEqual(conflict, [])
        self.assertEqual(queue, [])
        self.assertEqual(self.p['review_status'], 'proposed')

    def test_user_priority_opposite_phase_and_quarantine(self):
        observations = {'g': [dict(video='videos/a.mov', status='transcribed', authority='user_confirmed_source',
                                  start_sec=1., end_sec=2., action='safe_door_closing', observation_id='o1'),
                              dict(video='videos/a.mov', status='quarantined', authority='quarantined',
                                   start_sec=2., end_sec=2.5, action='cash_received', observation_id='o2', reason='mapping unclear')]}
        before = copy.deepcopy(observations)
        conflicts, queue = reconcile([self.p], [], observations, {'g': []}, {'g':3.}, self.reg)
        self.assertEqual({c['kind'] for c in conflicts}, {'possible_temporal_disagreement', 'quarantined_source'})
        self.assertEqual(observations, before)
        self.assertEqual(len(queue), 1)
        self.assertEqual(self.p['observed_action'], 'safe_opening')
        self.assertEqual(self.p['review_status'], 'proposed')
        self.assertEqual(queue[0]['context_end_sec'], 3.)

    def test_object_disagreement_preserves_user_account(self):
        p = dict(self.p, observed_action='document_handling', objects=['document'])
        o = dict(video='videos/a.mov', status='transcribed', authority='user_confirmed_source',
                 start_sec=1.5, end_sec=2.5, action='cash_unpacking', observation_id='o')
        conflicts, queue = reconcile([p], [], {'g': [o]}, {'g': []}, {'g':3.}, self.reg)
        self.assertEqual(conflicts[0]['kind'], 'possible_object_disagreement')
        self.assertIn('possible_object_disagreement', queue[0]['reasons'])
        self.assertEqual(o['action'], 'cash_unpacking')

    def test_empty_response_is_review_not_negative(self):
        c = dict(video='videos/a.mov', group_id='g', start_sec=1., end_sec=2., inference_status='empty')
        conflicts, queue = reconcile([], [c], {'g': []}, {'g': []}, {'g':3.}, self.reg)
        self.assertFalse(conflicts)
        self.assertEqual(queue[0]['reasons'], ['model_abstained_not_a_negative_label'])

    def test_candidates_are_bounded_and_include_quiet_audit(self):
        scan = [dict(timestamp=float(t), change=.8 if t in (5, 13) else .01) for t in range(20)]
        cs = candidates(scan, 20., 3, 4.)
        self.assertEqual(len(cs), 3)
        self.assertEqual(sum(c['reason']=='low_motion_audit' for c in cs), 1)
        self.assertTrue(all(0 <= c['start_sec'] < c['end_sec'] <= 20. for c in cs))


class RegistryTests(unittest.TestCase):
    def fixture(self, root):
        for folder in ['data/raw/open/videos', 'data/training', 'data/review', 'data/transcripts']:
            (root/folder).mkdir(parents=True)
        (root/'data/raw/open/videos/a.mov').write_bytes(b'source')
        (root/'data/raw/open/videos/duplicate.mov').write_bytes(b'source')
        sha = digest(root/'data/raw/open/videos/a.mov')
        (root/'data/training/manifest.csv').write_text(f'video,group_id,sha256,split\nvideos/a.mov,g,{sha},train\n')
        (root/'data/review/event_review.csv').write_text('video,review_status\nvideos/a.mov,pending\n')
        (root/'data/transcripts/observations.jsonl').write_text('')
        (root/'data/transcripts/sources.json').write_text('{"sections":[]}')
        return sha

    def test_exact_duplicates_share_group_and_unknown_versions_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            reg, _, _ = registry(root)
            self.assertEqual(reg['videos/a.mov']['group_id'], reg['videos/duplicate.mov']['group_id'])
            (root/'data/raw/open/videos/reencode.mp4').write_bytes(b'reencoded')
            with self.assertRaisesRegex(ValueError, 'Unknown source'):
                registry(root)
            links = root/'lineage.json'
            links.write_text('{"videos/reencode.mp4":"videos/a.mov"}')
            reg, _, _ = registry(root, links)
            self.assertEqual(reg['videos/reencode.mp4']['group_id'], 'g')
            self.assertEqual(reg['videos/reencode.mp4']['split'], 'train')

    def test_split_leak_rejected_before_sampling(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sha = self.fixture(root)
            with (root/'data/training/manifest.csv').open('a') as f:
                f.write(f'videos/duplicate.mov,g,{sha},val\n')
            with self.assertRaises(ValueError):
                registry(root)

    def test_lineage_cannot_move_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            p = root/'data/raw/open/videos/v.mov'; p.write_bytes(b'validation')
            with (root/'data/training/manifest.csv').open('a') as f:
                f.write(f'videos/v.mov,v,{digest(p)},val\n')
            links = root/'lineage.json'; links.write_text('{"videos/v.mov":"videos/a.mov"}')
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                registry(root, links)


class DecodeTests(unittest.TestCase):
    def test_actual_pts_dense_sampling_and_no_audio(self):
        import av
        import numpy as np
        from .sampling import frames, metadata
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'tiny.mp4'
            with av.open(str(path), 'w') as out:
                s = out.add_stream('libx264', rate=12); s.width=64; s.height=64; s.pix_fmt='yuv420p'
                for i in range(36):
                    pixels = np.zeros((64,64,3), dtype=np.uint8)
                    pixels[:, i: i+12] = 200
                    for packet in s.encode(av.VideoFrame.from_ndarray(pixels, format='rgb24')):
                        out.mux(packet)
                for packet in s.encode(): out.mux(packet)
            coarse = list(frames(path, 0, 3, 1.5))
            dense = list(frames(path, 1, 2.5, 4))
            self.assertGreater(len(dense), len(coarse))
            self.assertTrue(all(1 <= t <= 2.5 for t, _ in dense))
            self.assertTrue(all(abs(t*12-round(t*12)) < .001 for t, _ in dense))
            self.assertAlmostEqual(metadata(path)['duration_sec'], 3., places=2)
            fractional = list(frames(path, .666667, 2.666667, 4.))
            self.assertLessEqual(len(fractional), 9)
            self.assertTrue(all(b[0]-a[0] > .2 for a,b in zip(fractional, fractional[1:])))


if __name__ == '__main__':
    unittest.main()
