import json
import unittest

from .audit_data import replay_journal
from .group_protocol import build_protocol


class AuditSafetyTests(unittest.TestCase):
    def test_inflight_journal_tail_is_excluded_and_superseded_decision_is_latest(self):
        rows = [dict(task_id='a', revision=1, version=1, decision='accept'),
                dict(task_id='a', revision=2, version=2, decision='needs_context')]
        complete = ''.join(json.dumps(row) + '\n' for row in rows).encode()
        state, meta = replay_journal(complete + b'{"in_progress":')
        self.assertEqual(state['a']['decision'], 'needs_context')
        self.assertEqual(meta['revision'], 2)
        self.assertEqual(meta['complete_bytes'], len(complete))
        self.assertGreater(meta['incomplete_tail_bytes'], 0)

    def test_broken_journal_is_not_silently_accepted(self):
        row = dict(task_id='a', revision=2, version=1)
        with self.assertRaises(ValueError):
            replay_journal((json.dumps(row) + '\n').encode())

    def test_group_holdout_is_deterministic_balanced_and_excludes_all_aliases(self):
        targets = [dict(group_id=f'g{i:02d}', video=f'v{i}', video_sha256=str(i) * 64,
                        split='train' if i < 32 else 'val', label=i % 2,
                        aliases=[f'v{i}', f'duplicate_{i}']) for i in range(40)]
        human = [dict(group_id=f'g{i:02d}') for i in range(12)]
        protocol = build_protocol(targets, human)
        shuffled = build_protocol(list(reversed(targets)), list(reversed(human)))
        self.assertEqual(protocol['group_membership'], shuffled['group_membership'])
        self.assertEqual(len(protocol['fit_groups']), 24)
        self.assertEqual(len(protocol['inner_holdout_groups']), 8)
        self.assertEqual(protocol['inner_holdout_label_counts'], {'0': 4, '1': 4})
        self.assertGreaterEqual(len(protocol['inner_holdout_human_groups']), 2)
        self.assertFalse(set(protocol['fit_groups']) & set(protocol['inner_holdout_groups']))
        self.assertFalse(set(protocol['fit_groups']) & set(protocol['diagnostic_val_groups']))
        # Membership is keyed by source group, never filename/derived frame identity.
        for row in targets:
            self.assertEqual(len({protocol['group_membership'][row['group_id']] for _ in row['aliases']}), 1)

    def test_validation_human_labels_cannot_enter_training_protocol(self):
        targets = [dict(group_id=f'g{i}', video=f'v{i}', video_sha256=str(i) * 64,
                        split='train' if i < 32 else 'val', label=i % 2) for i in range(40)]
        with self.assertRaises(ValueError):
            build_protocol(targets, [dict(group_id='g32')])


if __name__ == '__main__':
    unittest.main()
