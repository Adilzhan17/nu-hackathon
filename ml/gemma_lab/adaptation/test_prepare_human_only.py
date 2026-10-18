import json
import hashlib
import unittest
from .prepare_human_only import select
from .test_train import TrainingSafetyTests


class HumanOnlyTests(unittest.TestCase):
    def test_only_confirmed_specific_fit_actions_survive(self):
        rows, protocol = TrainingSafetyTests().rows_protocol()
        rows[0].update(frames=[dict(timestamp=1), dict(timestamp=2)],
                       target=json.dumps(dict(observed_action='cash_counting', description='Count', review_status='proposed')))
        weak = dict(rows[0], sample_id='weak', kind='weak_caption', trust_level='weak_unconfirmed', loss_weight=.2)
        vague = dict(rows[0], sample_id='vague', target=json.dumps(dict(observed_action='other_observed_action')))
        selected, excluded = select(rows + [weak, vague], protocol)
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(excluded), 2)
        self.assertIn('principal visible action', selected[0]['prompt'])
        self.assertEqual(selected[0]['prompt_sha256'], hashlib.sha256(selected[0]['prompt'].encode()).hexdigest())
        with self.assertRaisesRegex(ValueError, 'leakage'):
            select([dict(rows[0], group_id='g2')], protocol)


if __name__ == '__main__':
    unittest.main()
