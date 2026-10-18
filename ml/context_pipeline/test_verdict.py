import unittest
from .rules import whole_video_verdict


class VerdictTests(unittest.TestCase):
    def test_supported_episode_survives_other_unknown_rules(self):
        self.assertEqual(whole_video_verdict([
            {'verdict': 'insufficient_data', 'evidence': []},
            {'verdict': 'violation_supported', 'evidence': [{'observation_id': 'test'}]},
        ]), 'violation_supported')

    def test_sparse_normal_findings_do_not_clear_whole_video(self):
        for rules in ([], [{'verdict': 'no_violation_observed', 'evidence': [{'observation_id': 'test'}]}],
                      [{'verdict': 'violation_supported', 'evidence': []}]):
            self.assertEqual(whole_video_verdict(rules), 'insufficient_data')
