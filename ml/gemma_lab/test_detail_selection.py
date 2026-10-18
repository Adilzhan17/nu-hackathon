import unittest
from .detail_selection import select_detail_centers


class DetailSelectionTests(unittest.TestCase):
    def test_late_departure_wins_over_early_routine_events(self):
        observations = [dict(action=action, start_sec=start, end_sec=end, observation_id=action)
                        for action, start, end in [('person_enters', 0, 2), ('cash_received', 5, 7),
                                                   ('employee_leaves', 60, 64)]]
        result = select_detail_centers(observations, [])
        self.assertEqual([r['timestamp'] for r in result], [62, 6])
        self.assertEqual(result[0]['observation_id'], 'employee_leaves')

    def test_clustered_observations_do_not_consume_both_checks(self):
        observations = [dict(action=action, start_sec=time, end_sec=time+1)
                        for action, time in [('employee_leaves', 20), ('employee_absent', 21), ('phone_used', 40)]]
        self.assertEqual([r['timestamp'] for r in select_detail_centers(observations, [])], [20.5, 40.5])

    def test_unknown_or_invalid_candidates_use_blind_midpoint(self):
        frames = [dict(timestamp=t) for t in [0, 2, 4]]
        observations = [dict(action='employee_leaves', start_sec=float('nan'), end_sec=5)]
        result = select_detail_centers(observations, frames)
        self.assertEqual(result[0]['timestamp'], 2)
        self.assertIsNone(result[0]['observation_id'])


if __name__ == '__main__':
    unittest.main()
