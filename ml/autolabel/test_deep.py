import copy
import unittest
from .deep import coverage_windows,convert_event,parse_deep


class DetailedTests(unittest.TestCase):
    def test_full_coverage_no_gaps_or_short_tail(self):
        for duration in [.5,6,6.01,19.6,71.6,142.533]:
            cs=coverage_windows(duration)
            self.assertEqual(cs[0]['start_sec'],0)
            self.assertAlmostEqual(cs[-1]['end_sec'],duration,places=5)
            self.assertTrue(all(0<=c['start_sec']<c['end_sec']<=duration for c in cs))
            self.assertTrue(all(b['start_sec']<=a['end_sec'] for a,b in zip(cs,cs[1:])))
            if duration>6:self.assertTrue(all(c['end_sec']-c['start_sec']>=5.99999 for c in cs))

    def fixture(self):
        c=dict(video='v',group_id='g',candidate_id='c',frames=[dict(timestamp=t) for t in [1.,1.25,1.5,1.75]])
        plan=dict(registry={'v':dict(group_id='g',split='train')},durations={'g':2.},model_lock=dict(model_id='m',model_revision='a'*40))
        e=dict(observed_action='object_handling',objects=['unidentified_item'],start_frame=0,end_frame=3,
               evidence_frames=[0,2,3],confidence=.6,uncertainty_reason='item unclear',evidence_description='visible movement')
        return e,c,plan

    def test_indices_convert_only_to_actual_pts(self):
        e,c,plan=self.fixture();p=convert_event(e,c,plan,'p')
        self.assertEqual(p['evidence_timestamps'],[1.,1.5,1.75])
        self.assertEqual(p['start_sec'],1.)
        self.assertEqual(p['review_status'],'proposed')
        self.assertIn('окно',p['uncertainty_reason'])

    def test_bad_indices_and_validation_prohibited(self):
        e,c,plan=self.fixture()
        for extra in [dict(end_frame=4),dict(evidence_frames=[0,4]),dict(evidence_frames=[0,0]),dict(start_frame=True),dict(evidence_frames=[2,0])]:
            with self.assertRaises(ValueError):convert_event(dict(e,**extra),c,plan,'p')
        plan['registry']['v']['split']='val'
        with self.assertRaises(ValueError):convert_event(e,c,plan,'p')

    def test_nonfinite_json_rejected(self):
        with self.assertRaises(ValueError):parse_deep('{"sequence_summary":"x","events":[],"confidence":NaN}')


if __name__=='__main__':unittest.main()
