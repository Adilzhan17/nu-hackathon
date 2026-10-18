import unittest
from .finalize import guarded_assessments,coverage
from .test_organizer import observation

class ContinuityTests(unittest.TestCase):
    def verdict(self,events,key):return {r['rule_id']:r for r in guarded_assessments(events)}[key]
    def test_authenticity_cannot_link_different_cash_tracks(self):
        a=observation(kind='event',action='cash_counter_used',object='cash',track_id='cash_1',t=1)
        b=observation(kind='event',action='banknote_detector_used',object='cash',track_id='cash_2',t=5)
        self.assertEqual(self.verdict([a,b],'authenticity_after_counting')['verdict'],'insufficient_data')
    def test_clear_but_uncertain_object_stays_uncertain(self):
        a=observation(kind='event',action='phone_put_away',object='phone',t=1,uncertainty='small object may be another item')
        b=observation(kind='event',action='cash_counter_used',object='cash',t=5)
        self.assertEqual(self.verdict([a,b],'phone_before_counting')['verdict'],'insufficient_data')
    def test_customer_change_blocks_cross_operation_normal(self):
        a=observation(kind='event',action='phone_put_away',object='phone',t=1)
        c=observation(kind='event',action='person_enters',actor='person_2',role='customer',t=3)
        b=observation(kind='event',action='cash_counter_used',object='cash',t=5)
        self.assertEqual(self.verdict([a,c,b],'phone_before_counting')['verdict'],'insufficient_data')
    def test_unknown_customer_blocks_id_check_for_later_cash(self):
        a=observation(kind='event',action='document_checked',object='document',t=1)
        b=observation(kind='event',action='cash_received',object='cash',t=5)
        self.assertEqual(self.verdict([a,b],'identity_before_cash')['verdict'],'insufficient_data')
    def test_scoped_positive_does_not_make_whole_video_normal(self):
        a=observation(kind='event',action='cash_counter_used',object='cash',track_id='cash_1',t=1)
        b=observation(kind='event',action='banknote_detector_used',object='cash',track_id='cash_1',t=5)
        r=self.verdict([a,b],'authenticity_after_counting')
        self.assertEqual(r['verdict'],'no_violation_observed');self.assertFalse(r['whole_video_normal']);self.assertFalse(r['human_confirmed'])

if __name__=='__main__':unittest.main()
