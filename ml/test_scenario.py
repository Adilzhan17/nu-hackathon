import unittest
from scenario import valid_events,hypotheses,ledger,parse

def event(action,start=0,end=1,**kw):
    return dict(id=action+str(start),start_sec=start,end_sec=end,action=action,actor='person_1',object='safe',recipient='unknown',visibility='clear',description='visible',evidence_frames=['f000'],status='model_proposed',**kw)

class ScenarioTests(unittest.TestCase):
    def test_missing_observation_is_not_a_violation(self):
        self.assertEqual(hypotheses([event('employee_leaves')]),[])
        self.assertEqual(hypotheses([event('cash_received'),event('employee_leaves',2,3)]),[])

    def test_requires_positive_state_evidence_after_departure(self):
        before=[event('safe_visibly_open'),event('employee_leaves',2,3),event('employee_absent',4,5)]
        self.assertEqual(hypotheses(before),[])
        result=hypotheses(before+[event('safe_visibly_open',4,5)])
        self.assertEqual(len(result),1);self.assertIsNone(result[0]['legal_violation'])

    def test_unclear_state_is_not_promoted(self):
        state=event('safe_visibly_open',4,5);state['visibility']='unclear'
        self.assertEqual(hypotheses([event('employee_leaves',2,3),event('employee_absent',4,5),state]),[])

    def test_rejects_fabricated_references_and_outside_focus(self):
        frames=[dict(id='f000',time_sec=.5)];e=event('safe_opened')
        good,errors=valid_events({'events':[e]},(0,4),frames);self.assertEqual(len(good),1);self.assertFalse(good[0]['ground_truth'])
        e['evidence_frames']=['fake'];self.assertEqual(valid_events({'events':[e]},(0,4),frames)[0],[])
        e['evidence_frames']=['f000'];e['end_sec']=5;self.assertEqual(valid_events({'events':[e]},(0,4),frames)[0],[])

    def test_memory_preserves_uncertainty_and_source(self):
        e=event('cash_received');e['object']='cash';e['visibility']='partial'
        m=ledger([e]);self.assertEqual(m['last_observed']['cash']['visibility'],'partial')
        self.assertEqual(m['transfer_history'][0]['id'],e['id']);self.assertEqual(m['last_observed']['cash']['status'],'model_proposed')

    def test_json_parser_does_not_invent_fields(self):
        self.assertEqual(parse('```json\n{"events": []}\n```'),{'events':[]})
        with self.assertRaises(ValueError):parse('No visual evidence')

if __name__=='__main__':unittest.main()
