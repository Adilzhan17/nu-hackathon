import json
from pathlib import Path
import unittest
from .context import memory,rule_assessments,validate_observations,RULES,parse_answer
from .source import SPECS,RAW_MESSAGE

def observation(kind='state',t=1,**kwargs):
    base=dict(observation_id=f'observation_{t}',kind=kind,action=None,property='safe_door',value='open',
      actor='person_1',role='employee',object='safe',track_id='safe_1',recipient='unknown',visibility='clear',
      start_sec=t,end_sec=t+.2,last_observed_sec=t+.2,description='Visible state',uncertainty='',
      source='local_qwen_visual_proposal',authority='visual_evidence',human_confirmed=False,
      evidence=[dict(id='f000',timestamp=t,path='test.jpg')],stage='focus_00',identity_link_status='model_proposed_not_verified')
    base.update(kwargs);return base

class OrganizerTests(unittest.TestCase):
    def test_source_count_and_exceptions(self):
        self.assertEqual(len(RAW_MESSAGE.splitlines()),22);self.assertEqual(len(SPECS),22)
        self.assertIsNone(SPECS[13]['presumed_violation']);self.assertEqual(SPECS[43]['statement_mode'],'scoped_non_violation_exception')
        self.assertIsNone(SPECS[43]['presumed_violation']);self.assertEqual(SPECS[44]['organizer_certainty'],'uncertain')
    def test_normative_statements_not_positive_targets(self):
        self.assertIsNone(SPECS[2]['presumed_violation']);self.assertIsNone(SPECS[10]['presumed_violation'])
    def test_generic_rules_do_not_use_val_explanations(self):
        sources=[s for rule in RULES.values() for s in rule['sources']]
        self.assertFalse(set(sources)&{f'organizer_train_{n}' for n in [4,20,24,43,44]})
        self.assertIn('preexisting_ml/autolabel/rules.py:banknote_check_not_observed',sources)
    def test_future_not_in_memory(self):
        m=memory([observation(t=1),observation(t=9)],5)
        self.assertEqual(len(m['last_known_facts']),1);self.assertEqual(m['last_known_facts'][0]['last_observed_sec'],1.2)
    def test_last_known_not_current(self):
        m=memory([observation(t=1,visibility='partial',uncertainty='occluded')],5)
        f=m['last_known_facts'][0]
        self.assertEqual(f['current_state'],'unknown_until_reobserved');self.assertEqual(f['visibility'],'partial')
        self.assertEqual(f['uncertainty'],'occluded');self.assertFalse(f['human_confirmed'])
    def test_ownership_remains_unknown(self):
        m=memory([observation(kind='event',action='cash_received',object='cash')],5)
        self.assertEqual(m['transfer_history'][0]['ownership'],'unknown')
    def test_absence_never_from_missing_sampled_event(self):
        rows=rule_assessments([observation(kind='event',action='cash_counter_used')])
        r=next(r for r in rows if r['rule_id']=='authenticity_after_counting')
        self.assertEqual(r['verdict'],'insufficient_data');self.assertTrue(r['missing_evidence'])
    def test_recount_is_not_an_error_rule(self):
        rows=rule_assessments([observation(kind='event',action='cash_counted_by_hand',t=1),observation(kind='event',action='cash_counted_by_hand',t=5)])
        self.assertFalse(any(r['verdict']=='violation_supported' for r in rows))
    def test_departure_requires_before_and_after(self):
        obs=[observation(t=1),observation(kind='event',action='employee_leaves',t=3),observation(kind='event',action='employee_absent',t=4)]
        r={x['rule_id']:x for x in rule_assessments(obs)}['safe_on_departure']
        self.assertEqual(r['verdict'],'insufficient_data')
        obs.append(observation(t=5))
        r={x['rule_id']:x for x in rule_assessments(obs)}['safe_on_departure']
        self.assertEqual(r['verdict'],'violation_supported');self.assertFalse(r['human_confirmed']);self.assertIsNone(r['legal_violation_target'])
    def test_intervening_closure_blocks_departure_claim(self):
        obs=[observation(t=1),observation(kind='event',action='employee_leaves',t=3),observation(kind='event',action='employee_absent',t=4),observation(t=5),observation(kind='event',action='safe_closing',t=2)]
        self.assertEqual({x['rule_id']:x for x in rule_assessments(obs)}['safe_on_departure']['verdict'],'insufficient_data')
    def test_dark_screen_does_not_prove_shutdown(self):
        obs=[observation(property='computer_screen',value='dark'),observation(kind='event',action='employee_leaves',t=4)]
        self.assertEqual({x['rule_id']:x for x in rule_assessments(obs)}['equipment_on_departure']['verdict'],'insufficient_data')
    def test_frame_reference_validation(self):
        frames=[dict(id='f000',timestamp=1),dict(id='f001',timestamp=2)]
        item=dict(action='cash_received',visibility='clear',evidence_frames=['f000','invented'])
        accepted,errors=validate_observations({'events':[item]},frames,'group','focus',(0,3))
        self.assertFalse(accepted);self.assertTrue(errors)
        item['evidence_frames']=['f000'];accepted,_=validate_observations({'events':[item]},frames,'group','focus',(0,3));self.assertFalse(accepted)
        item['evidence_frames']=['f000','f001'];accepted,_=validate_observations({'events':[item]},frames,'group','focus',(0,3));self.assertEqual(len(accepted),1)
    def test_truncated_json_is_not_repaired_into_evidence(self):
        with self.assertRaises(ValueError):parse_answer('{"events":[{"action":"cash_received"')
    def test_rule_output_shape(self):
        for r in rule_assessments([]):
            self.assertTrue(set(['rule_id','verdict','triggering_event','evidence','missing_evidence','explanation'])<=set(r))
            self.assertFalse(r['whole_video_normal'])

if __name__=='__main__':unittest.main()
