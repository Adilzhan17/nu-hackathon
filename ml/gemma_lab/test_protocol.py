import unittest
from .protocol import windows,coverage,inspect_answer,assess,memory_before,prompt_memory,parse_result


class ProtocolTests(unittest.TestCase):
    def test_echoed_memory_and_truncated_answers_are_not_counted_as_observations(self):
        with self.assertRaises(ValueError):parse_result('{"as_of_sec":5.5,"recent_actions":[]}')
        with self.assertRaises(ValueError):parse_result('{"events":[')
        with self.assertRaises(ValueError):parse_result('{"events":[],"states":null,"rule_checks":[]}')
        self.assertEqual(parse_result('{"events":[],"states":[],"rule_checks":[]}')['events'],[])

    def test_every_half_second_frame_is_in_an_ordered_window(self):
        frames=[dict(id=f'f{i:05d}',timestamp=i*.5) for i in range(120)]
        batches=windows(frames)
        self.assertEqual({f['id'] for b in batches for f in b},{f['id'] for f in frames})
        self.assertTrue(all(len(b)<=12 for b in batches))
        partial=coverage(frames,[f['id'] for f in batches[0]])
        self.assertFalse(partial['all_sampled_frames_inspected'])
        self.assertEqual(partial['inspected_sampled_frames'],12)

    def test_fabricated_references_and_single_frame_actions_are_rejected(self):
        refs=[dict(id='f00000',timestamp=0),dict(id='f00001',timestamp=.5)]
        obs,claims,errors=inspect_answer(dict(events=[dict(action='cash_received',evidence_frames=['f00000'])],
            rule_checks=[dict(rule_id='safe_on_departure',verdict='violation_supported',evidence_frames=['fake'])]),refs,'g','s')
        self.assertFalse(obs);self.assertFalse(claims);self.assertEqual(len(errors),2)

    def test_missing_action_is_not_promoted_to_violation(self):
        refs=[dict(id='f00000',timestamp=0),dict(id='f00001',timestamp=.5)]
        obs,claims,_=inspect_answer(dict(rule_checks=[dict(rule_id='authenticity_after_counting',
            verdict='violation_supported',evidence_frames=['f00000','f00001'],explanation='No checking seen')]),refs,'g','s')
        result=next(r for r in assess(obs,claims) if r['rule_id']=='authenticity_after_counting')
        self.assertEqual(result['verdict'],'insufficient_data')
        self.assertEqual(result['model_claims'][0]['model_verdict'],'violation_supported')
        self.assertFalse(result['human_confirmed'])

    def test_supported_scenario_carries_verified_regulatory_metadata(self):
        refs=[dict(id=f'f{i:05d}',timestamp=i) for i in range(4)]
        answer=dict(events=[
            dict(action='employee_leaves',actor='person_1',role='employee',object='unknown',
                 track_id='employee_1',visibility='clear',description='Сотрудник вышел',
                 uncertainty='',evidence_frames=['f00001','f00002']),
            dict(action='employee_absent',actor='person_1',role='employee',object='unknown',
                 track_id='employee_1',visibility='clear',description='Сотрудник отсутствует',
                 uncertainty='',evidence_frames=['f00002','f00003']),
        ],states=[
            dict(property='key_location',value='safe_top',actor='person_1',role='employee',
                 object='keys',track_id='keys_1',visibility='clear',description='Ключи на сейфе',
                 uncertainty='',evidence_frames=['f00000']),
            dict(property='key_location',value='safe_top',actor='person_1',role='employee',
                 object='keys',track_id='keys_1',visibility='clear',description='Ключи остались',
                 uncertainty='',evidence_frames=['f00003']),
        ],rule_checks=[])
        observations,claims,errors=inspect_answer(answer,refs,'g','s')
        self.assertFalse(errors)
        result=next(r for r in assess(observations,claims) if r['rule_id']=='valuables_on_departure')
        self.assertEqual(result['verdict'],'violation_supported')
        self.assertEqual(result['regulatory_source']['section'],'пункт 80')
        self.assertTrue(result['regulatory_source']['url'].startswith('https://adilet.zan.kz/'))

    def test_nullable_rule_metadata_does_not_discard_observations(self):
        refs=[dict(id='f00000',timestamp=0),dict(id='f00001',timestamp=.5)]
        obs,claims,errors=inspect_answer(dict(events=[dict(action='phone_used',visibility='partial',
            uncertainty='Hand occlusion',evidence_frames=['f00000','f00001'])],
            rule_checks=[dict(rule_id='phone_before_counting',verdict='insufficient_data',
                             evidence_frames=[],missing_evidence=None)]),refs,'g','s')
        self.assertEqual(len(obs),1);self.assertEqual(len(claims),1);self.assertFalse(errors)

    def test_overlap_memory_keeps_only_past_evidence_and_uncertainty(self):
        refs=[dict(id=f'f{i:05d}',timestamp=i*.5) for i in range(12)]
        obs,_,_=inspect_answer(dict(events=[dict(action='phone_used',visibility='partial',
            uncertainty='Hand occlusion',evidence_frames=[f['id'] for f in refs])]),refs,'g','s')
        value=memory_before(obs,5)
        action=value['recent_actions'][0]
        self.assertLess(action['last_observed_sec'],5)
        self.assertEqual(action['uncertainty'],'Hand occlusion')
        self.assertEqual(action['current_state'],'unknown_until_reobserved')
        compact=prompt_memory(value)
        self.assertEqual(len(value['recent_actions'][0]['evidence_timestamps']),10)
        self.assertLessEqual(len(compact['recent_actions'][0]['evidence_timestamps']),2)


if __name__=='__main__':unittest.main()
