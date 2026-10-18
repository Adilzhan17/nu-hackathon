import json
from pathlib import Path
import tempfile
import unittest
from .evaluation import family,partition,compare

class EvaluationTests(unittest.TestCase):
    def test_loading_and_locking_are_separate_actions(self):
        self.assertNotEqual(family('cash_counter_loading'),family('cash_counter_used'))
        self.assertNotEqual(family('safe_locking'),family('safe_closing'))
        self.assertNotEqual(family('leave_workstation'),family('employee_leaves'))
    def test_supported_human_action_not_discarded_for_missing_model_class(self):
        actions,states,other=partition([dict(observed_action='cash_counter_loading'),dict(observed_action='cash_visible'),dict(observed_action='other_observed_action')])
        self.assertEqual(len(actions),1);self.assertFalse(actions[0]['model_vocabulary_covered'])
        self.assertEqual(len(states),1);self.assertEqual(len(other),1)
    def test_ontology_gap_stays_in_denominator_state_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            blind=Path(tmp);folder=blind/'g';folder.mkdir()
            (folder/'complete.json').write_text(json.dumps(dict(ablation=True)))
            (folder/'focus_plan.json').write_text(json.dumps([dict(context_start_sec=0,context_end_sec=5)]))
            observation=dict(kind='event',stage='focus_00',action='cash_counter_used',observation_id='o',start_sec=1,end_sec=2)
            for f in ['visual_observations.jsonl','independent_observations.jsonl']:(folder/f).write_text(json.dumps(observation)+'\n')
            human=[dict(task_id=str(i),group_id='g',observed_action=a,start_sec=1,end_sec=2) for i,a in enumerate(['cash_counter_loading','cash_counting','cash_visible'])]
            r=compare(blind,[dict(group_id='g',video='video.mov')],human)
            self.assertEqual(r['totals']['with_history'],dict(recovered=1,references=2,references_outside_model_vocabulary=1))
            self.assertEqual(len(r['excluded_reference_types']),1)
            self.assertFalse(next(m for m in r['matches'] if m['action']=='cash_counter_loading')['recovered'])

if __name__=='__main__':unittest.main()
