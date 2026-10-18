import unittest
import subprocess
from pathlib import Path
import numpy as np
from ml.human_training.actions import fit,evaluate,norm
from ml.human_training.detector import partition
class Tests(unittest.TestCase):
 def test_group_holdout(self):
  rng=np.random.default_rng(42);x=norm(rng.normal(size=(6,9)));base=norm(rng.normal(size=(6,4)));p=norm(rng.normal(size=(3,4)));labels=np.array([0,1,2,0,1,2]);groups=['a','a','b','b','c','c']
  scores,folds=evaluate(x,base,p[labels],p,labels,groups)
  self.assertEqual(scores.shape,(6,3));self.assertTrue(np.isfinite(scores).all())
  for f in folds:self.assertNotIn(f['test_group'],f['train_groups'])
 def test_target_changes_adapter(self):
  x=np.eye(3);base=np.eye(3);a=fit(x,base,base,['a','b','c']);b=fit(x,base,np.roll(base,1,axis=0),['a','b','c'])
  self.assertTrue(np.allclose(a,0));self.assertGreater(np.linalg.norm(b),.1)
 def test_partition_stable_and_by_video(self):
  rows=[{'group_id':g} for g in ['a','a','b','c','d','e','f']];held=partition(rows)
  self.assertEqual(held,partition(list(reversed(rows))));self.assertTrue(0<len(held)<6)
 def test_queue_only_hides_contained_same_human_action(self):
  html=(Path(__file__).parents[1]/'human_review.html').read_text()
  fn=html[html.index('function coveredByHuman'):html.index('function activeTasks')]
  js="const state={data:{decisions:{x:{eligible_for_action_training:true,group_id:'v',observed_action:'cash_placed',start_sec:2,end_sec:8}}}};"+fn+"""
const t={group_id:'v',suggested_action:'cash_placed',start_sec:3,end_sec:7};
if(!coveredByHuman(t))throw Error('matching contained interval');
for(const change of [{group_id:'other'},{suggested_action:'document_transfer'},{start_sec:1},{end_sec:9}])if(coveredByHuman({...t,...change}))throw Error('unrelated interval hidden');
state.data.decisions.x.eligible_for_action_training=false;
if(coveredByHuman(t))throw Error('uncertain label hides interval');
"""
  subprocess.run(['node','-e',js],check=True,capture_output=True)
 def test_one_group_rejected(self):
  with self.assertRaises(ValueError):partition([{'group_id':'a'}]*5)
if __name__=='__main__':unittest.main()
