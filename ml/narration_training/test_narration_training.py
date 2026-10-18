import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from torch.nn import functional as F

from .data import NarrationDataset, VideoBagDataset, loaders
from .model import TemporalNarrationModel, narration_loss
from .prepare import candidates, valid_interval
from .train import train_fit, folds, metrics


def synthetic_store():
    torch.manual_seed(18)
    store=SimpleNamespace(dim=8,action_ids=['cash_received','safe_closing'],
       prototypes=F.normalize(torch.randn(2,8),dim=-1),bags=[],clips=[],pairs=[],visual={},text={},group_split={})
    for i in range(12):
        group=f'g{i}';split='train' if i<10 else 'val';store.group_split[group]=split
        store.bags.append(dict(group_id=group,split=split,video_target=i%2))
        for j in range(2):
            key=f'v{i}_{j}';store.clips.append(dict(clip_id=key,group_id=group,kind='video_bag_window'))
            store.visual[key]=F.normalize(torch.randn(4,8),dim=-1)
        if split=='train':
            key=f'p{i}';store.pairs.append(dict(pair_id=key,group_id=group,eligible_text_loss=True,
              eligible_action_positive_loss=i%2==0,action_candidates=['cash_received'],text_loss_weight=1.))
            store.visual[key]=F.normalize(torch.randn(4,8),dim=-1);store.text[key]=F.normalize(torch.randn(8),dim=-1)
    return store


class TrainingTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)
    def test_time_bounds(self):
        for triple in [(0,1,2),(.5,2,2)]: self.assertTrue(valid_interval(*triple))
        for triple in [(-1,1,2),(1,1,2),(0,3,2),(0,float('nan'),2),(0,float('inf'),2),(False,1,2)]:
            self.assertFalse(valid_interval(*triple))
    def test_source_decomposition_not_violation(self):
        row=dict(action_from_source='cash_removed_and_safe_closed',caption='Достаёт деньги и закрывает сейф.')
        a,o,_,ambiguous,cardinality,_,inference=candidates(row)
        self.assertEqual(a,['cash_removed','safe_closing']);self.assertEqual(cardinality,'multiple_candidates')
        self.assertFalse(inference);self.assertFalse(ambiguous)
    def test_ambiguous_has_no_invented_action(self):
        a,_,_,ambiguous,_,_,_=candidates(dict(action_from_source='unclear_handheld_item',caption='Непонятный предмет.'))
        self.assertEqual(a,[]);self.assertTrue(ambiguous)
    def test_narration_train_only(self):
        with self.assertRaises(ValueError): NarrationDataset(synthetic_store(),['g10'])
    def test_grouped_folds_are_disjoint_and_exhaustive(self):
        store=synthetic_store();groups=[f'g{i}' for i in range(10)];seen=[]
        for fit,held in folds(store,groups,3,42):
            self.assertFalse(set(fit)&set(held));seen+=held
            ds=NarrationDataset(store,fit)
            self.assertFalse({r['group_id'] for r in ds.rows}&set(held))
        self.assertEqual(sorted(seen),sorted(groups))
    def test_unknown_actions_do_not_create_negative_loss(self):
        store=synthetic_store();model=TemporalNarrationModel(8)
        _,loader=loaders(store,['g0','g2']);batch=next(iter(loader))
        before,_=narration_loss(model,batch,store.prototypes)
        changed=store.prototypes.clone();changed[1]*=-1
        after,_=narration_loss(model,batch,changed)
        torch.testing.assert_close(before,after)
    def test_original_text_changes_video_adapter_gradients(self):
        store=synthetic_store();model=TemporalNarrationModel(8)
        _,loader=loaders(store,['g0','g1']);batch=next(iter(loader))
        loss,_=narration_loss(model,batch,store.prototypes);loss.backward()
        grad=model.adapter[-1].weight.grad.clone();model.zero_grad()
        altered={**batch,'text':-batch['text']}
        loss,_=narration_loss(model,altered,store.prototypes);loss.backward()
        self.assertFalse(torch.allclose(grad,model.adapter[-1].weight.grad))
    def test_temporal_order_can_change_encoding(self):
        model=TemporalNarrationModel(8)
        torch.nn.init.normal_(model.adapter[-1].weight)
        x=F.normalize(torch.randn(2,4,8),dim=-1)
        self.assertFalse(torch.allclose(model.encode(x),model.encode(x.flip(1))))
    def test_real_consumer_counts_and_checkpoint(self):
        store=synthetic_store();groups=['g0','g1','g2','g3']
        base,b=train_fit(store,groups,'baseline',.003,2,1)
        model,n=train_fit(store,groups,'narration',.003,2,1)
        self.assertEqual(b['unique_text_pairs'],0);self.assertEqual(n['unique_text_pairs'],4)
        self.assertEqual(n['excluded_pairs'],{'held_out_train_group':6})
        self.assertEqual(n['pair_epoch_exposures'],8)
        self.assertFalse(torch.allclose(base.adapter[-1].weight,model.adapter[-1].weight))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt';torch.save(model.state_dict(),path)
            restored=TemporalNarrationModel(8);restored.load_state_dict(torch.load(path,weights_only=True))
            torch.testing.assert_close(model.encode(store.visual['p0'][None]),restored.encode(store.visual['p0'][None]))
    def test_val_fit_refused(self):
        with self.assertRaises(ValueError): train_fit(synthetic_store(),['g10'],'baseline',.001,1,1)
    def test_metrics_known_result_and_fixed_threshold(self):
        result=metrics([dict(target=y,score=p) for y,p in [(0,.1),(0,.7),(1,.4),(1,.8)]])
        self.assertEqual(result['confusion_matrix'],[[1,1],[1,1]])
        self.assertEqual(result['accuracy'],.5);self.assertEqual(result['auroc'],.75);self.assertEqual(result['threshold'],.5)
    def test_bag_target_once_not_per_window(self):
        row=VideoBagDataset(synthetic_store(),['g1'])[0]
        self.assertEqual(row['target'].ndim,0);self.assertEqual(row['video'].shape[0],2)

if __name__=='__main__': unittest.main()
