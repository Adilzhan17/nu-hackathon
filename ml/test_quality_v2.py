import unittest
from quality_v2.experiment import folds,fit,evaluate

class QualitySelectionTests(unittest.TestCase):
    def test_whole_video_folds_do_not_overlap(self):
        bags=[dict(target=i%2) for i in range(32)]
        parts=folds(list(range(32)),bags)
        self.assertEqual(sorted(i for part in parts for i in part),list(range(32)))
        for part in parts:self.assertEqual({bags[i]['target'] for i in part},{0,1})

    def test_holdout_training_is_rejected(self):
        with self.assertRaises(ValueError):fit(None,[dict(split='val')],[0],.01)

    def test_metrics_do_not_hide_false_alarms(self):
        result=evaluate([0,0,1,1],[.8,.7,.9,.6],.5)
        self.assertEqual(result['accuracy'],.5)
        self.assertEqual(result['precision'],.5)
        self.assertEqual(result['confusion']['fp'],2)

    def test_retrieval_cannot_use_holdout_neighbors(self):
        from quality_v2.retrieval import predict
        with self.assertRaises(ValueError):predict(None,[dict(split='val')],[0],[1],1)
        with self.assertRaises(ValueError):predict(None,[dict(split='train')],[0],[0],1)

    def test_time_warp_distance_keeps_order(self):
        import numpy as np
        from quality_v2.retrieval import dtw
        x=np.eye(3,dtype=np.float32)
        self.assertEqual(dtw(x,x),0)
        self.assertGreater(dtw(x,x[::-1]),0)

if __name__=='__main__':unittest.main()
