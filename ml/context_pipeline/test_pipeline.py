import json
from pathlib import Path
import tempfile
import unittest
from .bridge import assign_tracks
from .objects import tile_origins,nms,active_detector
from ml.live_console.publish import publish
from .evaluate_detector import match
from .publish_detector import check_promotion

class PipelineTests(unittest.TestCase):
    def test_zero_cash_recall_cannot_be_published_as_improvement(self):
        metrics=dict(support=9,tp=0,precision=0.,recall=0.,map50_95=.1)
        condition=dict(per_class=dict(cash=metrics,document_id=metrics))
        with self.assertRaisesRegex(ValueError,'no true positive'):
            check_promotion(dict(conditions=dict(old_tiles=condition,new_tiles=condition)))

    def test_tiles_cover_native_frame_and_overlap(self):
        starts=tile_origins(1920,960)
        self.assertEqual(starts,[0,720,960])
        self.assertEqual(starts[-1]+960,1920)
        self.assertTrue(all(b<a+960 for a,b in zip(starts,starts[1:])))
        self.assertEqual(tile_origins(500,960),[0])

    def test_nms_does_not_merge_different_classes(self):
        box=dict(xyxy=[.1,.1,.3,.3],score=.9,class_id=0)
        self.assertEqual(len(nms([box,dict(box,score=.8),dict(box,class_id=1)])),2)

    def test_real_portrait_sources_receive_multiple_native_tiles(self):
        self.assertGreater(len(tile_origins(464,384))*len(tile_origins(848,384)),1)

    def test_metric_matching_is_class_aware_and_one_to_one(self):
        box=dict(xyxy=[.1,.1,.3,.3],score=.9,class_id=0)
        correct,used,_=match([dict(box,class_id=1),box,dict(box,score=.8)],[box],.5)
        self.assertEqual(correct,[False,True,False]);self.assertEqual(used,{0})

    def test_occlusion_breaks_track_and_never_proves_ownership(self):
        box=dict(xyxy=[.1,.1,.3,.3],score=.9,class_id=0)
        f=[dict(time_sec=t,boxes=[dict(box)]) for t in [0,1,5]]
        assign_tracks(f)
        self.assertEqual(f[0]['boxes'][0]['track_id'],f[1]['boxes'][0]['track_id'])
        self.assertNotEqual(f[1]['boxes'][0]['track_id'],f[2]['boxes'][0]['track_id'])
        self.assertEqual(f[1]['boxes'][0]['ownership'],'unknown')

    def test_checkpoint_outside_registry_root_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'runs/context_pipeline';p.mkdir(parents=True)
            (p/'detector_active.json').write_text(json.dumps(dict(checkpoint='/etc/passwd',sha256='x')))
            with self.assertRaises(ValueError):active_detector(root)

    def test_published_json_cannot_close_script(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'ml/live_console').mkdir(parents=True)
            (root/'ml/live_console/page.html').write_text('<script type="application/json">STATE_JSON</script>')
            publish(root,dict(value='</script><script>alert(1)</script>'))
            self.assertEqual((root/'ml/scenario.html').read_text().count('</script>'),1)

if __name__=='__main__':unittest.main()
