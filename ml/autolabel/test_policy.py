import json
import tempfile
import unittest
from pathlib import Path
from .object_frames import validate_decision
from .rules import absence_assessment,assessments
from .video_bags import VideoBagDataset

class ObjectReviewTests(unittest.TestCase):
    def setUp(self):
        self.source=dict(image_id='f',image_sha256='a'*64,width=100,height=50)
        self.decision=dict(image_id='f',image_sha256='a'*64,reviewer='human',review_status='complete',image_assessment='targets_annotated',boxes=[dict(visibility='clear',**{'class':'cash'},xyxy=[0.,1.,10.,40.])])
    def test_reliable_boxes_and_explicit_negative(self):
        self.assertTrue(validate_decision(self.decision,self.source))
        d=dict(self.decision,boxes=[],image_assessment='confirmed_no_target')
        self.assertTrue(validate_decision(d,self.source))
    def test_uncertain_or_incomplete_is_not_negative(self):
        for patch in [dict(review_status='needs_context'),dict(image_assessment='uncertain'),dict(boxes=[dict(self.decision['boxes'][0],visibility='uncertain')])]:
            self.assertFalse(validate_decision(dict(self.decision,**patch),self.source))
    def test_bad_provenance_and_coordinates_rejected(self):
        for patch in [dict(reviewer=''),dict(image_sha256='wrong'),dict(boxes=[dict(self.decision['boxes'][0],xyxy=[1,0,101,40])]),dict(boxes=[dict(self.decision['boxes'][0],xyxy=[1,0,float('nan'),40])]),dict(image_assessment='confirmed_no_target')]:
            with self.assertRaises(ValueError):validate_decision(dict(self.decision,**patch),self.source)

class ObjectExportTests(unittest.TestCase):
    def test_coco_yolo_exclude_uncertain_image(self):
        from unittest.mock import patch
        from PIL import Image
        from .object_frames import import_decisions
        from .dataset import digest,read_jsonl
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'data/autolabel/source';source.mkdir(parents=True);(source/'images').mkdir()
            rows=[];decisions=[]
            for i in range(3):
                path=source/'images'/f'{i}.jpg';Image.new('RGB',(100,50)).save(path)
                row=dict(image_id=str(i),image_sha256=digest(path),image=f'images/{i}.jpg',video='v',group_id='g',split='train',source_sha256='h',timestamp=float(i),width=100,height=50)
                rows.append(row)
                boxes=[] if i==1 else [dict(visibility='clear' if i==0 else 'uncertain',**{'class':'cash'},xyxy=[10,10,20,20])]
                decisions.append(dict(image_id=str(i),image_sha256=row['image_sha256'],reviewer='synthetic_test_only',review_status='complete',image_assessment='confirmed_no_target' if i==1 else 'targets_annotated',boxes=boxes))
            (source/'object_review_manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            (source/'summary.json').write_text('{"protected_inputs":{}}')
            decision_file=root/'decisions.jsonl';decision_file.write_text('\n'.join(json.dumps(r) for r in decisions))
            output=root/'data/autolabel/reviewed'
            with patch('ml.autolabel.object_frames.snapshot',return_value={}),patch('ml.autolabel.object_frames.registry',return_value=({'v':dict(group_id='g',split='train',sha256='h')},None,None)):
                import_decisions(root,source,decision_file,output)
            coco=json.loads((output/'coco_train.json').read_text())
            self.assertEqual(len(coco['images']),2);self.assertEqual(len(coco['annotations']),1)
            self.assertEqual(coco['annotations'][0]['bbox'],[10,10,10,10])
            self.assertEqual((output/'labels/0.txt').read_text().split(),['0','0.15000000','0.30000000','0.10000000','0.20000000'])
            self.assertEqual((output/'labels/1.txt').read_text(),'')
            self.assertFalse((output/'labels/2.txt').exists())
            self.assertEqual(len(read_jsonl(output/'pending_images.jsonl')),1)

class RuleTests(unittest.TestCase):
    def test_missing_check_annotation_never_means_violation(self):
        self.assertEqual(absence_assessment({})['assessment'],'insufficient_data')
        d=absence_assessment(dict(whole_check_stage_visible=True,no_relevant_occlusion=True))
        self.assertEqual(d['assessment'],'insufficient_data');self.assertIsNone(d['legal_violation_target'])
    def test_rule_candidate_stays_pending(self):
        c=dict(check_id='c',video='v',group_id='g',start_sec=0,end_sec=5)
        rows=assessments([c],[dict(check_id='c',pattern='safe_open_across_worker_departure',pattern_id='p')])
        self.assertEqual(rows[0]['assessment'],'requires_visual_and_rule_review')
        self.assertEqual(sum(r['assessment']=='insufficient_data' for r in rows),3)
        self.assertTrue(all(r['legal_violation_target'] is None for r in rows))

class CompactQueueTests(unittest.TestCase):
    def test_overlap_merges_but_does_not_expand_beyond_budget(self):
        from .queue_compact import compact
        base=dict(group_id='g',video='v',priority=90,proposal_ids=[],conflict_ids=[],reasons=['x'],observations=[])
        tasks=[dict(base,review_id=str(i),start_sec=a,end_sec=b,context_start_sec=max(0,a-2),context_end_sec=b+2) for i,(a,b) in enumerate([(0,8),(1,7),(7,15)])]
        result=compact(tasks)
        self.assertEqual(len(result),2);self.assertEqual(result[0]['merged_task_ids'],['0','1'])
        self.assertTrue(all(t['end_sec']-t['start_sec']<=12 for t in result))

class VideoLabelTests(unittest.TestCase):
    def test_frozen_groups_positive_windows_unknown_and_loader(self):
        root=Path(__file__).resolve().parents[2];artifact=root/'data/autolabel/training_v2'
        if not artifact.exists():self.skipTest('Actual data artifacts unavailable')
        ds=VideoBagDataset(root,artifact,frames=4,size=64)
        self.assertEqual(len(ds),32)
        pos=next(i for i,r in enumerate(ds.rows) if r['organizer_video_target']==1)
        neg=next(i for i,r in enumerate(ds.rows) if r['organizer_video_target']==0)
        a=ds[pos];b=ds[neg]
        self.assertEqual(a['frames'].shape[1:],(3,4,64,64))
        self.assertTrue((a['interval_targets']==-1).all());self.assertFalse(a['interval_label_mask'].any())
        self.assertTrue((b['interval_targets']==0).all());self.assertTrue(b['interval_label_mask'].all())
    def test_propagated_positive_label_rejected(self):
        root=Path(__file__).resolve().parents[2];artifact=root/'data/autolabel/training_v2'
        if not artifact.exists():self.skipTest('Actual data artifacts unavailable')
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);(out/'summary.json').write_bytes((artifact/'summary.json').read_bytes())
            rows=[json.loads(x) for x in (artifact/'video_bags_train.jsonl').read_text().splitlines()]
            next(r for r in rows if r['organizer_video_target']==1)['windows'][0]['interval_target']=1
            (out/'video_bags_train.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaises(ValueError):VideoBagDataset(root,out)


class HumanActionImportTests(unittest.TestCase):
    def test_user_reference_requires_explicit_review_and_keeps_source(self):
        from unittest.mock import patch
        from .import_reviews import main
        from .dataset import read_jsonl
        import av,numpy as np
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);report=root/'data/autolabel/report';report.mkdir(parents=True)
            source=root/'data/raw/open/v.mp4';source.parent.mkdir(parents=True)
            with av.open(str(source),'w') as out:
                stream=out.add_stream('libx264',rate=30);stream.width=64;stream.height=64;stream.pix_fmt='yuv420p'
                for i in range(90):
                    for packet in stream.encode(av.VideoFrame.from_ndarray(np.full((64,64,3),i,dtype=np.uint8),format='rgb24')):out.mux(packet)
                for packet in stream.encode():out.mux(packet)
            (report/'summary.json').write_text('{"protected_inputs":{}}');(report/'moments.jsonl').write_text('')
            (report/'window_analysis.jsonl').write_text(json.dumps(dict(candidate_id='candidate_00000',video='v.mp4',group_id='g'))+'\n')
            (report/'user_reference_alignment.jsonl').write_text(json.dumps(dict(reference_id='observation_0000',video='v.mp4',group_id='g'))+'\n')
            (report/'video_timelines.jsonl').write_text(json.dumps(dict(group_id='g',duration_sec=3.,organizer_labels=[dict(source_verdict='без нарушения')]))+'\n')
            d=dict(moment_id='reference_observation_0000_1234567890123',group_id='g',video='v.mp4',decision='accept',reviewer='synthetic_test_only',start_sec=.5,end_sec=1.5,observed_action='cash_counting',objects=['cash'],note='synthetic fixture',reviewed_at='2026-10-17')
            decision_file=root/'decisions.jsonl';decision_file.write_text(json.dumps(d)+'\n');output=root/'data/autolabel/accepted'
            args=['import_reviews','--root',str(root),'--report',str(report),'--decisions',str(decision_file),'--output',str(output)]
            with patch('sys.argv',args),patch('ml.autolabel.import_reviews.snapshot',return_value={}),patch('ml.autolabel.import_reviews.registry',return_value=({'v.mp4':dict(group_id='g',split='train')},None,None)):
                main()
            accepted=read_jsonl(output/'reviewed_actions.jsonl')
            self.assertEqual(len(accepted),1);self.assertEqual(accepted[0]['source_reference_id'],'observation_0000')
            self.assertEqual(accepted[0]['event_violation_target'],'not_annotated');self.assertTrue((output/accepted[0]['clip']).exists())

class StateNormalizationTests(unittest.TestCase):
    def test_optional_prose_recovery_does_not_repair_bad_evidence(self):
        from .normalize_incidents import normalize
        c=dict(check_id='c',frames=[dict(timestamp=0),dict(timestamp=1)])
        state=dict(state='safe_door_open',evidence_frames=[0,1],confidence=.7)
        d=dict(departure_frames=[],departure_confidence=0,states=[state])
        r=normalize(dict(raw_response=json.dumps(d)),c)
        self.assertEqual(r['status'],'ok');self.assertEqual(len(r['normalizations']),2)
        d['states'][0]['evidence_frames']=[0,9]
        r=normalize(dict(raw_response=json.dumps(d)),c)
        self.assertEqual(r['status'],'partially_invalid');self.assertEqual(r['parsed']['states'],[])

if __name__=='__main__':unittest.main()
