import csv
import json
import tempfile
import unittest
from pathlib import Path

from .deep_report import merged_moments,select_review,review_tasks
from .import_reviews import validate_decision
from .normalize_deep import normalize_record
from .training_data import export_clip


class DatasetTests(unittest.TestCase):
    def test_vfr_export_pts_strictly_increasing(self):
        import av,numpy as np
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'src.mp4';dst=Path(td)/'dst.mp4'
            with av.open(str(src),'w') as out:
                s=out.add_stream('libx264',rate=30);s.width=64;s.height=64;s.pix_fmt='yuv420p'
                for i in range(90):
                    pixels=np.full((64,64,3),i,dtype=np.uint8)
                    for packet in s.encode(av.VideoFrame.from_ndarray(pixels,format='rgb24')):out.mux(packet)
                for packet in s.encode():out.mux(packet)
            result=export_clip(src,dst,.1,2.8,fps=25,max_side=64)
            with av.open(str(dst)) as c:
                self.assertEqual(len(c.streams.audio),0)
                times=[f.time for f in c.decode(video=0)]
            self.assertGreater(len(times),60)
            self.assertTrue(all(b>a for a,b in zip(times,times[1:])))
            self.assertAlmostEqual(times[-1],2.7,places=2)

    def test_only_overlapping_same_action_moments_merge(self):
        base=dict(video='v',group_id='g',objects=['cash'],observed_action='cash_counting',confidence=.8,
                  uncertainty_reason='',conflict_ids=[])
        p=[dict(base,proposal_id=str(i),start_sec=a,end_sec=b,evidence_timestamps=[a,b]) for i,(a,b) in enumerate([(1,3),(2,4),(4.5,5.5)])]
        m=merged_moments(p)
        self.assertEqual(len(m),2)
        self.assertEqual(m[0]['proposal_ids'],['0','1'])
        self.assertFalse(m[0]['eligible_for_action_training'])

    def test_explicit_human_decision_and_objects_required(self):
        m=dict(moment_id='m',group_id='g',video='v')
        d=dict(m,decision='accept',reviewer='reviewer',start_sec=1.,end_sec=2.,observed_action='cash_counting',objects=['cash'],note='',reviewed_at='2026-10-17')
        validate_decision(d,m,3.)
        for patch in [dict(reviewer=''),dict(objects=['document']),dict(end_sec=4.),dict(group_id='other')]:
            with self.assertRaises(ValueError):validate_decision(dict(d,**patch),m,3.)

    def test_missing_prose_can_be_normalized_without_inventing_evidence(self):
        c=dict(candidate_id='c',video='v',group_id='g',start_sec=0.,end_sec=1.,frames=[dict(timestamp=0.),dict(timestamp=1.)])
        plan=dict(registry={'v':dict(group_id='g',split='train')},durations={'g':2.},model_lock=dict(model_id='m',model_revision='a'*40))
        e=dict(observed_action='cash_counting',objects=['cash'],start_frame=0,end_frame=1,evidence_frames=[0,1],confidence=.8,uncertainty_reason='')
        r=dict(raw_response=json.dumps(dict(sequence_summary='test',events=[e])),elapsed_sec=1.)
        result=normalize_record(r,c,plan)
        self.assertEqual(len(result['proposals']),1)
        self.assertEqual(result['event_details'][0]['evidence_description'],'')
        e['evidence_frames']=[0,7];r['raw_response']=json.dumps(dict(sequence_summary='test',events=[e]))
        self.assertEqual(normalize_record(r,c,plan)['proposals'],[])

    def test_priority_queue_covers_videos(self):
        tasks=[dict(review_id=str(i),group_id=f'g{i//3}',video=f'v{i//3}',start_sec=float(i),priority=100-i) for i in range(12)]
        selected=select_review(tasks,6)
        self.assertEqual(len(selected),6)
        self.assertEqual({s['group_id'] for s in selected},{'g0','g1','g2','g3'})



class RiskEvidenceTests(unittest.TestCase):
    def test_pattern_requires_positive_state_across_departure(self):
        from .incident_report import collect
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'incident_check_records').mkdir()
            (root/'incident_check_summary.json').write_text('{"completed":true}')
            check=dict(check_id='c',video='v',group_id='g',frames=[dict(timestamp=float(i)) for i in range(5)],organizer_video_verdict='нарушение')
            (root/'incident_check_plan.json').write_text(json.dumps(dict(checks=[check])))
            d=dict(departure_frames=[2,4],departure_confidence=.8,uncertainty_reason='',states=[dict(state='safe_door_open',evidence_frames=[0,1],confidence=.7,uncertainty_reason='')])
            f=root/'incident_check_records/c.json';f.write_text(json.dumps(dict(parsed=d,raw_response=json.dumps(d))))
            plan=dict(model_lock=dict(model_id='m',model_revision='a'*40))
            states,patterns,_=collect(root,plan)
            self.assertEqual(len(states),1);self.assertEqual(patterns,[])
            d['states'][0]['evidence_frames']=[0,4];f.write_text(json.dumps(dict(parsed=d,raw_response=json.dumps(d))))
            states,patterns,_=collect(root,plan)
            self.assertEqual(len(patterns),1)
            self.assertIsNone(patterns[0]['legal_violation_target'])
            self.assertFalse(patterns[0]['eligible_for_violation_type_training'])

if __name__=='__main__':unittest.main()
