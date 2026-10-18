import io
import json
import tempfile
import unittest
from pathlib import Path
from infer_video import windows,merge_events
from video_jobs import VideoJobs,CHUNK_SIZE


class VideoJobTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        path=self.root/'artifacts/models/x3d';path.mkdir(parents=True);(path/'model.pt').write_bytes(b'test')
        self.jobs=VideoJobs(self.root,start_worker=False)

    def create(self,**changes):
        value=dict(name='проверка.mov',size=10,models=['x3d']);value.update(changes)
        return self.jobs.create(value)

    def test_cpu_metrics_require_same_checkpoint_and_complete_validation(self):
        folder=self.root/'artifacts/models/x3d'
        status=self.root/'runs/suite/x3d/status.json';status.parent.mkdir(parents=True)
        status.write_text(json.dumps(dict(checkpoint_sha256='abc')))
        predictions=[dict(video='v.mov',start_sec=0,target=1,probability=.7)]
        (folder/'validation.json').write_text(json.dumps(dict(metrics=dict(f1=.8,epoch=3),predictions=predictions)))
        cpu=dict(scope='organizer_windows',checkpoint_sha256='abc',metrics=dict(f1=.6),predictions=predictions)
        def actual():
            (folder/'cpu_validation.json').write_text(json.dumps(cpu))
            return next(m for m in self.jobs.models() if m['key']=='x3d')
        self.assertEqual(actual()['metrics'],dict(f1=.6,epoch=3))
        self.assertEqual(actual()['metrics_device'],'CPU')
        cpu['checkpoint_sha256']='old'
        self.assertEqual(actual()['metrics_device'],'GPU')
        cpu['checkpoint_sha256']='abc';cpu['predictions']=[]
        self.assertEqual(actual()['metrics_device'],'GPU')
        cpu['predictions']=predictions;cpu['scope']='cpu_smoke_subset'
        self.assertEqual(actual()['metrics_device'],'GPU')

    def test_chunks_reconstruct_bytes_and_submit_once(self):
        state=self.create();jid=state['id']
        self.jobs.chunk(jid,0,4,io.BytesIO(b'0123'))
        self.jobs.chunk(jid,4,6,io.BytesIO(b'456789'))
        self.assertEqual(self.jobs.submit(jid)['status'],'queued')
        self.jobs.submit(jid)
        self.assertEqual(self.jobs.tasks.qsize(),1)
        self.assertEqual((self.jobs.folder(jid)/'source').read_bytes(),b'0123456789')

    def test_offsets_size_and_incomplete_uploads_rejected(self):
        jid=self.create()['id']
        with self.assertRaises(FileExistsError):self.jobs.chunk(jid,1,1,io.BytesIO(b'x'))
        with self.assertRaises(ValueError):self.jobs.chunk(jid,0,11,io.BytesIO(b'long'))
        with self.assertRaises(ValueError):self.jobs.chunk(jid,0,CHUNK_SIZE+1,io.BytesIO(b'long'))
        with self.assertRaises(ValueError):self.jobs.submit(jid)

    def test_broken_chunk_rolls_back_partial_write(self):
        jid=self.create()['id']
        with self.assertRaises(ValueError):self.jobs.chunk(jid,0,10,io.BytesIO(b'short'))
        self.assertEqual(self.jobs.get(jid)['uploaded_bytes'],0)

    def test_unready_models_and_unsafe_ids_rejected(self):
        with self.assertRaises(ValueError):self.create(models=['videomaev2'])
        with self.assertRaises(ValueError):self.jobs.folder('../../etc')
        with self.assertRaises(ValueError):self.create(name='script.py')
        with self.assertRaises(ValueError):self.create(size=True)
        with self.assertRaises(ValueError):self.create(models=[{}])

    def test_incomplete_jobs_do_not_fill_unbounded_queue(self):
        for _ in range(4):self.create()
        with self.assertRaises(ValueError):self.create()

    def test_delete_cleans_failed_upload_but_not_queued_inference(self):
        jid=self.create()['id']
        self.jobs.delete(jid)
        self.assertFalse((self.jobs.directory/jid).exists())
        jid=self.create()['id']
        self.jobs.chunk(jid,0,10,io.BytesIO(b'0123456789'));self.jobs.submit(jid)
        with self.assertRaises(ValueError):self.jobs.delete(jid)

    def test_sliding_windows_cover_short_video_and_tail(self):
        self.assertEqual(windows(2),[(0.,2)])
        spans=windows(17.3)
        self.assertEqual(spans[0][0],0)
        self.assertEqual(spans[-1][1],17.3)
        self.assertTrue(all(a[1]>=b[0] for a,b in zip(spans,spans[1:])))

    def test_events_merge_overlap_without_inventing_precise_timestamp(self):
        scores=[dict(start_sec=0,end_sec=8,score=.8),dict(start_sec=4,end_sec=12,score=.9),
                dict(start_sec=8,end_sec=16,score=.1),dict(start_sec=16,end_sec=24,score=.7)]
        events=merge_events(scores)
        self.assertEqual(len(events),2)
        self.assertEqual(events[0],dict(start_sec=0,end_sec=12,score=.9,timecode_sec=8))


if __name__=='__main__':unittest.main()
