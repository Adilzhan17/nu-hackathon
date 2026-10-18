import tempfile
from pathlib import Path
import unittest
import av
from PIL import Image
from .bridge import prepare
from .bridge import probe
from unittest.mock import patch
import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from .server import make_handler


def fixture(folder,seconds):
    with av.open(str(folder/'source'),'w',format='mp4') as output:
        stream=output.add_stream('libx264',rate=10);stream.width=64;stream.height=64;stream.pix_fmt='yuv420p';stream.thread_count=2
        for index in range(round(seconds*10)):
            frame=av.VideoFrame.from_image(Image.new('RGB',(64,64),(index%255,0,0)))
            for packet in stream.encode(frame):output.mux(packet)
        for packet in stream.encode():output.mux(packet)

class SamplingTests(unittest.TestCase):
    def test_jury_footage_is_not_exposed_by_public_lab_routes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);directory=root/'runs/gemma_lab/uploads'
            ids={'a'*32:'jury_api','b'*32:'web_full'}
            for jid,profile in ids.items():
                folder=directory/jid;folder.mkdir(parents=True)
                (folder/'job.json').write_text(json.dumps(dict(id=jid,profile=profile,status='completed',name='private.mov')))
            server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(root))
            thread=threading.Thread(target=lambda:server.serve_forever(poll_interval=.01),daemon=True);thread.start()
            try:
                def get(path):
                    connection=http.client.HTTPConnection('127.0.0.1',server.server_port)
                    connection.request('GET',path);response=connection.getresponse()
                    data=json.loads(response.read());connection.close();return response.status,data
                status,data=get('/status')
                self.assertEqual(status,200)
                self.assertEqual([j['id'] for j in data['recent']],['b'*32])
                for suffix in ('','/media','/frames/f00000.jpg'):
                    self.assertEqual(get('/jobs/'+'a'*32+suffix)[0],404)
                self.assertEqual(get('/jobs/'+'b'*32)[0],200)
            finally:server.shutdown();server.server_close();thread.join()

    def test_72_7_seconds_is_not_rejected_or_truncated_at_one_minute(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);fixture(folder,72.7)
            plan=prepare(folder)
            self.assertAlmostEqual(plan['duration_sec'],72.7)
            self.assertEqual(len(plan['frames']),146)
            self.assertAlmostEqual(plan['frames'][-1]['timestamp'],72.5)
            self.assertAlmostEqual(plan['frames'][120]['timestamp'],60)

    def test_oversized_duration_rejected_before_accepting_gpu_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);fixture(folder,181)
            with self.assertRaisesRegex(ValueError,'181.0.*180'):
                probe(folder/'source')
            server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(folder))
            threading.Thread(target=server.serve_forever,daemon=True).start()
            try:
                with patch('ml.gemma_lab.server.process') as inference:
                    c=http.client.HTTPConnection('127.0.0.1',server.server_port)
                    c.request('POST','/jobs?name=long.mp4',body=(folder/'source').read_bytes(),headers={'X-Gemma-Local-Proxy':'1'})
                    response=c.getresponse();data=json.loads(response.read());c.close()
                    self.assertEqual(response.status,422);self.assertFalse(data['analysis_started'])
                    self.assertIn('181.0',data['error']);inference.assert_not_called()
            finally:server.shutdown();server.server_close()

    def test_half_second_schedule_and_video_only_preview(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            with av.open(str(folder/'source'),'w',format='mp4') as output:
                stream=output.add_stream('libx264',rate=10);stream.width=64;stream.height=64;stream.pix_fmt='yuv420p'
                for index in range(21):
                    frame=av.VideoFrame.from_image(Image.new('RGB',(64,64),(index*10,0,0)))
                    for packet in stream.encode(frame):output.mux(packet)
                for packet in stream.encode():output.mux(packet)
            plan=prepare(folder)
            self.assertEqual([f['requested_sec'] for f in plan['frames']],[0,.5,1,1.5,2])
            self.assertTrue(all(abs(f['timestamp']-f['requested_sec'])<.101 for f in plan['detail_frames']))
            self.assertEqual(set(plan),{'schema_version','duration_sec','source_sha256','frames','detail_frames'})
            with av.open(str(folder/'preview.mp4')) as preview:
                self.assertEqual([s.type for s in preview.streams],['video'])

    def test_jury_profile_uses_bounded_two_second_overview(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);fixture(folder,10)
            plan=prepare(folder,overview_step=2.0)
            self.assertEqual([f['requested_sec'] for f in plan['frames']],[0,2,4,6,8])
            self.assertGreater(len(plan['detail_frames']),len(plan['frames']))

    def test_adaptive_jury_step_caps_long_video_overview(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);fixture(folder,72.7)
            metadata=probe(folder/'source');step=max(2.0,__import__('math').ceil(metadata['duration_sec']/17*4)/4)
            plan=prepare(folder,overview_step=step)
            self.assertLessEqual(len(plan['frames']),18)
            self.assertGreaterEqual(step,4.0)

if __name__=='__main__':unittest.main()
