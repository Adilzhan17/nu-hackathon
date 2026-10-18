import base64
import hashlib
import http.client
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from infer_video import evidence,file_sha256
from jury_api import (APIError,JuryAPI,MAX_UPLOAD,READ_CHUNK,RequestReader,
                      completed_result,configured_key,multipart_file,multipart_header)
from train_x3d import atomic_json
from video_jobs import VideoJobs


class FragmentStream(io.BytesIO):
    def __init__(self,value,width):super().__init__(value);self.width=width;self.max_request=0
    def read(self,n=-1):
        if n<0:raise AssertionError('Unbounded read')
        self.max_request=max(self.max_request,n)
        return super().read(min(n,self.width))


def multipart(video=b'opaque mov bytes',name='video.mov',boundary=b'boundary-123'):
    return (b'--'+boundary+b'\r\nContent-Disposition: form-data; name="file"; filename="'+name.encode()+
            b'"\r\nContent-Type: video/quicktime\r\n\r\n'+video+b'\r\n--'+boundary+b'--\r\n')


class JuryAPITests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        checkpoint=self.root/'artifacts/models/x3d/model.pt';checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'local weights')
        self.jobs=VideoJobs(self.root,start_worker=False,directory=self.root/'data/jury_jobs',private=True)
        self.api=JuryAPI(self.root,jobs=self.jobs,key='test-key')
        api=self.api
        class Handler(BaseHTTPRequestHandler):
            def json_response(self,value,status=200):
                body=json.dumps(value).encode();self.send_response(status)
                self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
                self.end_headers();self.wfile.write(body)
            def do_POST(self):
                if not api.handle_post(self):self.send_error(404)
            def do_GET(self):
                if not api.handle_get(self):self.send_error(404)
            def log_message(self,*args):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=self.server.serve_forever,daemon=True);thread.start()
        self.addCleanup(self.server.server_close);self.addCleanup(self.server.shutdown)

    def request(self,method,path,body=None,headers=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        conn.request(method,path,body=body,headers=headers or {})
        result=conn.getresponse();raw=result.read();status=result.status;response_headers=dict(result.getheaders());conn.close()
        try:value=json.loads(raw) if response_headers.get('Content-Type','').startswith('application/json') else raw
        except (ValueError,UnicodeDecodeError):value=raw
        return status,value,response_headers

    def upload(self,video=b'test mov',name='video.mov',key='test-key'):
        return self.request('POST','/api/v1/analyze',multipart(video,name),
            {'Content-Type':'multipart/form-data; boundary=boundary-123','X-API-Key':key})

    def completed(self,positive=True):
        state=self.jobs.create(dict(name='case.mov',size=4,models=['x3d']))
        folder=self.jobs.folder(state['id']);(folder/'source.part').rename(folder/'source')
        (folder/'source').write_bytes(b'mov!')
        state.update(status='completed',updated_at=time.time(),duration_sec=4.,source_sha256=file_sha256(folder/'source'))
        result=dict(status='completed',processed=1,total=1,threshold=.5,model_sha256='a'*64,
                    windows=[dict(start_sec=0.,end_sec=4.,score=.9 if positive else .1)],events=[])
        if positive:
            Image.new('RGB',(16,12),'red').save(folder/'x3d-0.jpg',format='JPEG')
            result['events']=[dict(start_sec=0.,end_sec=4.,score=.9,timecode_sec=2.,evidence='x3d-0.jpg',
                                  evidence_timestamp_sec=2.04,evidence_sha256=file_sha256(folder/'x3d-0.jpg'))]
        state['results']={'x3d':result};atomic_json(folder/'job.json',state)
        return state,folder

    def test_auth_on_upload_status_and_preview(self):
        self.assertEqual(self.upload(key='wrong')[0],401)
        for suffix in ('','/preview'):
            status,value,_=self.request('GET','/api/v1/jobs/'+'a'*32+suffix)
            self.assertEqual(status,401);self.assertEqual(value['error']['code'],'unauthorized')
        self.assertEqual(list(self.jobs.directory.iterdir()),[])

    def test_upload_uses_private_storage_and_never_public_jobs(self):
        status,response,_=self.upload(b'a'*100000)
        self.assertEqual(status,202);jid=response['job_id']
        self.assertEqual((self.jobs.folder(jid)/'source').read_bytes(),b'a'*100000)
        self.assertEqual(self.jobs.get(jid)['purpose'],'jury')
        self.assertEqual(self.jobs.tasks.qsize(),1)
        public=VideoJobs(self.root,start_worker=False)
        with self.assertRaises(FileNotFoundError):public.folder(jid)
        self.assertEqual(self.jobs.directory.stat().st_mode&0o777,0o700)
        status,response,_=self.request('GET',f'/api/v1/jobs/{jid}',headers={'X-API-Key':'test-key'})
        self.assertEqual(response['status'],'processing')

    def test_multipart_rejects_wrong_extension_extra_parts_and_truncation(self):
        self.assertEqual(self.upload(name='renamed.jpg')[0],415)
        body=multipart();boundary=b'boundary-123'
        cases=[body[:-6],body.replace(b'--boundary-123--\r\n',b'--boundary-123\r\nContent-Disposition: form-data; name="other"\r\n\r\nx\r\n--boundary-123--\r\n')]
        for body in cases:
            status,_,_=self.request('POST','/api/v1/analyze',body,{'X-API-Key':'test-key','Content-Type':'multipart/form-data; boundary=boundary-123'})
            self.assertEqual(status,400)
        self.assertEqual(list(self.jobs.directory.iterdir()),[])

    def test_streaming_boundary_at_every_fragment_position(self):
        video=b'\x00\xff'+b'q'*400+b'\r\n--boundary-123AB'+b'z'*100
        body=multipart(video)
        for width in (1,2,3,17,64,READ_CHUNK):
            stream=FragmentStream(body,width);reader=RequestReader(stream,len(body),time.time()+30)
            self.assertEqual(multipart_header(reader,b'boundary-123'),'video.mov')
            size,digest=multipart_file(reader,b'boundary-123',self.root/'part')
            self.assertEqual((self.root/'part').read_bytes(),video);self.assertEqual(size,len(video))
            self.assertEqual(digest,hashlib.sha256(video).hexdigest());self.assertLessEqual(stream.max_request,READ_CHUNK)

    def test_upload_size_and_transfer_encoding_rejected_before_reading(self):
        headers={'X-API-Key':'test-key','Content-Type':'multipart/form-data; boundary=b','Content-Length':str(MAX_UPLOAD+20000)}
        self.assertEqual(self.request('POST','/api/v1/analyze',b'',headers)[0],413)
        headers['Content-Length']='1';headers['Transfer-Encoding']='chunked'
        self.assertEqual(self.request('POST','/api/v1/analyze',b'',headers)[0],411)

    def test_upload_read_deadline_covers_every_chunk(self):
        reader=RequestReader(io.BytesIO(b'abc'),3,time.time()-1)
        with self.assertRaises(APIError) as failure:reader.read(1)
        self.assertEqual(failure.exception.code,'deadline_exceeded')
        self.assertEqual(reader.remaining,3)

    def test_positive_has_actual_jpeg_and_no_invented_action_or_legal_rule(self):
        state,folder=self.completed();response=self.api.job_result(state['id'])
        self.assertEqual(response['status'],'done');self.assertEqual(response['verdict'],'violation')
        event=response['violations'][0]
        self.assertEqual(event['classification_status'],'unclassified');self.assertIsNone(event['regulatory_source'])
        self.assertIn('не установлены',event['description']);self.assertTrue(response['warnings'])
        frame=event['evidence_frames'][0]
        self.assertEqual(frame['timestamp_sec'],2.04);self.assertEqual(event['timecode_sec'],2.04)
        self.assertEqual(base64.b64decode(frame['image_base64']),(folder/'x3d-0.jpg').read_bytes())

    def test_no_candidates_requires_complete_coverage(self):
        state,folder=self.completed(positive=False)
        self.assertEqual(self.api.job_result(state['id'])['verdict'],'no_violation')
        state['results']['x3d']['windows'][0]['end_sec']=2.
        atomic_json(folder/'job.json',state)
        response=self.api.job_result(state['id'])
        self.assertEqual(response['status'],'failed');self.assertNotIn('verdict',response)

    def test_missing_forged_or_tampered_evidence_fails_not_no_violation(self):
        state,folder=self.completed()
        (folder/'x3d-0.jpg').write_bytes(b'not a jpeg')
        self.assertEqual(self.api.job_result(state['id'])['status'],'failed')
        state['results']['x3d']['events'][0]['evidence']='../../secret.jpg';atomic_json(folder/'job.json',state)
        self.assertEqual(self.api.job_result(state['id'])['status'],'failed')

    def test_queue_deadline_counts_and_worker_does_not_start_expired_job(self):
        state=self.jobs.create(dict(name='case.mov',size=4,models=['x3d']),deadline_at=time.time()-1)
        folder=self.jobs.folder(state['id']);state['status']='queued';atomic_json(folder/'job.json',state)
        self.assertEqual(self.api.job_result(state['id'])['error']['code'],'deadline_exceeded')
        with patch('video_jobs.subprocess.run') as launch:self.jobs.run_job(state['id'])
        launch.assert_not_called()
        self.assertEqual(self.jobs.get(state['id'])['status'],'failed')

    def test_worker_uses_only_remaining_whole_job_budget(self):
        state=self.jobs.create(dict(name='case.mov',size=4,models=['x3d']),deadline_at=time.time()+12)
        folder=self.jobs.folder(state['id']);state['status']='queued';atomic_json(folder/'job.json',state)
        with patch('video_jobs.subprocess.run',side_effect=subprocess.TimeoutExpired('local inference',11)) as launch:
            self.jobs.run_job(state['id'])
        self.assertLessEqual(launch.call_args.kwargs['timeout'],12)
        self.assertEqual(self.jobs.get(state['id'])['error_code'],'deadline_exceeded')

    def test_completed_after_deadline_is_failed(self):
        state,folder=self.completed(False);state['updated_at']=state['deadline_at']+1;atomic_json(folder/'job.json',state)
        self.assertEqual(self.api.job_result(state['id'])['status'],'failed')

    def test_authenticated_preview_supports_range_only_private_job(self):
        state,folder=self.completed(False);state['preview_ready']=True;atomic_json(folder/'job.json',state)
        (folder/'preview.mp4').write_bytes(b'0123456789')
        status,raw,headers=self.request('GET',f'/api/v1/jobs/{state["id"]}/preview',headers={'X-API-Key':'test-key','Range':'bytes=2-5'})
        self.assertEqual(status,206);self.assertEqual(raw,b'2345');self.assertEqual(headers['Content-Range'],'bytes 2-5/10')
        status,_,_=self.request('GET',f'/api/v1/jobs/{state["id"]}/preview',headers={'X-API-Key':'test-key','Range':'bytes=100-'})
        self.assertEqual(status,416)

    def test_key_file_rejects_world_readable_permissions_and_symlinks(self):
        secret=self.root/'.secrets/jury_api_key';secret.parent.mkdir();secret.write_text('x'*32);secret.chmod(0o644)
        with patch.dict(os.environ,{},clear=True):
            self.assertIsNone(configured_key(self.root));secret.chmod(0o600)
            self.assertEqual(configured_key(self.root),b'x'*32)
            secret.rename(secret.with_name('real'));secret.symlink_to(secret.with_name('real'))
            self.assertIsNone(configured_key(self.root))

    def test_jpeg_renamed_mov_is_failed_during_real_decode(self):
        state=self.jobs.create(dict(name='case.mov',size=4,models=['x3d']))
        folder=self.jobs.folder(state['id']);Image.new('RGB',(16,16),'blue').save(folder/'source',format='JPEG')
        state['status']='queued';atomic_json(folder/'job.json',state)
        process=subprocess.run([sys.executable,str(Path(__file__).with_name('infer_video.py')),'--job',str(folder)],capture_output=True,timeout=20)
        self.assertNotEqual(process.returncode,0)
        final=self.jobs.get(state['id']);self.assertEqual(final['status'],'failed');self.assertFalse(final.get('preview_ready'))

    def test_evidence_timestamp_is_decoded_pts_not_requested_midpoint(self):
        import av
        from fractions import Fraction
        video=self.root/'synthetic.mov'
        for offset in (0,30):
            with av.open(str(video),'w') as output:
                track=output.add_stream('mpeg4',rate=10);track.width=32;track.height=24;track.pix_fmt='yuv420p'
                for i in range(10):
                    frame=av.VideoFrame.from_image(Image.new('RGB',(32,24),(i*20,0,0)))
                    frame.pts=i+offset;frame.time_base=Fraction(1,10)
                    for packet in track.encode(frame):output.mux(packet)
                for packet in track.encode():output.mux(packet)
            stamp=evidence(video,.31,self.root/'actual.jpg')
            self.assertAlmostEqual(stamp,.4,places=5)
            with Image.open(self.root/'actual.jpg') as image:self.assertEqual(image.format,'JPEG')


if __name__=='__main__':unittest.main()
