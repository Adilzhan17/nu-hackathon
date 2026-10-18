"""Local selected-frame routes and removed cloud routes, using synthetic data."""
import hashlib
import http.client
from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from PIL import Image
from annotator import make_handler
from test_human_routes import HumanRoutesTests


class FakeFrames:
    def __init__(self,case):
        self.case=case;self.calls=0
        self.path=case.root/'data/human_review/synthetic-selected.jpg'
        Image.new('RGB',(40,60),'white').save(self.path)
        self.metadata=dict(path=self.path,width=40,height=60,image_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            timestamp_sec=3.0,requested_timestamp_sec=3,video=case.image_row['video'],group_id=case.image_row['group_id'],
            source_sha256=case.video_sha,split='train')
    def prepare_frame(self,payload):return self.metadata


class SelectedFrameRoutesTests(unittest.TestCase):
    def setUp(self):
        self.fixture=HumanRoutesTests();self.fixture.setUp()
        self.fixture.server.shutdown();self.fixture.server.server_close();self.fixture.thread.join()
        self.frames=FakeFrames(self.fixture)
        parent=make_handler(self.fixture.store,self.fixture.report/'previews',human=self.fixture.human,
            objects=self.fixture.objects,review_frames=self.frames)
        class Quiet(parent):
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Quiet)
        self.fixture.server=server
        self.fixture.thread=threading.Thread(target=lambda:server.serve_forever(poll_interval=.01),daemon=True)
        self.fixture.thread.start()
        self.fixture.origin='http://127.0.0.1:'+str(server.server_port)
    def tearDown(self):self.fixture.tearDown()

    def test_selected_frame_registration_native_image_and_no_auto_boxes(self):
        status,_,value=self.fixture.json_request('POST','/api/human-objects/frames',{'task_id':'synthetic'})
        self.assertEqual(status,201)
        image_id=value['image_id'];self.assertEqual(len(self.fixture.objects.rows),2)
        status,headers,raw=self.fixture.request('GET','/api/human-objects/images/'+image_id+'.jpg')
        self.assertEqual(status,200);self.assertEqual(headers['Content-Type'],'image/jpeg')
        self.assertEqual(raw,self.frames.path.read_bytes())
        self.assertEqual(self.frames.calls,0);self.assertEqual(self.fixture.objects.decisions(),{})

    def test_malformed_requests_are_rejected(self):
        for origin in ('','https://another.example','http://['):
            status,_,_=self.fixture.json_request('POST','/api/human-objects/frames',{},headers={'Origin':origin})
            self.assertEqual(status,403)
        for body in ('null','[]','{invalid'):
            status,_,_=self.fixture.json_request('POST','/api/human-objects/frames',body=body)
            self.assertEqual(status,422)
        self.assertEqual(self.frames.calls,0)

    def test_removed_cloud_routes_and_paths_are_not_exposed(self):
        status,_,_=self.fixture.request('GET','/api/frame-assist/jobs/'+'b'*32)
        self.assertEqual(status,404)
        status,_,_=self.fixture.request('GET','/api/human-objects/images/../../.secrets/private_key')
        self.assertEqual(status,404)


if __name__=='__main__':unittest.main()
