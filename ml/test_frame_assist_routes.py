"""Selected-frame UI routes: synthetic data, no external requests."""
import hashlib
import http.client
from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from PIL import Image
from annotator import make_handler
from test_human_routes import HumanRoutesTests


class FakeAssistant:
    def __init__(self,case):
        self.case=case;self.calls=0
        self.path=case.root/'data/human_review/synthetic-selected.jpg'
        Image.new('RGB',(40,60),'white').save(self.path)
        self.metadata=dict(path=self.path,width=40,height=60,image_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            timestamp_sec=3.0,requested_timestamp_sec=3,video=case.image_row['video'],group_id=case.image_row['group_id'],
            source_sha256=case.video_sha,split='train')
    def status(self):return {'configured':True,'model':'gpt-5.6-terra','busy':False}
    def start(self,payload):self.calls+=1;return {'job_id':'a'*32,'status':'queued'}
    def get(self,key):
        if key!='a'*32:raise KeyError(key)
        return {'job_id':key,'status':'completed','eligible_for_training':False}
    def frame_path(self,key):self.get(key);return self.path
    def frame_metadata(self,key):self.get(key);return self.metadata
    def prepare_frame(self,payload):return self.metadata


class FrameAssistRoutesTests(unittest.TestCase):
    def setUp(self):
        self.fixture=HumanRoutesTests();self.fixture.setUp()
        self.fixture.server.shutdown();self.fixture.server.server_close();self.fixture.thread.join()
        self.assistant=FakeAssistant(self.fixture)
        parent=make_handler(self.fixture.store,self.fixture.report/'previews',human=self.fixture.human,
            objects=self.fixture.objects,frame_assist=self.assistant)
        class Quiet(parent):
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Quiet)
        self.fixture.server=server
        self.fixture.thread=threading.Thread(target=lambda:server.serve_forever(poll_interval=.01),daemon=True)
        self.fixture.thread.start()
        self.fixture.origin='http://127.0.0.1:'+str(server.server_port)
    def tearDown(self):self.fixture.tearDown()

    def test_loading_status_does_not_start_paid_work(self):
        status,_,value=self.fixture.json_request('GET','/api/frame-assist/status')
        self.assertEqual(status,200);self.assertTrue(value['configured']);self.assertEqual(self.assistant.calls,0)
        status,_,_=self.fixture.request('GET','/label/frame-assist.js')
        self.assertEqual(status,200)

    def test_explicit_start_and_result_do_not_create_labels(self):
        status,_,value=self.fixture.json_request('POST','/api/frame-assist',{'task_id':'synthetic'})
        self.assertEqual(status,202);self.assertEqual(self.assistant.calls,1)
        status,_,value=self.fixture.json_request('GET','/api/frame-assist/jobs/'+'a'*32)
        self.assertEqual(status,200);self.assertFalse(value['eligible_for_training'])
        self.assertEqual(self.fixture.objects.decisions(),{})

    def test_selected_frame_registration_native_image_and_no_auto_boxes(self):
        status,_,value=self.fixture.json_request('POST','/api/frame-assist/frames',{'job_id':'a'*32})
        self.assertEqual(status,201);self.assertIn('assist_job=',value['object_url'])
        image_id=value['image_id'];self.assertEqual(len(self.fixture.objects.rows),2)
        status,headers,raw=self.fixture.request('GET','/api/human-objects/images/'+image_id+'.jpg')
        self.assertEqual(status,200);self.assertEqual(headers['Content-Type'],'image/jpeg')
        self.assertEqual(raw,self.assistant.path.read_bytes())
        self.assertEqual(self.assistant.calls,0);self.assertEqual(self.fixture.objects.decisions(),{})

    def test_malformed_requests_do_not_start_paid_work(self):
        for origin in ('','https://another.example','http://['):
            status,_,_=self.fixture.json_request('POST','/api/frame-assist',{},headers={'Origin':origin})
            self.assertEqual(status,403)
        for body in ('null','[]','{invalid'):
            status,_,_=self.fixture.json_request('POST','/api/frame-assist',body=body)
            self.assertEqual(status,422)
        self.assertEqual(self.assistant.calls,0)

    def test_unknown_job_and_paths_are_not_exposed(self):
        status,_,_=self.fixture.json_request('GET','/api/frame-assist/jobs/'+'b'*32)
        self.assertEqual(status,404)
        status,_,_=self.fixture.request('GET','/api/human-objects/images/../../.secrets/openai_api_key')
        self.assertEqual(status,404)


if __name__=='__main__':unittest.main()
