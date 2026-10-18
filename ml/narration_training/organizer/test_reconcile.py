import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .reconcile import check_record,validate_plan,focus_prompt,survey_prompt
from .resume import patched_source

class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'frame.jpg').write_bytes(b'synthetic frame fixture')
        self.ref=dict(id='f000',timestamp=1.,path='frame.jpg',sha256=hashlib.sha256(b'synthetic frame fixture').hexdigest())
        self.plan=dict(model_id='local/model',model_revision='pinned')
        self.video=dict(duration_sec=2.)
        self.record=dict(prompt='prompt',frames=[self.ref],model_id='local/model',model_revision='pinned',answer={'events':[]},raw_response='{"events":[]}',parse_error=None)
    def tearDown(self):self.temp.cleanup()
    def errors(self):return check_record(self.record,'prompt',[self.ref],self.root,self.video,self.plan)
    def test_exact_input_reusable(self):self.assertEqual(self.errors(),[])
    def test_changed_history_invalidates_answer(self):
        self.record['prompt']='different history';self.assertIn('prompt_or_prior_memory_mismatch',self.errors())
    def test_changed_frame_bytes_rejected(self):
        (self.root/'frame.jpg').write_bytes(b'different');self.assertIn('frame_bytes_mismatch',self.errors())
    def test_wrong_revision_rejected(self):
        self.record['model_revision']='different';self.assertIn('model_revision_mismatch',self.errors())
    def test_invalid_bounds_rejected(self):
        self.ref['timestamp']=3.;self.assertIn('timestamp_outside_video',self.errors())
    def test_raw_response_cannot_be_silently_repaired(self):
        self.record['raw_response']='{"events":[';self.assertIn('invalid_raw_response',self.errors())
    def test_parsed_response_cannot_be_edited(self):
        self.record['answer']={'events':['invented']};self.assertIn('raw_parsed_answer_mismatch',self.errors())
    def test_ablation_differs_only_in_memory(self):
        w=dict(start_sec=1.,end_sec=2.,context_start_sec=0.,context_end_sec=3.)
        with_history=focus_prompt(w,dict(last_known='unknown'))
        without=focus_prompt(w)
        self.assertEqual(with_history.replace('Relevant prior history (unconfirmed, current state unknown until reobserved): {"last_known": "unknown"}','No prior history is available.'),without)
        self.assertNotIn('organizer_claim',survey_prompt())
    def test_resume_patch_pinned_and_compiles(self):
        original=Path('ml/narration_training/organizer/analyze.py').read_text()
        patched=patched_source(original);compile(patched,'patched','exec')
        self.assertIn("cached.get('prompt')!=prompt",patched)
        self.assertIn('dtype=torch.float16',patched)
        self.assertIn("record['runtime_ref']",patched)
        with self.assertRaises(ValueError):patched_source(original+'\n')
    def test_manifest_aliases_and_cross_split_guard(self):
        folder=self.root/'data/training';folder.mkdir(parents=True)
        videos=self.root/'data/raw/open/videos';videos.mkdir(parents=True)
        for n in ['X-4.mov','X-47.mov']:(videos/n).write_bytes(b'original')
        sha=hashlib.sha256(b'original').hexdigest();path=folder/'manifest.csv'
        path.write_text(f'video,group_id,sha256,split\nvideos/X-4.mov,g1,{sha},val\n')
        v=dict(video='videos/X-4.mov',group_id='g1',sha256=sha,split='val',aliases=['videos/X-4.mov','videos/X-47.mov'],duration_sec=2.)
        p=dict(videos=[v],manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        validate_plan(self.root,p)
        path.write_text(path.read_text()+f'videos/X-47.mov,g1,{sha},train\n');p['manifest_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'leakage'):validate_plan(self.root,p)

if __name__=='__main__':unittest.main()
