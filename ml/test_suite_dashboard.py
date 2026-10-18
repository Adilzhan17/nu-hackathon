import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from suite_dashboard import detailed_snapshot

class DetailedSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.folder=self.root/'data/autolabel/detailed_train';self.folder.mkdir(parents=True)
        self.plan=dict(candidates=[dict(candidate_id='c1',video='a.mov'),dict(candidate_id='c2',video='b.mov')],durations={'a':10,'b':20})
        self.path=self.folder/'plan.json';self.path.write_text(json.dumps(self.plan))
        self.digest=hashlib.sha256(self.path.read_bytes()).hexdigest()
    def test_plan_is_not_reported_as_running_or_finished(self):
        s=detailed_snapshot(self.root)
        self.assertEqual(s['stage'],'prepared');self.assertEqual(s['received_windows'],0);self.assertEqual(s['planned_windows'],2)
    def test_mismatching_or_incomplete_progress_is_not_counted(self):
        (self.folder/'inference_progress.json').write_text(json.dumps(dict(plan_sha256='old',completed_windows=2)))
        self.assertEqual(detailed_snapshot(self.root)['received_windows'],0)
        (self.folder/'inference_progress.json').write_text('{')
        self.assertEqual(detailed_snapshot(self.root)['stage'],'prepared')
    def test_partial_records_deduplicated_and_unrelated_records_ignored(self):
        records=self.folder/'inference_records';records.mkdir()
        for name,cid in [('one','c1'),('duplicate','c1'),('other','wrong')]:
            (records/(name+'.json')).write_text(json.dumps(dict(candidate_id=cid,proposals=[{}],inference_status='ok')))
        s=detailed_snapshot(self.root)
        self.assertEqual(s['received_windows'],1);self.assertEqual(s['proposals_in_local_records'],1);self.assertEqual(s['stage'],'partial')
    def test_completed_snapshot_needs_matching_plan(self):
        (self.folder/'inference_progress.json').write_text(json.dumps(dict(plan_sha256=self.digest,completed_windows=2,completed=True)))
        s=detailed_snapshot(self.root);self.assertEqual(s['stage'],'completed');self.assertEqual(s['local_records'],0)

if __name__=='__main__':unittest.main()
