import csv
import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from annotator import ReviewStore, make_handler
from prepare_training import prepare
from training_control import TrainingControl
from train_x3d import metrics


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_duplicate_videos_stay_together_and_review_is_not_modified(self):
        source=self.root/'data/raw/open/videos'
        source.mkdir(parents=True)
        review=self.root/'data/review/event_review.csv'
        review.parent.mkdir(parents=True)
        rows=[]
        for i in range(16):
            (source/f'{i}.mov').write_bytes(f'video {i}'.encode())
            rows.append(dict(video=f'videos/{i}.mov',source_verdict='нарушение' if i<8 else 'без нарушения',
                             anchor_sec=10 if i<8 else '',duration_sec=25))
        (source/'duplicate.mov').write_bytes((source/'0.mov').read_bytes())
        rows.append(dict(rows[0],video='videos/duplicate.mov',anchor_sec=20))
        with review.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        before=review.read_bytes()
        summary=prepare(self.root)
        self.assertEqual(summary['unique_videos'],16)
        self.assertEqual(review.read_bytes(),before)
        manifest=self.root/'data/training/manifest.csv'
        original=manifest.read_bytes()
        prepare(self.root)
        self.assertEqual(manifest.read_bytes(),original)
        with manifest.open() as f: clips=list(csv.DictReader(f))
        groups={}
        for row in clips:
            groups.setdefault(row['sha256'],set()).add(row['split'])
        self.assertTrue(all(len(s)==1 for s in groups.values()))
        self.assertEqual(len([r for r in clips if r['video']=='videos/0.mov']),2)

    def test_metrics_expose_false_positives_and_misses(self):
        result=metrics([1,1,0,0],[1,0,1,0])
        self.assertEqual(result['confusion'],dict(tp=1,tn=1,fp=1,fn=1))
        self.assertEqual(result['f1'],.5)
        self.assertEqual(result['balanced_accuracy'],.5)

    def test_start_requires_connection_and_blocks_second_run(self):
        control=TrainingControl(self.root)
        with self.assertRaises(ValueError):control.action('start')
        control.update(connected=True)
        with patch('training_control.threading.Thread') as thread:
            control.action('start')
            thread.return_value.start.assert_called_once()
            with self.assertRaises(FileExistsError):control.action('start')

    def test_active_split_prevents_legacy_runner_overwriting_frozen_manifest(self):
        directory=self.root/'data/training';directory.mkdir(parents=True)
        manifest=directory/'manifest.csv';manifest.write_text('frozen original')
        (directory/'active_split.json').write_text(json.dumps({'version':'split95_v1'}))
        control=TrainingControl(self.root);control.update(connected=True)
        with patch('training_control.threading.Thread') as thread:
            with self.assertRaisesRegex(ValueError,'95/5'):control.action('start')
            thread.assert_not_called()
        self.assertEqual(manifest.read_text(),'frozen original')
        self.assertFalse(control.snapshot()['legacy_start_supported'])

    def test_user_input_cannot_be_remote_shell_command(self):
        control=TrainingControl(self.root)
        for value in ['root; id','$(id)','-oProxyCommand=id','a@host','../x']:
            with self.assertRaises(ValueError):control.set_user(value)
        control.set_user('ubuntu')
        self.assertEqual(control.ssh()[-1],'ubuntu@global.prd.ga.run.brev.nvidia.com')

    def test_reload_recovers_remote_monitor_and_does_not_relaunch(self):
        control=TrainingControl(self.root)
        control.update(status='running',remote_launched=True,run_id='20261017-100000-abcdef',remote_root='/home/ubuntu/nuhack-training')
        with patch('training_control.threading.Thread') as thread:
            restored=TrainingControl(self.root)
            self.assertTrue(restored.busy)
            self.assertEqual(thread.call_args.kwargs['target'].__name__,'monitor')

    def test_stop_only_writes_marker_in_current_run(self):
        control=TrainingControl(self.root)
        control.update(remote_launched=True,remote_root='/home/ubuntu/nuhack-training',run_id='20261017-100000-abcdef')
        with patch.object(control,'remote') as remote:
            control.request_stop()
            self.assertEqual(remote.call_args.args[0],'touch /home/ubuntu/nuhack-training/runs/20261017-100000-abcdef/STOP')

    def test_failed_checkpoint_copy_does_not_claim_backup(self):
        control=TrainingControl(self.root)
        with patch('training_control.subprocess.run') as run:
            run.return_value.returncode=1
            with self.assertRaises(RuntimeError):control.sync_checkpoint('/remote','20261017-100000-abcdef',1)
        self.assertNotIn('synced_checkpoint_epoch',control.state)

    def test_training_http_rejects_foreign_origin_and_serves_snapshot(self):
        control=TrainingControl(self.root)
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(ReviewStore(self.root/'unused.csv'),self.root,training=control))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            c=http.client.HTTPConnection(*server.server_address)
            c.request('GET','/api/training/status');r=c.getresponse()
            self.assertEqual(r.status,200);self.assertEqual(json.loads(r.read())['status'],'awaiting_access')
            c.request('POST','/api/training/action',body='{"action":"start"}',headers={'Origin':'https://elsewhere.example'})
            r=c.getresponse();self.assertEqual(r.status,403);r.read()
            c.request('POST','/api/training/action',body='{"action":"check"}',
                      headers={'Host':'forte.example','Origin':'https://forte.example'})
            r=c.getresponse();self.assertEqual(r.status,403);r.read()
            c.request('GET','/api/training/artifact/../../control.json');r=c.getresponse()
            self.assertEqual(r.status,404);r.read();c.close()
        finally:
            server.shutdown();server.server_close();thread.join()


if __name__=='__main__': unittest.main()
