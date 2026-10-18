import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from train_large import checked_rows

class LargeManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);base=self.root/'data/raw/open';base.mkdir(parents=True)
        self.rows=[]
        for i,(split,target) in enumerate([('train',0),('train',1),('val',0),('val',1)]):
            content=str(i).encode();(base/f'{i}.mov').write_bytes(content)
            self.rows.append(dict(video=f'{i}.mov',sha256=hashlib.sha256(content).hexdigest(),split=split,target=str(target),start_sec='0',end_sec='8'))
        self.path=self.root/'manifest.csv'
    def write(self):
        with self.path.open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(self.rows[0]));w.writeheader();w.writerows(self.rows)
    def test_unchanged_split_accepted(self):
        self.write();self.assertEqual(len(checked_rows(self.root,self.path)),4)
    def test_cross_split_duplicate_rejected(self):
        self.rows[2].update(video=self.rows[0]['video'],sha256=self.rows[0]['sha256']);self.write()
        with self.assertRaisesRegex(ValueError,'leakage'):checked_rows(self.root,self.path)
    def test_changed_source_rejected(self):
        self.write();(self.root/'data/raw/open/0.mov').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'hash'):checked_rows(self.root,self.path)
    def test_unsafe_path_and_bad_interval_rejected(self):
        self.rows[0]['video']='../../../escape';self.write()
        with self.assertRaises(ValueError):checked_rows(self.root,self.path)
        self.rows[0]['video']='0.mov';self.rows[0]['end_sec']='0';self.write()
        with self.assertRaises(ValueError):checked_rows(self.root,self.path)

if __name__=='__main__':unittest.main()
