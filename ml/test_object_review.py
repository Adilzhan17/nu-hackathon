import concurrent.futures
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from PIL import Image
from autolabel.dataset import INPUTS, snapshot
from object_review import ObjectReview, ObjectConflict


class ObjectReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        for name in INPUTS:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('')
        (self.root/INPUTS[0]).write_text('video,group_id,sha256,split\nsynthetic.mov,video_1234567890abcdef,'+'a'*64+',train\n')
        source=self.root/'data/autolabel/objects_v1';(source/'images').mkdir(parents=True)
        image=source/'images/frame_0000.jpg';Image.new('RGB',(40,60),(80,100,120)).save(image)
        self.row=dict(image_id='frame_0000',image='images/frame_0000.jpg',image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            video='synthetic.mov',group_id='video_1234567890abcdef',source_sha256='a'*64,split='train',timestamp=2,width=40,height=60)
        (source/'object_review_manifest.jsonl').write_text(json.dumps(self.row)+'\n')
        (source/'summary.json').write_text(json.dumps({'protected_inputs':snapshot(self.root)}))
        self.store=ObjectReview(self.root)

    def tearDown(self):self.temp.cleanup()

    def payload(self,**changes):
        value=dict(image_id=self.row['image_id'],image_sha256=self.row['image_sha256'],expected_version=0,
            reviewer='synthetic-test',review_status='complete',image_assessment='targets_annotated',note='',
            boxes=[{'class':'cash','visibility':'clear','xyxy':[3,5,15,20]}])
        value.update(changes);return value

    def test_durable_human_decision_and_coco_coordinates(self):
        self.store.save(self.payload())
        reopened=ObjectReview(self.root)
        self.assertEqual(reopened.catalog()['counts']['human_boxes'],1)
        result=reopened.export_dataset();out=self.root/result['path']
        coco=json.loads((out/'coco_train.json').read_text())
        self.assertEqual(coco['annotations'][0]['bbox'],[3,5,12,15])
        self.assertTrue((out/'images/frame_0000.jpg').is_file())
        with zipfile.ZipFile(out/'dataset.zip') as archive:
            self.assertIn('coco_train.json',archive.namelist())
            self.assertIn('images/frame_0000.jpg',archive.namelist())
        self.assertFalse(any(p.name.endswith('.partial') for p in out.parent.iterdir()))

    def test_uncertain_frame_never_becomes_detector_negative(self):
        self.store.save(self.payload(review_status='needs_context',image_assessment='uncertain',boxes=[]))
        self.assertEqual(self.store.catalog()['counts']['training_frames'],0)
        result=self.store.export_dataset()
        self.assertEqual(result['training_frames'],0)
        self.assertEqual(json.loads((self.root/result['path']/'coco_train.json').read_text())['images'],[])

    def test_explicit_checked_negative_is_eligible(self):
        self.store.save(self.payload(image_assessment='confirmed_no_target',boxes=[]))
        result=self.store.export_dataset()
        self.assertEqual(result['training_frames'],1)
        self.assertEqual(result['human_boxes'],0)
        self.assertEqual((self.root/result['path']/'labels/frame_0000.txt').read_text(),'')

    def test_concurrent_review_does_not_overwrite_other_person(self):
        def write(i):
            try:self.store.save(self.payload(reviewer=str(i)));return 'saved'
            except ObjectConflict:return 'conflict'
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            self.assertCountEqual(list(pool.map(write,[1,2])),['saved','conflict'])
        self.assertEqual(self.store.decisions()['frame_0000']['version'],1)

    def test_bad_geometry_and_provenance_rejected(self):
        for change in [dict(image_sha256='b'*64),dict(boxes=[{'class':'key','visibility':'clear','xyxy':[0,0,90,30]}]),
                       dict(boxes=[{'class':'key','visibility':'clear','xyxy':[0,0,float('nan'),30]}]),
                       dict(image_assessment='confirmed_no_target')]:
            with self.assertRaises(ValueError):self.store.save(self.payload(**change))
        self.assertEqual(self.store.decisions(),{})

    def test_changed_source_blocks_saving(self):
        (self.root/INPUTS[1]).write_text('changed')
        with self.assertRaises(ObjectConflict):self.store.save(self.payload())

    def test_replaced_frame_does_not_reuse_old_boxes_after_restart(self):
        self.store.save(self.payload())
        source=self.root/'data/autolabel/objects_v1'
        Image.new('RGB',(40,60),(0,0,0)).save(source/self.row['image'])
        changed=dict(self.row,image_sha256=hashlib.sha256((source/self.row['image']).read_bytes()).hexdigest())
        (source/'object_review_manifest.jsonl').write_text(json.dumps(changed)+'\n')
        with self.assertRaises(ObjectConflict):self.store.catalog()
        reopened=ObjectReview(self.root)
        self.assertTrue(reopened.decisions()['frame_0000']['stale'])
        self.assertEqual(reopened.export_dataset()['training_frames'],0)

    def frame_metadata(self):
        path=self.store.directory/'synthetic-frame.jpg'
        Image.new('RGB',(40,60),(0,100,150)).save(path)
        return dict(path=path,width=40,height=60,image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            timestamp_sec=3.25,requested_timestamp_sec=3.23,video=self.row['video'],group_id=self.row['group_id'],
            source_sha256=self.row['source_sha256'],split='train')

    def test_selected_frame_persists_without_creating_labels(self):
        metadata=self.frame_metadata()
        result=self.store.register_frame(metadata)
        self.assertEqual(self.store.register_frame(metadata)['image_id'],result['image_id'])
        reopened=ObjectReview(self.root)
        self.assertEqual(len(reopened.rows),2)
        self.assertEqual(reopened.decisions(),{})
        row=reopened.rows[result['image_id']]
        self.assertTrue(reopened.image_path(row).is_file())
        reopened.save(self.payload(image_id=row['image_id'],image_sha256=row['image_sha256']))
        export=reopened.export_dataset()
        self.assertEqual(export['human_boxes'],1)
        self.assertEqual(export['training_frames'],1)

    def test_registration_rejects_wrong_split_hash_and_path(self):
        metadata=self.frame_metadata()
        for change in [dict(split='val'),dict(source_sha256='f'*64),dict(image_sha256='e'*64),
                       dict(timestamp_sec=float('nan')),dict(path=self.root/'data/autolabel/objects_v1/images/frame_0000.jpg')]:
            with self.assertRaises(ValueError):self.store.register_frame(dict(metadata,**change))
        self.assertEqual(len(self.store.rows),1)


if __name__=='__main__':unittest.main()
