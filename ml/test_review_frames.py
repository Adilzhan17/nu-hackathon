"""Manual frame extraction: preserve image lineage without any model calls."""
from fractions import Fraction
import hashlib
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from review_frames import ReviewFrames, extract_frame, AssistError


class ReviewFramesTests(unittest.TestCase):
    def test_native_frame_pts_and_server_resolved_input(self):
        import av
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'data/raw/open/videos/synthetic.mov'
            path.parent.mkdir(parents=True)
            with av.open(str(path), 'w') as output:
                stream = output.add_stream('mpeg4', rate=10)
                stream.width, stream.height = 64, 48
                stream.pix_fmt = 'yuv420p'
                stream.time_base = Fraction(1, 10)
                for i in range(20):
                    frame = av.VideoFrame.from_image(Image.new('RGB', (64, 48), (i * 8, 40, 70)))
                    frame.pts, frame.time_base = 90 + i, Fraction(1, 10)
                    for packet in stream.encode(frame): output.mux(packet)
                for packet in stream.encode(): output.mux(packet)
            image, timestamp = extract_frame(path, .76)
            self.assertEqual(image.size, (64, 48))
            self.assertAlmostEqual(timestamp, .8, places=3)
            class Human:
                def catalog(self):
                    return dict(tasks=[dict(id='task', source_fingerprint='v1', video='videos/synthetic.mov',
                                            video_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                            group_id='g', duration_sec=2)])
            service = ReviewFrames(root, Human(), None)
            payload = dict(task_id='task', source_fingerprint='v1', timestamp_sec=.76)
            metadata = service.prepare_frame(payload)
            self.assertEqual(metadata['width'], 64)
            self.assertTrue(metadata['path'].is_relative_to(root / 'data/human_review'))
            self.assertAlmostEqual(metadata['timestamp_sec'], .8)
            with self.assertRaises(AssistError):
                service.prepare_frame(dict(payload, url='https://example.invalid/frame.jpg'))
            with self.assertRaises(AssistError):
                service.prepare_frame(dict(payload, source_fingerprint='stale'))


if __name__ == '__main__':
    unittest.main()
