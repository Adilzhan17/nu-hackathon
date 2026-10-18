"""Decode synthetic MOVs to verify clip-relative sampling with nonzero PTS."""

import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch

from model_suite import load_frames


def make_mov(path, origin_sec):
    # Encode identical changing frames with different container time origins.
    # MPEG-4 is provided by the project's PyAV build; no external ffmpeg needed.
    with av.open(str(path), "w", format="mov") as container:
        stream = container.add_stream("mpeg4", rate=2)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 2)
        stream.codec_context.max_b_frames = 0
        for i in range(8):
            array = np.full((32, 32, 3), [i * 25, 20, 30], dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts = int(origin_sec * 2) + i
            frame.time_base = Fraction(1, 2)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class FrameTimestampTests(unittest.TestCase):
    def test_nonzero_container_origin_does_not_change_relative_frame_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            zero = Path(directory) / "zero.mov"
            shifted = Path(directory) / "shifted.mov"
            make_mov(zero, 0)
            make_mov(shifted, 5)
            with av.open(str(shifted)) as container:
                stream = container.streams.video[0]
                self.assertEqual(float(stream.start_time * stream.time_base), 5.0)
            # These windows need different frames and different seek positions.
            # Comparing to a zero-origin encode catches both origin mistakes:
            # seeking in relative PTS and comparing absolute decoded timestamps.
            for start, end in [(1.0, 2.5), (2.5, 3.5)]:
                with self.subTest(start=start, end=end):
                    expected = load_frames(zero, start, end, frames=2, size=32)
                    actual = load_frames(shifted, start, end, frames=2, size=32)
                    self.assertEqual(tuple(actual.shape), (3, 2, 32, 32))
                    self.assertTrue(torch.allclose(actual, expected, atol=1 / 255, rtol=0))
                    self.assertGreater(float(actual[0, 1].mean() - actual[0, 0].mean()), 0.1)
                    self.assertGreater(float(actual[0, 0].mean()), 0.1)


if __name__ == "__main__":
    unittest.main()
