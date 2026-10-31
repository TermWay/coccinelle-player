"""ffmpeg-backed frame reading and writing, on a tiny generated video and image."""
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from fsr4 import media


def make_test_video(path: Path, frames: int = 8, width: int = 64, height: int = 48) -> None:
    command = [media.tool_path("ffmpeg"), "-v", "error", "-y", "-f", "lavfi",
               "-i", f"testsrc=size={width}x{height}:rate=10", "-frames:v", str(frames),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(command, check=True)


class MediaTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_probe_and_read_video(self):
        path = self.root / "test.mp4"
        make_test_video(path)
        info = media.probe(path)
        self.assertEqual((info.width, info.height), (64, 48))
        self.assertEqual(info.frame_rate, Fraction(10))
        self.assertEqual(info.frame_count, 8)
        self.assertFalse(info.has_audio)
        self.assertFalse(media.is_image(path))
        frames = [frame.copy() for frame in media.FrameReader(path, info)]
        self.assertEqual(len(frames), 8)
        self.assertEqual(frames[0].shape, (48, 64, 4))
        self.assertTrue(np.all(frames[0][..., 3] == 255))
        self.assertFalse(np.array_equal(frames[0], frames[7]), "testsrc frames should differ")

    def test_write_video_roundtrip(self):
        path = self.root / "out.mp4"
        writer = media.FrameWriter(path, 64, 48, Fraction(10), "x264")
        frame = np.zeros((48, 64, 4), dtype=np.uint8)
        frame[..., 3] = 255
        for index in range(5):
            frame[..., 0] = index * 40
            writer.write(frame)
        writer.close()
        info = media.probe(path)
        self.assertEqual(info.frame_count, 5)
        self.assertEqual((info.width, info.height), (64, 48))

    def test_image_roundtrip(self):
        path = self.root / "image.png"
        pixels = np.random.default_rng(1).integers(0, 256, size=(20, 30, 4), dtype=np.uint8)
        pixels[..., 3] = 255
        media.write_image(path, pixels)
        self.assertTrue(media.is_image(path))
        loaded = media.read_image(path)
        self.assertEqual(loaded.shape, (20, 30, 4))
        np.testing.assert_array_equal(loaded[..., :3], pixels[..., :3])

    def test_background_encoders_start(self):
        """Background pieces cap the encoder's threads; x265 must get them through its pool, not -threads."""
        frame = np.zeros((48, 64, 4), dtype=np.uint8)
        for encoder in ("x264", "x265"):
            path = self.root / f"piece_{encoder}.mp4"
            writer = media.FrameWriter(path, 64, 48, Fraction(10), encoder, gop=20, background=True)
            for _ in range(3):
                writer.write(frame)
            writer.close()
            self.assertEqual(media.probe(path).frame_count, 3, encoder)

    def test_writer_rejects_wrong_size(self):
        writer = media.FrameWriter(self.root / "bad.mp4", 64, 48, Fraction(10), "x264")
        with self.assertRaises(ValueError):
            writer.write(np.zeros((10, 10, 4), dtype=np.uint8))
        writer.abort()


if __name__ == "__main__":
    unittest.main()
