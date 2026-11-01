"""ffmpeg-backed frame reading and writing, on a tiny generated video and image."""
import os
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

import needs
from fsr4 import media


def make_test_video(path: Path, frames: int = 8, width: int = 64, height: int = 48) -> None:
    command = [media.tool_path("ffmpeg"), "-v", "error", "-y", "-f", "lavfi",
               "-i", f"testsrc=size={width}x{height}:rate=10", "-frames:v", str(frames),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(command, check=True)


@needs.ffmpeg
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
        self.assertEqual(info.pixel_aspect, 1, "the output has square pixels")

    def test_anamorphic_video(self):
        """A 16:9 DVD: 720 × 480 pixels tagged 32:27 wide, which players show 853 wide."""
        path = self.root / "dvd.mpg"
        subprocess.run([media.tool_path("ffmpeg"), "-v", "error", "-y", "-f", "lavfi",
                        "-i", "testsrc=size=720x480:rate=25:duration=0.4", "-vf", "setsar=32/27",
                        "-c:v", "mpeg2video", str(path)], check=True)
        info = media.probe(path)
        self.assertEqual(info.pixel_aspect, Fraction(32, 27))
        self.assertAlmostEqual(info.display_width, 853.33, places=2)

        def pixel_aspect(text: str, name: str = "a.mkv", rotation: int | None = None) -> Fraction:
            stream = {"index": 0, "codec_type": "video", "codec_name": "mpeg2video", "width": 720, "height": 480,
                      "r_frame_rate": "25/1", "avg_frame_rate": "25/1", "sample_aspect_ratio": text}
            if rotation is not None:
                stream["side_data_list"] = [{"rotation": rotation}]
            return media.parse_probe({"streams": [stream], "format": {"duration": "1.0"}}, Path(name)).pixel_aspect

        self.assertEqual(pixel_aspect("0:1"), 1, "unknown")
        self.assertEqual(pixel_aspect("N/A"), 1)
        self.assertEqual(pixel_aspect("1:100"), 1, "a broken tag is ignored")
        self.assertEqual(pixel_aspect("32:27", rotation=-90), Fraction(27, 32), "turned upright, the shape turns")
        self.assertEqual(pixel_aspect("32:27", name="a.png"), 1, "image viewers ignore it")

    def test_image_roundtrip(self):
        path = self.root / "image.png"
        pixels = np.random.default_rng(1).integers(0, 256, size=(20, 30, 4), dtype=np.uint8)
        pixels[..., 3] = 255
        media.write_image(path, pixels)
        self.assertTrue(media.is_image(path))
        loaded = media.read_image(path)
        self.assertEqual(loaded.shape, (20, 30, 4))
        np.testing.assert_array_equal(loaded[..., :3], pixels[..., :3])

    def test_writer_replaces_the_output_only_when_complete(self):
        path = self.root / "out.mp4"
        path.write_bytes(b"previous result")
        writer = media.FrameWriter(path, 64, 48, Fraction(10), "x264")
        writer.write(np.zeros((48, 64, 4), dtype=np.uint8))
        self.assertEqual(path.read_bytes(), b"previous result")
        writer.close()
        self.assertEqual(media.probe(path).frame_count, 1)
        self.assertFalse(media.partial_path(path).exists())
        writer = media.FrameWriter(path, 64, 48, Fraction(10), "x264")
        writer.write(np.zeros((48, 64, 4), dtype=np.uint8))
        writer.abort()
        self.assertEqual(media.probe(path).frame_count, 1, "an aborted write must leave the previous file")
        self.assertFalse(media.partial_path(path).exists())

    def test_image_keeps_its_transparency(self):
        source = self.root / "source.png"
        pixels = np.zeros((24, 32, 4), dtype=np.uint8)
        pixels[:, :16] = (255, 0, 0, 255)
        pixels[:, 16:] = (0, 255, 0, 128)
        pixels[16:, 24:] = 0
        subprocess.run([media.tool_path("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
                        "-s", "32x24", "-i", "-", "-frames:v", "1", "-update", "1", str(source)],
                       input=pixels.tobytes(), check=True)
        upscaled = np.repeat(np.repeat(pixels, 2, axis=0), 2, axis=1)
        upscaled[..., 3] = 255
        for suffix in (".png", ".webp", ".jpg"):
            output = self.root / f"upscaled{suffix}"
            media.write_image(output, upscaled, alpha_source=source)
            loaded = media.read_image(output)
            self.assertEqual(loaded.shape, (48, 64, 4))
            if suffix == ".jpg":
                self.assertEqual(int(loaded[-1, -1, 3]), 255, "JPEG has no transparency")
                continue
            self.assertEqual(int(loaded[-1, -1, 3]), 0, suffix)
            self.assertEqual(int(loaded[2, 2, 3]), 255, suffix)
            self.assertAlmostEqual(int(loaded[2, 40, 3]), 128, delta=2, msg=suffix)

    def test_awkward_file_names(self):
        """A relative name starting with '-' is not an ffmpeg option, and '%' is not an image pattern."""
        make_test_video(self.root / "-dash.mp4")
        previous = os.getcwd()
        os.chdir(self.root)
        try:
            info = media.probe(Path("-dash.mp4"))
            self.assertEqual(len([frame for frame in media.FrameReader(Path("-dash.mp4"), info)]), 8)
            pixels = np.full((20, 30, 4), 200, dtype=np.uint8)
            media.write_image(Path("-frame%d.png"), pixels)
            self.assertTrue((self.root / "-frame%d.png").is_file())
        finally:
            os.chdir(previous)

    def test_a_video_that_ends_before_its_estimated_length(self):
        """2 s of video and 3 s of sound in Matroska (no frame count): the probe's estimate counts the sound."""
        path = self.root / "tail.mkv"
        subprocess.run([media.tool_path("ffmpeg"), "-v", "error", "-y",
                        "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=2",
                        "-f", "lavfi", "-i", "sine=duration=3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", str(path)], check=True)
        info = media.probe(path)
        frames = sum(1 for _ in media.FrameReader(path, info))
        self.assertGreater(info.frame_count, frames + 5)
        with self.assertRaises(media.PastEnd):
            media.read_frame(path, info, info.frame_count - 1)
        self.assertEqual(media.last_frame_before(path, info, info.frame_count - 1), frames - 1)
        with mock.patch.object(media, "LAST_FRAME_SEARCH_SECONDS", 0.2):  # several steps back, then the start
            self.assertEqual(media.last_frame_before(path, info, info.frame_count - 1), frames - 1)
        self.assertEqual(media.read_frame(path, info, frames - 1).shape, (48, 64, 4))

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
