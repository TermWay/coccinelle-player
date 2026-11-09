"""A whole image job on the GPU: the image is upscaled and written; cancelled, nothing is written."""
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

import needs
from fsr4.job import JobOptions, run_job
from fsr4.media import read_image, write_image


@needs.ffmpeg
class ImageJobTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        needs.require_fsr()

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.source = Path(directory.name) / "in.png"
        pixels = np.random.default_rng(1).integers(0, 256, size=(144, 256, 4), dtype=np.uint8)
        pixels[..., 3] = 255
        write_image(self.source, pixels)

    def test_an_image_is_upscaled_and_written(self):
        result = run_job(JobOptions(input=self.source, scale=2.0, verbose=False))
        self.assertEqual((result.frames, result.cancelled), (1, False))
        self.assertEqual(read_image(result.output).shape, (288, 512, 4))

    def test_a_cancelled_image_is_not_written(self):
        cancel = threading.Event()
        cancel.set()
        result = run_job(JobOptions(input=self.source, scale=2.0, verbose=False), cancel=cancel)
        self.assertEqual((result.frames, result.cancelled), (0, True))
        self.assertFalse(result.output.exists())
        self.assertEqual(sorted(path.name for path in self.source.parent.iterdir()), ["in.png"])


if __name__ == "__main__":
    unittest.main()
