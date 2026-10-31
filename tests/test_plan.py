"""Job planning: output sizes, the 1x-3x range and the GPU encoders' frame size limits (no GPU needed)."""
import unittest
from fractions import Fraction
from pathlib import Path

from fsr4.job import JobOptions, plan_from_info
from fsr4.media import MediaInfo


def info(width: int, height: int) -> MediaInfo:
    return MediaInfo(width=width, height=height, frame_rate=Fraction(30), frame_count=10, duration=1.0, codec="h264")


class PlanTest(unittest.TestCase):
    def test_sizes_are_even_and_scaled(self):
        plan = plan_from_info(JobOptions(input=Path("a.mp4"), scale=1.5), info(1281, 721))
        self.assertEqual(plan.output_size, (1922, 1082))
        self.assertEqual(plan.output, Path("a.fsr4.mp4"))

    def test_scale_range(self):
        with self.assertRaises(ValueError):
            plan_from_info(JobOptions(input=Path("a.mp4"), scale=3.5), info(640, 360))

    def test_gpu_encoder_limits(self):
        source = info(2880, 2160)
        with self.assertRaises(ValueError) as raised:
            plan_from_info(JobOptions(input=Path("a.mp4"), scale=2.0, codec="h264_amf"), source)
        self.assertIn("4096", str(raised.exception))
        self.assertEqual(plan_from_info(JobOptions(input=Path("a.mp4"), scale=2.0, codec="amf"), source).output_size,
                         (5760, 4320))
        with self.assertRaises(ValueError):
            plan_from_info(JobOptions(input=Path("a.mp4"), scale=3.0, codec="av1_amf"), source)

    def test_images_ignore_video_encoders(self):
        plan = plan_from_info(JobOptions(input=Path("a.png"), scale=2.0, codec="h264_amf"), info(2880, 2160))
        self.assertTrue(plan.image_mode)


if __name__ == "__main__":
    unittest.main()
