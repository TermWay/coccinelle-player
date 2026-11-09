"""Job planning: output sizes, the 1x-3x range, the GPU encoders' frame size limits and the output path (no GPU
needed)."""
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from fsr4.job import JobOptions, plan_from_info
from fsr4.media import MediaInfo


def info(width: int, height: int, pixel_aspect: Fraction = Fraction(1)) -> MediaInfo:
    return MediaInfo(width=width, height=height, frame_rate=Fraction(30), frame_count=10, duration=1.0, codec="h264",
                     pixel_aspect=pixel_aspect)


class PlanTest(unittest.TestCase):
    def test_sizes_are_even_and_scaled(self):
        plan = plan_from_info(JobOptions(input=Path("a.mp4"), scale=1.5), info(1281, 721))
        self.assertEqual(plan.output_size, (1922, 1082))
        self.assertEqual(plan.output, Path("a.fsr4.mp4"))

    def test_scale_range(self):
        with self.assertRaises(ValueError):
            plan_from_info(JobOptions(input=Path("a.mp4"), scale=3.5), info(640, 360))

    def test_sharpening_range(self):
        for sharpness in (None, 0.0, 0.4, 1.0):
            plan_from_info(JobOptions(input=Path("a.mp4"), sharpness=sharpness), info(640, 360))
        for sharpness in (-0.1, 1.5, float("nan")):
            with self.subTest(sharpness=sharpness), self.assertRaises(ValueError):
                plan_from_info(JobOptions(input=Path("a.mp4"), sharpness=sharpness), info(640, 360))

    def test_gpu_encoder_limits(self):
        source = info(2880, 2160)
        with self.assertRaises(ValueError) as raised:
            plan_from_info(JobOptions(input=Path("a.mp4"), scale=2.0, codec="h264_amf"), source)
        self.assertIn("4096", str(raised.exception))
        self.assertEqual(plan_from_info(JobOptions(input=Path("a.mp4"), scale=2.0, codec="amf"), source).output_size,
                         (5760, 4320))
        with self.assertRaises(ValueError):
            plan_from_info(JobOptions(input=Path("a.mp4"), scale=3.0, codec="av1_amf"), source)

    def test_odd_sizes_stay_inside_the_scale_range(self):
        """Rounding an odd size to even must not ask for a hair more than 3x or less than 1x."""
        self.assertEqual(plan_from_info(JobOptions(input=Path("a.mp4"), scale=3.0), info(853, 480)).output_size,
                         (2558, 1440))
        self.assertEqual(plan_from_info(JobOptions(input=Path("a.mp4"), mode="native"), info(853, 481)).output_size,
                         (854, 482))

    def test_anamorphic_sources_come_out_at_their_shape(self):
        """A 16:9 DVD stores 720 × 480 pixels that players widen: the output has square pixels at 16:9."""
        dvd = info(720, 480, Fraction(32, 27))

        def size(source: MediaInfo = dvd, **options) -> tuple[int, int]:
            return plan_from_info(JobOptions(input=Path("a.mpg"), **options), source).output_size

        self.assertEqual(size(scale=2.0), (1706, 960))
        self.assertEqual(size(scale=2.25), (1920, 1080))
        self.assertEqual(size(height=1080), (1920, 1080))
        self.assertEqual(size(width=1920), (1920, 1080))
        # 3x would widen it 3.56 times: both sides shrink together to stay inside FSR's range.
        self.assertEqual(size(scale=3.0), (2160, 1216))
        # A 4:3 DVD's pixels are narrower: at 1x its width would shrink, so both sides grow together.
        self.assertEqual(size(info(720, 480, Fraction(8, 9)), mode="native"), (720, 540))
        plan = plan_from_info(JobOptions(input=Path("a.mpg"), scale=2.0), dvd)
        self.assertEqual(plan.describe_scale(), "2.37x by 2.00x")

    def test_av1_widths_are_multiples_of_8(self):
        """AV1 on the GPU pads other widths and tags the padded one, so the file would read back sheared."""
        plan = plan_from_info(JobOptions(input=Path("a.mp4"), scale=1.5, codec="av1_amf"), info(1281, 721))
        self.assertEqual(plan.output_size, (1920, 1082))
        plan = plan_from_info(JobOptions(input=Path("a.mp4"), width=1922, codec="av1_amf"), info(1281, 721))
        self.assertEqual(plan.output_size[0] % 8, 0)

    def test_output_never_overwrites_the_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "clip.mp4"
            source.write_bytes(b"source")
            for output in (source, Path(directory) / "." / "CLIP.mp4"):
                with self.assertRaises(ValueError):
                    plan_from_info(JobOptions(input=source, output=output), info(640, 360))
            # The output is written under its partial name until complete: that must not be the input either.
            partial_source = Path(directory) / "clip.partial.mp4"
            partial_source.write_bytes(b"source")
            with self.assertRaises(ValueError):
                plan_from_info(JobOptions(input=partial_source, output=source), info(640, 360))
            self.assertEqual(source.read_bytes(), b"source")

    def test_images_ignore_video_encoders(self):
        plan = plan_from_info(JobOptions(input=Path("a.png"), scale=2.0, codec="h264_amf"), info(2880, 2160))
        self.assertTrue(plan.image_mode)


if __name__ == "__main__":
    unittest.main()
