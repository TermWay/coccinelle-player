"""AMD's HQ scaler: its settings, what it refuses, and on the machine pictures, jobs and the renderer's pieces."""
import queue
import subprocess
import tempfile
import time
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

import needs
from fsr4 import hq_scaler, media
from fsr4.hq_scaler import HQ_SCALER, EncoderScale, HqPipeline, filter_sharpness, upscale_picture
from fsr4.job import JobOptions, plan_from_info, run_job
from fsr4.pipeline import Timings
from fsr4.presets import BUILTIN, preset_from_json
from fsr4.render import RenderSettings, Renderer

WIDTH, HEIGHT, RATE, SECONDS = 320, 180, 30, 3
TIMEOUT = 60


def info(width: int, height: int) -> media.MediaInfo:
    return media.MediaInfo(width, height, Fraction(30), 30, 1.0, "h264")


def psnr(first: np.ndarray, second: np.ndarray) -> float:
    difference = first[..., :3].astype(np.float32) - second[..., :3].astype(np.float32)
    return 10 * float(np.log10(255.0 ** 2 / np.mean(difference * difference)))


def detail(picture: np.ndarray) -> float:
    """How much the picture changes from pixel to pixel."""
    return float(np.mean(np.abs(np.diff(picture[..., 1].astype(np.float32), axis=1))))


class SettingsTest(unittest.TestCase):
    def test_sharpening_only_adds_to_the_neutral_setting(self):
        self.assertEqual(filter_sharpness(None), hq_scaler.NEUTRAL_SHARPNESS)
        self.assertEqual(filter_sharpness(0.0), hq_scaler.NEUTRAL_SHARPNESS)
        self.assertEqual(filter_sharpness(1.0), 0.0)
        self.assertLess(filter_sharpness(0.6), filter_sharpness(0.3))

    def test_the_filter_names_the_output_size_and_comes_back_as_rgba(self):
        text = EncoderScale((640, 360), 1.0).video_filter(1280, 720)
        self.assertIn("sr_amf=w=1280:h=720:algorithm=sr1-1:sharpness=0.000", text)
        self.assertTrue(text.startswith("hwupload,") and text.endswith("hwdownload,format=rgba"), text)

    def test_the_encoder_gets_a_device_of_its_own(self):
        # ffmpeg hands an encoder the device of its type only when there is exactly one: two keep it off the scaler's.
        self.assertEqual(sum(argument.startswith("amf=") for argument in hq_scaler.DEVICE_ARGUMENTS), 2)

    def test_the_pipeline_needs_the_filter(self):
        with mock.patch.object(hq_scaler, "available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "ffmpeg 8.0"):
                HqPipeline((640, 360), (1280, 720))

    def test_frames_go_through_unchanged_and_are_counted(self):
        with mock.patch.object(hq_scaler, "available", return_value=True):
            pipeline = HqPipeline((64, 36), (128, 72), 0.5)
        frame = np.zeros((36, 64, 4), np.uint8)
        self.assertIs(pipeline.process(frame), frame)
        self.assertEqual(pipeline.encoder_scale, EncoderScale((64, 36), 0.5))
        pipeline.timings.add("flow (worker thread)", 0.001)
        self.assertEqual(pipeline.timings.summary(), "1 frames")

    def test_timings_of_stages_on_the_main_thread_are_unchanged(self):
        timings = Timings()
        timings.frames = 2
        timings.add("gpu", 0.01)
        self.assertIn("5.0 ms per frame on the main thread (200.0 fps)", timings.summary())


class PlanTest(unittest.TestCase):
    def options(self, **values) -> JobOptions:
        return JobOptions(input=Path("clip.mp4"), output=Path("out.mp4"), upscaler=HQ_SCALER, **values)

    def test_a_picture_too_small_is_refused_on_either_side(self):
        for size in ((31, 720), (1280, 31)):
            with self.assertRaisesRegex(ValueError, "at least 32"):
                plan_from_info(self.options(), info(*size))
        self.assertEqual(plan_from_info(self.options(codec="x264"), info(32, 32)).output_size, (64, 64))

    def test_the_options_that_belong_to_fsr_are_refused(self):
        for values in ({"debug_view": True}, {"debug_checking": True}, {"validate": True}, {"gpu": 1}):
            with self.assertRaisesRegex(ValueError, "do not apply"):
                plan_from_info(self.options(**values), info(640, 360))


class PresetTest(unittest.TestCase):
    def test_the_builtin_preset_is_named_without_motion(self):
        preset = next(preset for preset in BUILTIN if preset.upscaler == HQ_SCALER)
        settings = preset.applied_to(RenderSettings((1280, 720)))
        self.assertEqual((settings.flow, settings.flow_preset), ("zero", "fast"))
        self.assertEqual(settings.describe(), "AMD HQ scaler")
        self.assertEqual(RenderSettings((1, 1), flow="zero", sharpness=0.5, upscaler=HQ_SCALER).describe(),
                         "AMD HQ scaler · sharpening 0.50")

    def test_a_saved_preset_never_holds_motion(self):
        preset = preset_from_json({"name": "Mine", "upscaler": HQ_SCALER, "flow": "dis", "flow_preset": "medium"})
        self.assertEqual((preset.flow, preset.flow_preset), ("zero", "fast"))


@needs.ffmpeg
@needs.opencv
class ScalerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        needs.require_hq_scaler()
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.source = cls.root / "source.mp4"
        subprocess.run([media.tool_path("ffmpeg"), "-v", "error", "-y",
                        "-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate={RATE}",
                        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", str(SECONDS),
                        "-c:v", "libx264", "-g", "45", "-pix_fmt", "yuv420p", "-c:a", "aac", str(cls.source)], check=True)
        cls.info = media.probe(cls.source)
        cls.frames = [frame.copy() for frame in media.FrameReader(cls.source, cls.info)]
        cls.output_size = (WIDTH * 2, HEIGHT * 2)
        cls.single = cls.root / "single.mp4"
        run_job(JobOptions(input=cls.source, output=cls.single, scale=2.0, upscaler=HQ_SCALER, verbose=False))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def decoded(self, path: Path) -> list[np.ndarray]:
        return [frame[..., :3].astype(np.int16) for frame in media.FrameReader(path, media.probe(path))]

    def reference(self, frame: np.ndarray) -> np.ndarray:
        """`frame` at the output size by plain resampling."""
        import cv2
        return cv2.resize(np.ascontiguousarray(frame), self.output_size, interpolation=cv2.INTER_CUBIC)

    def test_a_picture_comes_back_opaque_at_the_size_asked_for(self):
        picture = upscale_picture(self.frames[0], (500, 400), None)  # each side by a scale of its own
        self.assertEqual(picture.shape, (400, 500, 4))
        self.assertTrue((picture[..., 3] == 255).all())

    def test_a_picture_keeps_its_colours(self):
        picture = upscale_picture(self.frames[0], self.output_size, None)
        self.assertGreater(psnr(picture, self.reference(self.frames[0])), 26.0)
        means = picture[..., :3].mean(axis=(0, 1)) - self.frames[0][..., :3].mean(axis=(0, 1))
        self.assertLess(float(np.abs(means).max()), 1.0)

    def test_sharpening_adds_detail(self):
        plain, sharp = (detail(upscale_picture(self.frames[0], self.output_size, value)) for value in (None, 1.0))
        self.assertGreater(sharp, plain * 1.1)

    def test_a_job_writes_every_frame_upscaled_with_its_sound(self):
        info = media.probe(self.single)
        self.assertEqual((info.width, info.height), self.output_size)
        self.assertTrue(info.audio)
        frames = self.decoded(self.single)
        self.assertEqual(len(frames), len(self.frames))
        for index in (0, len(frames) // 2, len(frames) - 1):
            self.assertGreater(psnr(frames[index], self.reference(self.frames[index])), 24.0, f"frame {index}")

    def test_an_image_job_writes_the_picture(self):
        source = self.root / "still.png"
        media.write_image(source, self.frames[0])
        output = self.root / "still_up.png"
        result = run_job(JobOptions(input=source, output=output, scale=2.0, upscaler=HQ_SCALER, verbose=False))
        self.assertEqual(result.frames, 1)
        self.assertEqual(media.read_image(output).shape[:2], self.output_size[::-1])

    def test_every_gpu_encoder_finishes_beside_the_scaler(self):
        # On one shared device these hang in about half the runs: a few runs of each must all finish.
        for encoder in sorted(media.GPU_ENCODERS):
            for run in range(3):
                output = self.root / f"{encoder}_{run}.mp4"
                writer = media.FrameWriter(output, *self.output_size, self.info.frame_rate, encoder,
                                           scale=EncoderScale((WIDTH, HEIGHT)))
                for frame in self.frames:
                    writer.write(frame)
                writer.close()
                self.assertEqual(media.probe(output).width, self.output_size[0])

    def test_pieces_around_a_seek_join_into_the_single_pass(self):
        events: queue.Queue = queue.Queue()
        settings = RenderSettings(self.output_size, flow="zero", upscaler=HQ_SCALER)
        renderer = Renderer(self.source, self.info, settings, Path("unused"),
                            lambda kind, *values: events.put((kind, values)))
        try:
            renderer.want_live = True
            timeline = renderer.timeline
            renderer.set_target(50)
            self.wait_for(events, lambda: timeline.ready_until(50) >= 60, "frames after the seek")
            renderer.set_batch(True)
            self.wait_for(events, timeline.complete, "the whole video")
            output = self.root / "joined.mp4"
            self.assertEqual(renderer.export(output), len(self.frames))
            self.assertTrue(renderer.thumbnails(), "the filmstrip gets its pictures")
        finally:
            renderer.close()
        kinds = set()
        while not events.empty():
            kinds.add(events.get()[0])
        self.assertNotIn("live", kinds, "the frames never reach this process upscaled")
        joined, single = self.decoded(output), self.decoded(self.single)
        self.assertEqual(len(joined), len(single))
        for index, frame in enumerate(joined):
            here = np.abs(frame - single[index]).mean()
            self.assertLess(here, 4.0, f"frame {index} differs from the single pass")
            for neighbour in (index - 1, index + 1):
                if 0 <= neighbour < len(single):
                    self.assertLess(here, np.abs(frame - single[neighbour]).mean(), f"frame {index}")

    def wait_for(self, events: queue.Queue, condition, what: str) -> None:
        deadline = time.monotonic() + TIMEOUT
        while not condition():
            try:
                kind, values = events.get(timeout=0.1)
            except queue.Empty:
                kind = None
            if kind == "error":
                raise values[0]
            if time.monotonic() > deadline:
                self.fail(f"timed out waiting for {what}")


if __name__ == "__main__":
    unittest.main()
