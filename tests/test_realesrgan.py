"""Real-ESRGAN: choosing its scale, what it refuses, and on the machine pictures, jobs and the renderer's pieces."""
import queue
import subprocess
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from itertools import islice
from pathlib import Path
from unittest import mock

import numpy as np

import needs
from fsr4 import media, realesrgan, temporary
from fsr4.image_preview import ImagePreviewer
from fsr4.job import JobOptions, plan_from_info, run_job
from fsr4.pipeline import upscale_stream
from fsr4.presets import BUILTIN, preset_from_json
from fsr4.realesrgan import (MODELS, REAL_ESRGAN, REAL_ESRGAN_X4PLUS, STILL_ONLY, RealEsrganPipeline, Stopped,
                             chunk_lengths, model_scale)
from fsr4.render import RenderSettings, Renderer

WIDTH, HEIGHT, RATE, SECONDS = 320, 180, 30, 2
TIMEOUT = 120
VIDEO_MODEL = MODELS[REAL_ESRGAN]
SLOW_SIZE = (1280, 720)  # seconds with x4plus, so a stop can come in the middle
STOP_SECONDS = 2.0


def info(width: int, height: int) -> media.MediaInfo:
    return media.MediaInfo(width, height, Fraction(30), 30, 1.0, "h264")


def psnr(first: np.ndarray, second: np.ndarray) -> float:
    difference = first[..., :3].astype(np.float32) - second[..., :3].astype(np.float32)
    return 10 * float(np.log10(255.0 ** 2 / max(float(np.mean(difference * difference)), 1e-9)))


class ChoicesTest(unittest.TestCase):
    def test_the_model_runs_at_the_smallest_scale_that_reaches_the_output(self):
        self.assertEqual(model_scale((1920, 1080), (3840, 2160), VIDEO_MODEL), 2)
        self.assertEqual(model_scale((1920, 1080), (1920, 1080), VIDEO_MODEL), 2)
        self.assertEqual(model_scale((1000, 1000), (2001, 2000), VIDEO_MODEL), 3)
        self.assertEqual(model_scale((1000, 1000), (3000, 3000), VIDEO_MODEL), 3)

    def test_the_wider_side_decides_and_the_largest_scale_is_the_limit(self):
        self.assertEqual(model_scale((720, 480), (2560, 1440), VIDEO_MODEL), 4)
        self.assertEqual(model_scale((100, 100), (900, 900), VIDEO_MODEL), 4)

    def test_the_models_for_still_images_only_draw_at_4x(self):
        for upscaler in STILL_ONLY:
            self.assertEqual(model_scale((1920, 1080), (2880, 1620), MODELS[upscaler]), 4)

    def test_chunks_start_short_and_stop_growing(self):
        lengths = list(islice(chunk_lengths(), 5))
        self.assertEqual(lengths, [*realesrgan.FIRST_CHUNKS, *[realesrgan.CHUNK_FRAMES] * 3])

    def test_a_folder_with_a_percent_sign_keeps_it(self):
        self.assertEqual(realesrgan._frames_pattern(Path("C:/100%/out")), "C:\\100%%\\out/%08d.png")

    def test_the_program_is_looked_for_with_its_models(self):
        with tempfile.TemporaryDirectory() as directory:
            program = Path(directory) / realesrgan.PROGRAM
            program.touch()
            with mock.patch.object(realesrgan.shutil, "which", return_value=str(program)), \
                    mock.patch.object(realesrgan, "FOLDER", Path("no such folder")):
                self.assertIsNone(realesrgan.find_program(), "without its models it cannot run")
                (program.parent / "models").mkdir()
                (program.parent / "models" / VIDEO_MODEL.weights).touch()
                self.assertEqual(realesrgan.find_program(), program)
                self.assertIsNone(realesrgan.find_program(REAL_ESRGAN_X4PLUS), "each model has its own weights")


class PlanTest(unittest.TestCase):
    def options(self, **values) -> JobOptions:
        return JobOptions(input=Path("clip.mp4"), output=Path("out.mp4"), upscaler=REAL_ESRGAN, **values)

    def test_a_missing_program_says_how_to_install_it(self):
        with mock.patch("fsr4.job.find_program", return_value=None):
            with self.assertRaisesRegex(ValueError, "not installed.*releases/tag"):
                plan_from_info(self.options(), info(640, 360))
        with mock.patch.object(realesrgan, "find_program", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "not installed"):
                RealEsrganPipeline((640, 360), (1280, 720))

    def test_the_options_that_belong_to_others_are_refused(self):
        for values in ({"sharpness": 0.5}, {"debug_view": True}, {"debug_checking": True}, {"validate": True},
                       {"gpu": 1}):
            with mock.patch("fsr4.job.find_program", return_value=Path("program.exe")):
                with self.assertRaisesRegex(ValueError, "do not apply"):
                    plan_from_info(self.options(**values), info(640, 360))

    def test_the_models_for_still_images_refuse_a_video(self):
        with mock.patch("fsr4.job.find_program", return_value=Path("program.exe")):
            for upscaler in STILL_ONLY:
                with self.assertRaisesRegex(ValueError, "still images"):
                    plan_from_info(JobOptions(input=Path("clip.mp4"), output=Path("out.mp4"), upscaler=upscaler),
                                   info(640, 360))
                plan = plan_from_info(JobOptions(input=Path("still.png"), output=Path("out.png"), upscaler=upscaler),
                                      info(640, 360))
                self.assertTrue(plan.image_mode)


class PresetTest(unittest.TestCase):
    def test_the_builtin_preset_is_named_without_motion(self):
        preset = next(preset for preset in BUILTIN if preset.upscaler == REAL_ESRGAN)
        settings = preset.applied_to(RenderSettings((1280, 720)))
        self.assertEqual((settings.flow, settings.flow_preset, settings.sharpness), ("zero", "fast", None))
        self.assertEqual(settings.describe(), "Real-ESRGAN")

    def test_every_model_has_a_builtin_preset_with_its_name(self):
        for upscaler in MODELS:
            preset = next(preset for preset in BUILTIN if preset.upscaler == upscaler)
            self.assertEqual(preset.applied_to(RenderSettings((1280, 720))).describe(), preset.name)

    def test_a_saved_preset_never_holds_motion_or_sharpening(self):
        for upscaler in MODELS:
            preset = preset_from_json({"name": "Mine", "upscaler": upscaler, "flow": "dis", "flow_preset": "medium",
                                       "sharpness": 0.5})
            self.assertEqual((preset.flow, preset.flow_preset, preset.sharpness), ("zero", "fast", None))


@needs.ffmpeg
@needs.opencv
class UpscaleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        needs.require_realesrgan()
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.source = cls.root / "source.mp4"
        subprocess.run([media.tool_path("ffmpeg"), "-v", "error", "-y",
                        "-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate={RATE}",
                        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", str(SECONDS),
                        "-c:v", "libx264", "-g", "15", "-pix_fmt", "yuv420p", "-c:a", "aac", str(cls.source)], check=True)
        cls.info = media.probe(cls.source)
        cls.frames = [frame.copy() for frame in media.FrameReader(cls.source, cls.info)]
        cls.output_size = (WIDTH * 2, HEIGHT * 2)
        cls.single = cls.root / "single.mp4"
        run_job(JobOptions(input=cls.source, output=cls.single, scale=2.0, upscaler=REAL_ESRGAN, verbose=False))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.addCleanup(temporary.use, temporary.folder())
        self.temporary = temporary.use(self.root / f"temporary_{self._testMethodName}")

    def reference(self, frame: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
        """`frame` at the output size by plain resampling."""
        import cv2
        return cv2.resize(np.ascontiguousarray(frame), size or self.output_size, interpolation=cv2.INTER_CUBIC)

    def assert_each_frame_is_its_own(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """Every upscaled frame looks like its source, more than like the neighbouring frames."""
        for index, (source, upscaled) in enumerate(pairs):
            np.testing.assert_array_equal(source, self.frames[index], f"source of frame {index}")
            here = psnr(upscaled, self.reference(self.frames[index]))
            self.assertGreater(here, 20.0, f"frame {index}")
            for neighbour in (index - 1, index + 1):
                if 0 <= neighbour < len(self.frames):
                    self.assertGreater(here, psnr(upscaled, self.reference(self.frames[neighbour])), f"frame {index}")

    def test_frames_keep_their_order_across_the_chunks(self):
        pipeline = RealEsrganPipeline((WIDTH, HEIGHT), self.output_size)
        pairs = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            for source, upscaled in upscale_stream(iter(self.frames), pipeline, executor):
                pairs.append((source.copy(), upscaled.copy()))
        self.assertEqual(len(pairs), len(self.frames))
        self.assertGreater(len(self.frames), sum(realesrgan.FIRST_CHUNKS) + realesrgan.CHUNK_FRAMES, "several chunks")
        self.assert_each_frame_is_its_own(pairs)
        self.assertEqual(list(self.temporary.iterdir()), [], "the files are removed")

    def test_a_pair_stays_valid_until_the_next_one(self):
        pipeline = RealEsrganPipeline((WIDTH, HEIGHT), self.output_size)
        with ThreadPoolExecutor(max_workers=1) as executor:
            stream = upscale_stream(iter(self.frames), pipeline, executor)
            for index, (source, upscaled) in enumerate(stream):
                before = upscaled.copy()
                time.sleep(0.001)
                np.testing.assert_array_equal(upscaled, before)
                self.assertGreater(psnr(upscaled, self.reference(self.frames[index])), 20.0, f"frame {index}")

    def test_stopping_early_ends_the_program_and_removes_the_files(self):
        pipeline = RealEsrganPipeline((WIDTH, HEIGHT), self.output_size)
        started = []
        real_popen = realesrgan._popen

        def recording_popen(command, *arguments, **keywords):
            process = real_popen(command, *arguments, **keywords)
            started.append(process)
            return process

        with mock.patch.object(realesrgan, "_popen", recording_popen), ThreadPoolExecutor(max_workers=1) as executor:
            stream = upscale_stream(iter(self.frames), pipeline, executor)
            next(stream)
            self.assertTrue(list(self.temporary.iterdir()))
            stream.close()
        self.assertEqual(list(self.temporary.iterdir()), [])
        self.assertTrue(all(process.poll() is not None for process in started))

    def test_a_file_held_for_a_moment_does_not_stay_or_stop_the_work(self):
        real_rmtree = realesrgan.shutil.rmtree
        calls = []

        def held_once(path):
            calls.append(path)
            if len(calls) == 1:
                raise PermissionError(32, "being used by another process")
            real_rmtree(path)

        with mock.patch.object(realesrgan.shutil, "rmtree", held_once), mock.patch.object(realesrgan, "REMOVE_WAIT_SECONDS", 0):
            RealEsrganPipeline((WIDTH, HEIGHT), self.output_size).picture(self.frames[0])
        self.assertEqual(len(calls), 2)
        self.assertEqual(list(self.temporary.iterdir()), [])

    def test_a_folder_that_cannot_be_removed_is_logged(self):
        folder = self.temporary / "kept"
        with mock.patch.object(realesrgan.shutil, "rmtree", side_effect=PermissionError(32, "in use")),                 mock.patch.object(realesrgan, "REMOVE_WAIT_SECONDS", 0), mock.patch.object(realesrgan.log, "emit") as emit:
            realesrgan._remove(folder)
        self.assertIn("could not remove", emit.call_args.args[0])

    def test_a_failed_run_says_why_and_removes_the_files(self):
        pipeline = RealEsrganPipeline((WIDTH, HEIGHT), self.output_size)
        pipeline.scale = 7  # the program refuses it
        with self.assertRaisesRegex(RuntimeError, "Real-ESRGAN failed"):
            pipeline.picture(self.frames[0])
        self.assertEqual(list(self.temporary.iterdir()), [])

    def test_a_picture_comes_back_opaque_at_any_size(self):
        for size in ((500, 200), (WIDTH * 2 + 1, HEIGHT * 2), (WIDTH, HEIGHT)):  # each side on its own, and 1x
            picture = RealEsrganPipeline((WIDTH, HEIGHT), size).picture(self.frames[0])
            self.assertEqual(picture.shape, (size[1], size[0], 4))
            self.assertTrue((picture[..., 3] == 255).all())
            self.assertGreater(psnr(picture, self.reference(self.frames[0], size)), 20.0, f"{size}")

    def test_the_models_for_still_images_draw_a_picture(self):
        for upscaler in STILL_ONLY:
            pipeline = RealEsrganPipeline((WIDTH, HEIGHT), self.output_size, upscaler)
            self.assertIn(MODELS[upscaler].name, pipeline._program_command(Path("in"), Path("out")))
            picture = pipeline.picture(self.frames[0])
            self.assertEqual(picture.shape, (self.output_size[1], self.output_size[0], 4))
            self.assertGreater(psnr(picture, self.reference(self.frames[0])), 20.0, upscaler)

    def slow_still(self) -> Path:
        """A picture x4plus takes seconds on."""
        path = self.temporary / "slow.png"
        media.write_image(path, self.reference(self.frames[0], SLOW_SIZE))
        return path

    def assert_nothing_is_left(self) -> None:
        self.assertEqual([path.name for path in self.temporary.iterdir() if path.name.startswith("fsr4-")], [])

    def test_stop_ends_a_picture_at_once(self):
        pipeline = RealEsrganPipeline(SLOW_SIZE, (SLOW_SIZE[0] * 2, SLOW_SIZE[1] * 2), REAL_ESRGAN_X4PLUS)
        picture = media.read_image(self.slow_still())
        timer = threading.Timer(0.5, pipeline.stop.set)
        timer.start()
        started = time.perf_counter()
        with self.assertRaises(Stopped):
            pipeline.picture(picture)
        timer.join()
        self.assertLess(time.perf_counter() - started, STOP_SECONDS)
        self.assert_nothing_is_left()

    def test_cancelling_an_image_job_stops_the_picture_and_writes_nothing(self):
        output = self.temporary / "slow_up.png"
        cancel = threading.Event()
        timer = threading.Timer(0.5, cancel.set)
        timer.start()
        started = time.perf_counter()
        result = run_job(JobOptions(input=self.slow_still(), output=output, scale=2.0, upscaler=REAL_ESRGAN_X4PLUS,
                                    verbose=False), cancel=cancel)
        timer.join()
        self.assertLess(time.perf_counter() - started, STOP_SECONDS)
        self.assertEqual((result.cancelled, result.frames, output.exists()), (True, 0, False))
        self.assert_nothing_is_left()

    def previewer(self) -> tuple[ImagePreviewer, queue.Queue]:
        needs.require_fsr()
        events: queue.Queue = queue.Queue()
        return ImagePreviewer(self.slow_still(), Path("unused"), lambda kind, *values: events.put((kind, values))), events

    def test_closing_the_previewer_stops_its_picture(self):
        previewer, events = self.previewer()
        previewer.request([RenderSettings((SLOW_SIZE[0] * 2, SLOW_SIZE[1] * 2), flow="zero", upscaler=REAL_ESRGAN_X4PLUS)])
        time.sleep(1.0)
        started = time.perf_counter()
        previewer.close()
        self.assertLess(time.perf_counter() - started, STOP_SECONDS)
        self.assertTrue(events.empty(), "a stop is no error")
        self.assert_nothing_is_left()

    def test_a_picture_no_longer_shown_makes_way(self):
        previewer, events = self.previewer()
        output_size = (SLOW_SIZE[0] * 2, SLOW_SIZE[1] * 2)
        slow = RenderSettings(output_size, flow="zero", upscaler=REAL_ESRGAN_X4PLUS)
        fast = RenderSettings(output_size, flow="zero", upscaler=REAL_ESRGAN)
        try:
            previewer.request([slow])
            time.sleep(1.0)
            previewer.request([fast])
            self.assertEqual(events.get(timeout=TIMEOUT), ("ready", (fast,)))
            self.assertIsNone(previewer.result(slow))
            self.assertFalse(previewer.working)
        finally:
            previewer.close()

    def test_resizing_keeps_the_colours(self):
        exact = RealEsrganPipeline((WIDTH, HEIGHT), self.output_size).picture(self.frames[0])
        resized = RealEsrganPipeline((WIDTH, HEIGHT), (WIDTH * 2 - 2, HEIGHT * 2 - 2)).picture(self.frames[0])
        difference = exact[..., :3].mean(axis=(0, 1)) - resized[..., :3].mean(axis=(0, 1))
        self.assertLess(float(np.abs(difference).max()), 0.5)

    def test_a_job_writes_every_frame_upscaled_with_its_sound(self):
        info = media.probe(self.single)
        self.assertEqual((info.width, info.height), self.output_size)
        self.assertTrue(info.audio)
        self.assertEqual(sum(1 for _ in media.FrameReader(self.single, info)), len(self.frames))

    def test_an_image_job_writes_the_picture(self):
        source = self.root / "still.png"
        media.write_image(source, self.frames[0])
        output = self.root / "still_up.png"
        result = run_job(JobOptions(input=source, output=output, scale=2.0, upscaler=REAL_ESRGAN, verbose=False))
        self.assertEqual(result.frames, 1)
        self.assertEqual(media.read_image(output).shape[:2], self.output_size[::-1])

    def test_pieces_around_a_seek_join_into_the_single_pass(self):
        events: queue.Queue = queue.Queue()
        settings = RenderSettings(self.output_size, flow="zero", upscaler=REAL_ESRGAN)
        renderer = Renderer(self.source, self.info, settings, Path("unused"),
                            lambda kind, *values: events.put((kind, values)))
        try:
            renderer.want_live = True
            timeline = renderer.timeline
            renderer.set_target(30)
            self.wait_for(events, lambda: timeline.ready_until(30) >= 40, "frames after the seek")
            renderer.set_batch(True)
            self.wait_for(events, timeline.complete, "the whole video")
            output = self.root / "joined.mp4"
            self.assertEqual(renderer.export(output), len(self.frames))
            self.assertTrue(renderer.thumbnails(), "the filmstrip gets its pictures")
        finally:
            renderer.close()
        joined, single = (
            [frame[..., :3].astype(np.int16) for frame in media.FrameReader(path, media.probe(path))]
            for path in (output, self.single))
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
