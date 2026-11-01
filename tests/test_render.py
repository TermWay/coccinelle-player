"""The piece renderer on the GPU: seeks, batch, joins and settings changes give the same video as one pass.

Frames rendered in separate runs differ slightly from a single pass near each run start (FSR's history starts
over after the pre-roll), so the check is that every joined frame is closest to the single pass's frame with
the same number: an off-by-one at a join would match a neighbour instead.
"""
import json
import queue
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import needs
from fsr4 import media, render
from fsr4.ffx import find_sdk_dir
from fsr4.job import JobOptions, run_job
from fsr4.render import RenderSettings, Renderer

WIDTH, HEIGHT, RATE, SECONDS = 320, 180, 30, 5
TIMEOUT = 60


def make_source(path: Path) -> None:
    subprocess.run([media.tool_path("ffmpeg"), "-v", "error", "-y",
                    "-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate={RATE}",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", str(SECONDS),
                    "-c:v", "libx264", "-g", "45", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)], check=True)


def probe_streams(path: Path) -> list[dict]:
    return json.loads(subprocess.run(
        [media.tool_path("ffprobe"), "-v", "error", "-show_entries",
         "stream=codec_type,duration,color_space,color_primaries,color_transfer,color_range,codec_tag_string",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout)["streams"]


def decode(path: Path) -> list[np.ndarray]:
    info = media.probe(path)
    return [frame[..., :3].astype(np.int16) for frame in media.FrameReader(path, info)]


@needs.ffmpeg
@needs.opencv
class RendererTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        needs.require_fsr()
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.source = cls.root / "source.mp4"
        make_source(cls.source)
        cls.info = media.probe(cls.source)
        single = cls.root / "single.mp4"
        run_job(JobOptions(input=cls.source, output=single, scale=2.0, codec="amf", verbose=False))
        cls.single = decode(single)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.events: queue.Queue = queue.Queue()
        self.renderer = Renderer(self.source, self.info, RenderSettings((WIDTH * 2, HEIGHT * 2)), find_sdk_dir(None),
                                 lambda kind, *values: self.events.put((kind, values)))

    def tearDown(self):
        directory = self.renderer.directory
        self.renderer.close()
        self.assertFalse(directory.exists(), "close deletes the pieces")

    def wait_for(self, condition, what: str) -> None:
        deadline = time.monotonic() + TIMEOUT
        while not condition():
            try:
                kind, values = self.events.get(timeout=0.1)
            except queue.Empty:
                kind = None
            if kind == "error":
                raise values[0]
            if time.monotonic() > deadline:
                self.fail(f"timed out waiting for {what}")

    def assert_matches_single_pass(self, path: Path) -> None:
        joined = decode(path)
        self.assertEqual(len(joined), len(self.single))
        for index, frame in enumerate(joined):
            here = np.abs(frame - self.single[index]).mean()
            for neighbour in (index - 1, index + 1):
                if 0 <= neighbour < len(self.single):
                    self.assertLess(here, np.abs(frame - self.single[neighbour]).mean(), f"frame {index}")
            self.assertLess(here, 4.0, f"frame {index} differs from the single pass")

    def test_preview_follows_seeks_then_batch_completes_and_joins(self):
        timeline = self.renderer.timeline
        self.wait_for(lambda: timeline.ready_until(0) >= 30, "the first second in preview mode")
        self.renderer.set_target(95)  # seek into frames nobody rendered yet
        self.wait_for(lambda: timeline.ready_until(95) >= 110, "frames after the seek")
        starts = [piece.start for piece in timeline.pieces()]
        self.assertIn(95, starts, "a run starts exactly at the seek")
        self.renderer.set_batch(True)
        self.wait_for(timeline.complete, "the whole video")
        output = self.root / "joined.mp4"
        self.assertEqual(self.renderer.export(output), len(self.single))
        self.assert_matches_single_pass(output)
        streams = probe_streams(output)
        durations = {stream["codec_type"]: float(stream["duration"]) for stream in streams}
        self.assertAlmostEqual(durations["video"], durations["audio"], delta=0.1)
        tags = ("color_space", "color_primaries", "color_transfer", "color_range", "codec_tag_string")
        single_video = probe_streams(self.root / "single.mp4")[0]
        self.assertEqual({tag: streams[0][tag] for tag in tags}, {tag: single_video[tag] for tag in tags},
                         "the join keeps the encoder's colour tags and the hvc1 tag")

    def test_partial_export_keeps_the_start(self):
        timeline = self.renderer.timeline
        self.wait_for(lambda: timeline.ready_until(0) >= 45, "the first frames")
        output = self.root / "partial.mp4"
        frames = self.renderer.export(output, partial=True)
        self.assertGreaterEqual(frames, 45)
        self.assertEqual(media.probe(output).frame_count, frames)
        with self.assertRaises(RuntimeError):
            self.renderer.export(self.root / "whole.mp4")

    def export_whole(self, source: Path) -> tuple[int, int]:
        """Upscale `source` in batch mode and save it; returns (frames saved, frames the decoder reads)."""
        info = media.probe(source)
        events: queue.Queue = queue.Queue()
        renderer = Renderer(source, info, RenderSettings((WIDTH * 2, HEIGHT * 2)), find_sdk_dir(None),
                            lambda kind, *values: events.put((kind, values)))
        try:
            renderer.set_batch(True)
            deadline = time.monotonic() + TIMEOUT
            while not renderer.timeline.complete():
                self.assertLess(time.monotonic(), deadline, f"{source.name} never completed")
                try:
                    kind, values = events.get(timeout=0.1)
                except queue.Empty:
                    continue
                if kind == "error":
                    raise values[0]
            output = self.root / f"{source.stem}.fsr4.mp4"
            saved = renderer.export(output)
        finally:
            renderer.close()
        return saved, sum(1 for _ in media.FrameReader(source, info))

    def test_estimated_frame_counts_keep_the_end(self):
        dropped = self.root / "dropped.mp4"  # a capture that dropped every 10th frame: the probe counts short
        subprocess.run([media.tool_path("ffmpeg"), "-v", "error", "-y", "-f", "lavfi",
                        "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate={RATE}", "-t", "4",
                        "-vf", "select='not(eq(mod(n\\,10)\\,9))'", "-fps_mode", "vfr", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(dropped)], check=True)
        raw = self.root / "raw.m2v"  # an elementary stream: no frame count at all
        subprocess.run([media.tool_path("ffmpeg"), "-v", "error", "-y", "-f", "lavfi",
                        "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate={RATE}", "-t", "3", "-c:v", "mpeg2video",
                        str(raw)], check=True)
        for source in (dropped, raw):
            saved, decoded = self.export_whole(source)
            self.assertEqual(saved, decoded, f"{source.name}: the saved video ends where the source does")

    def test_a_seek_past_the_real_end_finds_it(self):
        """A video shorter than its estimated length (2 s of video, 3 s of sound, no frame count in Matroska):
        a run started past the end decodes nothing, and the real end is found at once, not in steps."""
        tail = self.root / "tail.mkv"
        subprocess.run([media.tool_path("ffmpeg"), "-v", "error", "-y",
                        "-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate={RATE}:duration=2",
                        "-f", "lavfi", "-i", "sine=duration=3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", str(tail)], check=True)
        info = media.probe(tail)
        frames = sum(1 for _ in media.FrameReader(tail, info))
        self.assertGreater(info.frame_count, frames + render.PREROLL_FRAMES)
        events: queue.Queue = queue.Queue()
        renderer = Renderer(tail, info, RenderSettings((WIDTH * 2, HEIGHT * 2)), find_sdk_dir(None),
                            lambda kind, *values: events.put((kind, values)), target=info.frame_count - 1)
        try:
            deadline = time.monotonic() + TIMEOUT
            totals = []
            while not renderer.timeline.exact:
                self.assertLess(time.monotonic(), deadline, "the end was never found")
                try:
                    kind, values = events.get(timeout=0.1)
                except queue.Empty:
                    continue
                if kind == "error":
                    raise values[0]
                if kind == "total":
                    totals.append(values[0])
            self.assertEqual(renderer.timeline.total, frames)
            self.assertEqual(totals, [frames])
        finally:
            renderer.close()

    def test_versions_are_kept_and_upscaled_side_by_side(self):
        timeline = self.renderer.timeline
        plain = self.renderer.settings
        self.wait_for(lambda: timeline.ready_until(0) >= 45, "a second and a half")
        kept = [piece.path for piece in timeline.pieces(ready_only=True)]
        sharp = RenderSettings((WIDTH * 2, HEIGHT * 2), sharpness=0.5)
        self.renderer.change_settings(sharp)
        self.assertTrue(all(path.exists() for path in kept), "the previous version is kept")
        self.wait_for(lambda: self.renderer.timeline.ready_until(0) >= 30, "the new version's first second")
        self.renderer.change_settings(plain)  # back to the previous settings: no work to redo
        self.assertIs(self.renderer.timeline, timeline)
        self.assertGreaterEqual(self.renderer.timeline.ready_until(0), 45)
        fsr3 = RenderSettings((WIDTH * 2, HEIGHT * 2), upscaler="3.1.6")
        self.renderer.show([plain, fsr3])  # one on each side of the picture: both are upscaled at the playhead
        self.renderer.set_target(75)
        other = self.renderer.timeline_of(fsr3)
        self.wait_for(lambda: timeline.ready_until(75) >= 105 and other.ready_until(75) >= 105,
                      "both versions after the seek")

    def test_forgotten_versions_are_deleted_even_while_played(self):
        first = self.renderer.timeline
        self.wait_for(lambda: first.ready_until(0) >= 30, "the first second")
        playing = first.pieces(ready_only=True)[0].path
        with mock.patch.object(render, "KEPT_VERSIONS", 0):
            with open(playing, "rb"):  # like a playback decoder: Windows refuses to delete the file meanwhile
                self.renderer.change_settings(RenderSettings((WIDTH * 2, HEIGHT * 2), sharpness=0.3))
                self.assertTrue(playing.exists())
            self.wait_for(lambda: self.renderer.timeline.ready_until(0) >= 30 and not playing.exists(),
                          "the forgotten version deleted once its file is closed")

    def test_settings_change_starts_over(self):
        timeline = self.renderer.timeline
        self.wait_for(lambda: timeline.ready_until(0) >= 30, "the first second")
        self.renderer.change_settings(RenderSettings((WIDTH * 2, HEIGHT * 2), sharpness=0.5))
        self.assertIsNot(self.renderer.timeline, timeline)
        self.wait_for(lambda: self.renderer.timeline.ready_until(0) >= 30, "the first second again")
        self.assertTrue(all(piece.path.name.startswith("piece_1_") for piece in self.renderer.timeline.pieces()))


if __name__ == "__main__":
    unittest.main()
