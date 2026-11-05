"""The player's command merging, its frame streams and the end of a video, with fake decoders (no window, GPU
or sound)."""
import queue
import sys
import threading
import time
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from fsr4 import player as player_module
from fsr4.audio import tempo_filter
from fsr4.media import KeyframeIndex, MediaInfo, PastEnd
from fsr4.player import FrameStream, Player, ReverseFrameStream, merge_seeks, reverse_chunk


class MergeSeeksTest(unittest.TestCase):
    def test_a_drag_becomes_its_last_position(self):
        self.assertEqual(merge_seeks(("seek", 10), ("seek", 250)), ("seek", 250))
        self.assertEqual(merge_seeks(("seek", 10), ("step", 1)), ("seek", 11))
        self.assertEqual(merge_seeks(("step", -1), ("step", -1)), ("step", -2))
        self.assertEqual(merge_seeks(("step", 3), ("seek", 7)), ("seek", 7))


class FakeReader:
    """Frames numbered from `first`, `count` of them; each frame's pixels hold its number."""

    opened: list[int] = []

    def __init__(self, first: int, count: int):
        FakeReader.opened.append(first)
        self.next = first
        self.left = count

    def read_into(self, buffer: np.ndarray) -> bool:
        if self.left <= 0:
            return False
        buffer[...] = self.next % 256
        self.next += 1
        self.left -= 1
        return True

    def stop(self) -> None:
        self.left = 0


def pieces_segments(pieces: list[tuple[int, int]]):
    """Segments for ready pieces [(start, end), ...]: None where no piece is."""
    def segments(frame: int):
        for start, end in pieces:
            if start <= frame < end:
                return (lambda first, end=end: FakeReader(first, end - first)), start, end
        return None
    return segments


def take_all(stream, frames) -> list[int]:
    numbers = []
    for frame in frames:
        buffer = stream.take(frame, timeout=2.0)
        if buffer is None:
            break
        numbers.append(int(buffer[0, 0, 0]))
        stream.release(buffer)
    return numbers


class FrameStreamTest(unittest.TestCase):
    def test_source_ending_early_is_the_end(self):
        stream = FrameStream(lambda frame: ((lambda first: FakeReader(first, 3 - first)), 0, 100), 0, 4, 2,
                             "source", strict=False)
        try:
            self.assertEqual(take_all(stream, range(5)), [0, 1, 2])
            self.assertTrue(stream.ended)
        finally:
            stream.stop()

    def test_a_source_that_ends_before_the_frame_passed_it(self):
        """Asked for a frame past the end, the source stream ends (forward) or starts lower (backwards)."""
        def segments(frame):
            return (lambda first: FakeReader(first, 3 - first)), 0, sys.maxsize

        for stream_type in (FrameStream, ReverseFrameStream):
            stream = stream_type(segments, 5, 4, 2, "source", strict=False)
            try:
                self.assertIsNone(stream.take(5, timeout=2.0))
                self.assertTrue(stream.passed(5), stream_type.__name__)
            finally:
                stream.stop()

    def test_damaged_piece_is_an_error(self):
        stream = FrameStream(lambda frame: ((lambda first: FakeReader(first, 3 - first)), 0, 10), 0, 4, 2, "piece",
                             strict=True)
        try:
            with self.assertRaises(RuntimeError):
                take_all(stream, range(5))
        finally:
            stream.stop()

    def test_waits_for_a_piece_that_becomes_ready(self):
        pieces = [(0, 2)]
        stream = FrameStream(pieces_segments(pieces), 0, 4, 2, "pieces", strict=True)
        try:
            self.assertEqual(take_all(stream, range(2)), [0, 1])
            self.assertIsNone(stream.take(2, timeout=0.2), "the next piece is not ready yet")
            self.assertFalse(stream.ended, "the stream waits instead of ending")
            threading.Timer(0.1, lambda: pieces.append((2, 4))).start()
            self.assertEqual(take_all(stream, range(2, 4)), [2, 3])
        finally:
            stream.stop()


class PutBackTest(unittest.TestCase):
    def test_a_frame_put_back_is_taken_again(self):
        """A frame taken for a tick that cannot be shown (another layer is late) must come back next tick."""
        stream = FrameStream(pieces_segments([(0, 10)]), 0, 4, 2, "pieces", strict=True)
        try:
            first = stream.take(0, 2.0)
            stream.put_back(0, first)
            self.assertIs(stream.take(0, 2.0), first)
            stream.release(first)
            self.assertEqual(int(stream.take(1, 2.0)[0, 0, 0]), 1)
        finally:
            stream.stop()


class SkippingStreamTest(unittest.TestCase):
    def test_a_version_not_being_upscaled_jumps_its_gaps(self):
        """A preset shown during a whole-file upscale has holes; its stream jumps them instead of waiting."""
        pieces = [(0, 2), (5, 7)]

        def skip(frame: int):
            return next((first for first, end in pieces if end > frame), None)

        stream = FrameStream(pieces_segments(pieces), 0, 4, 2, "preset", strict=True, skip=skip)
        try:
            self.assertEqual([int(stream.take(frame, 2.0)[0, 0, 0]) for frame in (0, 1)], [0, 1])
            self.assertIsNone(stream.take(3, timeout=0.5), "no picture in the gap")
            self.assertEqual(int(stream.take(5, 2.0)[0, 0, 0]), 5)
            self.assertEqual(int(stream.take(6, 2.0)[0, 0, 0]), 6)
            self.assertIsNone(stream.take(7, timeout=0.5))
            self.assertTrue(stream.ended)
        finally:
            stream.stop()


class ReverseStreamTest(unittest.TestCase):
    """Playing backwards: chunks decoded forward, handed over last frame first, across pieces."""

    def test_frames_come_in_reverse_across_pieces(self):
        FakeReader.opened = []
        stream = ReverseFrameStream(pieces_segments([(0, 10), (10, 25)]), 24, 4, 2, "pieces", strict=True)
        try:
            self.assertEqual(take_all(stream, range(24, -1, -1)), list(range(24, -1, -1)))
            self.assertIn(10, FakeReader.opened, "a chunk stops at the start of its piece")
        finally:
            stream.stop()

    def test_a_slow_consumer_still_gets_every_frame(self):
        """Backwards every frame is shown in order: nothing is skipped when playback is slower than decoding."""
        stream = ReverseFrameStream(pieces_segments([(0, 200)]), 199, 4, 2, "pieces", strict=True)
        try:
            self.assertEqual(take_all(stream, range(199, 139, -1)), list(range(199, 139, -1)))
        finally:
            stream.stop()

    def test_waits_for_frames_upscaled_behind_the_playhead(self):
        pieces = [(20, 30)]
        stream = ReverseFrameStream(pieces_segments(pieces), 29, 4, 2, "pieces", strict=True)
        try:
            self.assertEqual(take_all(stream, range(29, 19, -1)), list(range(29, 19, -1)))
            self.assertIsNone(stream.take(19, timeout=0.2), "frame 19 is not upscaled yet")
            threading.Timer(0.1, lambda: pieces.append((10, 20))).start()
            self.assertEqual(take_all(stream, range(19, 9, -1)), list(range(19, 9, -1)))
        finally:
            stream.stop()

    def test_a_version_not_being_upscaled_jumps_back_over_gaps(self):
        pieces = [(0, 3), (8, 10)]

        def skip(frame: int):
            return next((min(frame, end - 1) for start, end in reversed(pieces) if start <= frame), None)

        stream = ReverseFrameStream(pieces_segments(pieces), 9, 4, 2, "preset", strict=True, skip=skip)
        try:
            self.assertEqual(take_all(stream, (9, 8)), [9, 8])
            self.assertIsNone(stream.take(5, timeout=0.5), "no picture in the gap")
            self.assertEqual(take_all(stream, (2, 1, 0)), [2, 1, 0])
        finally:
            stream.stop()

    def test_chunks_fit_the_memory_budget(self):
        self.assertEqual(reverse_chunk(1920, 1080), 8)
        self.assertEqual(reverse_chunk(3840, 2160), 4)
        self.assertEqual(reverse_chunk(320, 180), 30)


class FakeView:
    """What the player draws on: records the frame number of each still it shows."""

    def __init__(self):
        self.stills: queue.Queue = queue.Queue()

    def needed_layers(self) -> set[str]:
        return set()

    def show_still(self, source: np.ndarray, layers: dict, text: str = "", spinner: bool = False) -> None:
        self.stills.put(int(source[0, 0, 0]))

    def set_spinner(self, visible: bool, advance: bool = False) -> None:
        pass


class EndTest(unittest.TestCase):
    """A video that ends before its estimated length (30 frames announced, 20 real ones)."""

    REAL = 20

    def read_frame(self, path, info, frame, keyframes, pixel_format):
        if frame >= self.REAL:
            raise PastEnd(path, frame)
        return np.full((2, 4, 4), frame, np.uint8)

    def test_a_seek_past_the_end_shows_the_last_frame(self):
        source = Path("tail.mkv")
        info = MediaInfo(width=4, height=2, frame_rate=Fraction(10), frame_count=30, duration=3.0, codec="h264",
                         container="matroska")
        events: queue.Queue = queue.Queue()
        view = FakeView()
        with mock.patch.object(player_module, "read_frame", self.read_frame), \
                mock.patch.object(player_module, "last_frame_before", lambda *arguments: self.REAL - 1):
            player = Player(view, source, info, KeyframeIndex(source, info),
                            lambda kind, *values: events.put((kind, values)))
            try:
                player.seek(29)
                self.assertEqual(view.stills.get(timeout=5), self.REAL - 1, "the last frame is shown")
                received = []
                while not events.empty():
                    received.append(events.get())
                self.assertIn(("end", (self.REAL,)), received, "the window learns the real length")
                self.assertNotIn("error", [kind for kind, _ in received])
                self.assertEqual(received[-1], ("position", (self.REAL - 1,)))
                self.assertEqual(player.total, self.REAL)
            finally:
                player.close()


class PlayingView(FakeView):
    """Records the frame number of each frame played."""

    def __init__(self):
        super().__init__()
        self.played: list[int] = []

    def present(self, source: np.ndarray, shown: dict) -> None:
        self.played.append(int(source[0, 0, 0]))

    def decode_target(self) -> tuple:
        return "the view", None, (4, 2), None


class SpeedTest(unittest.TestCase):
    """Changing speed while playing a silent video of 200 frames a second, with fake decoders."""

    FRAMES = 250

    def setUp(self):
        FakeReader.opened = []
        self.view = PlayingView()
        source = Path("clip.mkv")
        info = MediaInfo(width=4, height=2, frame_rate=Fraction(200), frame_count=self.FRAMES, duration=1.25,
                         codec="h264", container="matroska")
        patches = (mock.patch.object(player_module, "FrameReader", self.reader),
                   mock.patch.object(player_module, "read_frame",
                                     lambda path, info, frame, keyframes, pixels: np.full((2, 4, 4), 0, np.uint8)),
                   mock.patch.object(player_module, "AudioPlayer", side_effect=OSError("no sound in this test")))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.player = Player(self.view, source, info, KeyframeIndex(source, info), lambda kind, *values: None)
        self.addCleanup(self.player.close)
        self.player.attach(SimpleNamespace(total=self.FRAMES, exact=True), (4, 2))

    def reader(self, path, info, start_frame, keyframes, pixel_format):
        return FakeReader(start_frame, self.FRAMES - start_frame)

    def wait_for(self, condition, what: str) -> None:
        deadline = time.monotonic() + 5
        while not condition():
            self.assertLess(time.monotonic(), deadline, f"waiting for {what}")
            time.sleep(0.002)

    def played_on(self) -> None:
        count = len(self.view.played)
        self.wait_for(lambda: len(self.view.played) >= count + 3, "more frames")

    def test_a_new_speed_keeps_the_decoders(self):
        self.player.play()
        self.played_on()
        opened = list(FakeReader.opened)
        for speed in (0.5, 0.25, 0.1):
            self.player.set_speed(speed)
            self.wait_for(lambda: self.player.speed == speed, "the new speed")
            self.played_on()
        self.assertEqual(FakeReader.opened, opened)
        frames = self.view.played
        self.assertEqual(frames, sorted(frames))

    def test_a_drag_on_the_speed_keeps_the_picture_moving(self):
        self.player.play()
        self.played_on()
        played = len(self.view.played)
        for step in range(150):  # a new speed every 2 ms is more often than any frame comes
            self.player.set_speed(0.5 + step / 1000)
            time.sleep(0.002)
        self.assertGreater(len(self.view.played) - played, 10, "0.3 s at more than 100 frames a second")

    def test_above_2x_frames_are_skipped_evenly(self):
        self.player.set_speed(5.0)
        self.player.play()
        self.wait_for(lambda: len(self.view.played) >= 10, "frames")
        frames = self.view.played[1:10]  # the first one starts the decoders
        self.assertEqual({later - earlier for earlier, later in zip(frames, frames[1:])}, {3})

    def test_a_turn_starts_the_decoders_over(self):
        self.player.play()
        self.played_on()
        self.player.set_speed(0.5)
        self.played_on()
        opened = len(FakeReader.opened)
        self.player.set_speed(-0.5)
        self.wait_for(lambda: len(FakeReader.opened) > opened, "a decoder going backwards")
        turned_at = len(self.view.played)
        self.played_on()
        frames = self.view.played[turned_at:]
        self.assertEqual(frames, sorted(frames, reverse=True))

    def test_a_speed_of_0_pauses_and_the_next_speed_plays_on(self):
        self.player.play()
        self.played_on()
        self.player.set_speed(0)
        self.wait_for(lambda: not self.player.playing, "the pause")
        self.player.set_speed(0.5)
        self.wait_for(lambda: self.player.playing, "playing again")
        self.assertEqual(self.player.speed, 0.5)
        self.played_on()

    def test_a_speed_while_paused_does_not_play(self):
        self.player.set_speed(0)
        self.player.set_speed(0.5)
        self.wait_for(lambda: self.player.speed == 0.5, "the new speed")
        self.assertFalse(self.player.playing)

    def test_speeds_queued_behind_each_other_become_the_last_one(self):
        player = Player.__new__(Player)
        player._commands, player._deferred = queue.Queue(), player_module.deque()
        for command in (("speed", 0.5), ("speed", 0.6), ("speed", 0.7), ("seek", 3), ("seek", 4), ("pause",)):
            player._commands.put(command)
        self.assertEqual(player._next_command(0.0), ("speed", 0.7))
        self.assertEqual(player._next_command(0.0), ("seek", 3), "kept in order")
        self.assertEqual(player._next_command(0.0), ("seek", 4))
        self.assertEqual(player._next_command(0.0), ("pause",))


class TempoTest(unittest.TestCase):
    def test_atempo_chain_stays_in_its_range(self):
        self.assertEqual(tempo_filter(1.5), "atempo=1.5")
        self.assertEqual(tempo_filter(0.25), "atempo=0.5,atempo=0.5")
        self.assertEqual(tempo_filter(4.0), "atempo=2,atempo=2")


if __name__ == "__main__":
    unittest.main()
