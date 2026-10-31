"""The player's command merging and its frame streams, with fake decoders (no window, GPU or sound)."""
import threading
import unittest

import numpy as np

from fsr4.audio import tempo_filter
from fsr4.player import FrameStream, ReverseFrameStream, merge_seeks, reverse_chunk


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


class TempoTest(unittest.TestCase):
    def test_atempo_chain_stays_in_its_range(self):
        self.assertEqual(tempo_filter(1.5), "atempo=1.5")
        self.assertEqual(tempo_filter(0.25), "atempo=0.5,atempo=0.5")
        self.assertEqual(tempo_filter(4.0), "atempo=2,atempo=2")


if __name__ == "__main__":
    unittest.main()
