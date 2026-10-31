"""The player's command merging and its frame streams, with fake decoders (no window, GPU or sound)."""
import threading
import unittest

import numpy as np

from fsr4.player import FrameStream, merge_seeks


class MergeSeeksTest(unittest.TestCase):
    def test_a_drag_becomes_its_last_position(self):
        self.assertEqual(merge_seeks(("seek", 10), ("seek", 250)), ("seek", 250))
        self.assertEqual(merge_seeks(("seek", 10), ("step", 1)), ("seek", 11))
        self.assertEqual(merge_seeks(("step", -1), ("step", -1)), ("step", -2))
        self.assertEqual(merge_seeks(("step", 3), ("seek", 7)), ("seek", 7))


class FakeReader:
    """Frames numbered from `first`, `count` of them; each frame's pixels hold its number."""

    def __init__(self, first: int, count: int):
        self.next = first
        self.left = count

    def read_into(self, buffer: np.ndarray) -> bool:
        if self.left == 0:
            return False
        buffer[...] = self.next % 256
        self.next += 1
        self.left -= 1
        return True

    def stop(self) -> None:
        self.left = 0


class FrameStreamTest(unittest.TestCase):
    def take_all(self, stream: FrameStream, first: int, count: int) -> list[int]:
        numbers = []
        for frame in range(first, first + count):
            buffer = stream.take(frame, timeout=2.0)
            if buffer is None:
                break
            numbers.append(int(buffer[0, 0, 0]))
            stream.release(buffer)
        return numbers

    def test_source_ending_early_is_the_end(self):
        stream = FrameStream(lambda start: ((lambda: FakeReader(start, 3 - start)), 100), 0, 4, 2, "source",
                             strict=False)
        try:
            self.assertEqual(self.take_all(stream, 0, 5), [0, 1, 2])
            self.assertTrue(stream.ended)
        finally:
            stream.stop()

    def test_damaged_piece_is_an_error(self):
        stream = FrameStream(lambda start: ((lambda: FakeReader(start, 3 - start)), 10), 0, 4, 2, "piece",
                             strict=True)
        try:
            with self.assertRaises(RuntimeError):
                self.take_all(stream, 0, 5)
        finally:
            stream.stop()

    def test_waits_for_a_piece_that_becomes_ready(self):
        ready = threading.Event()

        def segments(start: int):
            if start < 2:
                return (lambda: FakeReader(start, 2 - start)), 2
            if ready.is_set():
                return (lambda: FakeReader(start, 4 - start)), 4
            return None

        stream = FrameStream(segments, 0, 4, 2, "pieces", strict=True)
        try:
            self.assertEqual(self.take_all(stream, 0, 2), [0, 1])
            self.assertIsNone(stream.take(2, timeout=0.2), "the next piece is not ready yet")
            self.assertFalse(stream.ended, "the stream waits instead of ending")
            threading.Timer(0.1, ready.set).start()
            self.assertEqual(self.take_all(stream, 2, 2), [2, 3])
        finally:
            stream.stop()


class SkippingStreamTest(unittest.TestCase):
    def test_before_version_jumps_its_gaps(self):
        """The version with the previous settings has holes; its stream jumps them instead of waiting."""
        pieces = [(0, 2), (5, 7)]

        def segments(start: int):
            for first, end in pieces:
                if first <= start < end:
                    return (lambda: FakeReader(start, end - start)), end
            return None

        def skip(frame: int):
            return next((first for first, end in pieces if end > frame), None)

        stream = FrameStream(segments, 0, 4, 2, "before", strict=True, skip=skip)
        try:
            self.assertEqual([int(stream.take(frame, 2.0)[0, 0, 0]) for frame in (0, 1)], [0, 1])
            self.assertIsNone(stream.take(3, timeout=0.5), "no picture in the gap")
            self.assertEqual(int(stream.take(5, 2.0)[0, 0, 0]), 5)
            self.assertEqual(int(stream.take(6, 2.0)[0, 0, 0]), 6)
            self.assertIsNone(stream.take(7, timeout=0.5))
            self.assertTrue(stream.ended)
        finally:
            stream.stop()


if __name__ == "__main__":
    unittest.main()
