"""Piece bookkeeping and the renderer's scheduling decisions (pure logic, no GPU)."""
import unittest
from pathlib import Path

from fsr4 import player, render
from fsr4.timeline import Piece, Timeline, intersect_ranges, keep_going, next_start, piece_lengths


def piece(start: int, end: int, ready: bool = True) -> Piece:
    return Piece(start, end, Path(f"piece_{start}.mp4"), ready)


class TimelineTest(unittest.TestCase):
    def setUp(self):
        self.timeline = Timeline(1000, exact=True)
        for start, end, ready in ((0, 100, True), (100, 130, True), (300, 400, False), (400, 500, True)):
            self.timeline.add(piece(start, end, ready))

    def test_overlaps_are_refused(self):
        with self.assertRaises(ValueError):
            self.timeline.add(piece(90, 120))
        with self.assertRaises(ValueError):
            self.timeline.add(piece(250, 301))
        with self.assertRaises(ValueError):
            self.timeline.add(piece(5, 5))
        self.timeline.add(piece(130, 300))

    def test_playable_stops_at_pending_pieces_and_gaps(self):
        self.assertEqual(self.timeline.ready_until(0), 130)
        self.assertEqual(self.timeline.ready_until(50), 130)
        self.assertEqual(self.timeline.ready_until(200), 200)
        self.assertEqual(self.timeline.ready_until(300), 300, "pending piece is not playable")
        self.assertEqual(self.timeline.ready_until(420), 500)

    def test_missing_counts_pending_pieces_and_the_busy_run(self):
        self.assertEqual(self.timeline.next_missing(0), 130)
        self.assertEqual(self.timeline.next_missing(300), 500)
        self.assertEqual(self.timeline.next_missing(0, busy=(130, 220)), 220)
        self.assertEqual(self.timeline.next_missing(0, busy=(130, 300)), 500)
        self.timeline.add(piece(500, 1000))
        self.assertIsNone(self.timeline.next_missing(450))

    def test_gap_end_and_completion(self):
        self.assertEqual(self.timeline.gap_end(130), 300)
        self.assertEqual(self.timeline.gap_end(600), 1000)
        self.assertFalse(self.timeline.complete())
        self.timeline.add(piece(130, 300))
        self.timeline.add(piece(500, 1000))
        self.assertFalse(self.timeline.complete(), "one piece is still pending")
        self.timeline.mark_ready(self.timeline.piece_at(350))
        self.assertTrue(self.timeline.complete())
        self.assertEqual(self.timeline.ready_ranges(), [(0, 1000)])

    def test_ranges_and_prefix(self):
        self.assertEqual(self.timeline.ready_ranges(), [(0, 130), (400, 500)])
        self.assertEqual([p.start for p in self.timeline.ready_prefix()], [0, 100])
        self.assertEqual(self.timeline.ready_frames(), 230)

    def test_ranges_ready_in_two_versions(self):
        other = [(0, 50), (90, 120), (450, 700)]
        self.assertEqual(intersect_ranges(self.timeline.ready_ranges(), other), [(0, 50), (90, 120), (450, 500)])
        self.assertEqual(intersect_ranges([], other), [])

    def test_total_follows_the_real_end(self):
        self.timeline.add(piece(990, 1020))
        self.assertEqual(self.timeline.total, 1020)
        self.timeline.set_total(1010)
        self.assertEqual(self.timeline.total, 1020, "never below the last piece")
        timeline = Timeline(1000)
        timeline.set_total(960)
        self.assertEqual(timeline.total, 960)


class EstimatedTotalTest(unittest.TestCase):
    """The probe's frame count can be short or missing: past it, frames count as missing until a read reaches
    the end of the video, so nothing is dropped from the end of the saved file."""

    def test_frames_past_an_estimate_are_missing(self):
        timeline = Timeline(100)
        timeline.add(piece(0, 100))
        self.assertEqual(timeline.next_missing(50), 100)
        self.assertFalse(timeline.complete())
        self.assertEqual(next_start(timeline, 50, batch=True, low_water=600), 100)
        self.assertTrue(keep_going(timeline, 50, True, 600, busy=(100, 120), position=120))

    def test_reaching_the_end_settles_the_count(self):
        timeline = Timeline(100)
        timeline.add(piece(0, 100))
        timeline.add(piece(100, 114))
        self.assertEqual(timeline.total, 114)
        self.assertFalse(timeline.complete())
        timeline.set_total(114)
        self.assertTrue(timeline.complete())
        self.assertIsNone(timeline.next_missing(0))

    def test_unknown_count(self):
        timeline = Timeline(1)
        self.assertEqual(next_start(timeline, 0, batch=False, low_water=600), 0)
        self.assertTrue(keep_going(timeline, 0, False, 1800, busy=(0, 40), position=40))


class SchedulingTest(unittest.TestCase):
    def setUp(self):
        self.timeline = Timeline(10_000, exact=True)
        self.timeline.add(piece(1000, 1300))

    def test_preview_starts_only_below_the_low_water_mark(self):
        self.assertEqual(next_start(self.timeline, 900, batch=False, low_water=600), 900)
        self.assertEqual(next_start(self.timeline, 1000, batch=False, low_water=600), 1300)
        self.assertIsNone(next_start(self.timeline, 1000, batch=False, low_water=200), "300 frames ready ahead")

    def test_batch_starts_at_the_playhead_then_wraps(self):
        self.assertEqual(next_start(self.timeline, 1100, batch=True, low_water=600), 1300)
        self.timeline.add(piece(1300, 10_000))
        self.assertEqual(next_start(self.timeline, 1100, batch=True, low_water=600), 0)
        self.timeline.add(piece(0, 1000))
        self.assertIsNone(next_start(self.timeline, 1100, batch=True, low_water=600))

    def test_run_continues_while_it_renders_what_is_needed(self):
        self.assertTrue(keep_going(self.timeline, 900, False, 1800, busy=(900, 950), position=950))
        # rendered up to the piece at 1000: the next needed frame is after it, not 1000
        self.assertFalse(keep_going(self.timeline, 900, False, 1800, busy=(900, 1000), position=1000))

    def test_run_stops_after_a_seek(self):
        self.assertFalse(keep_going(self.timeline, 5000, False, 1800, busy=(1300, 1400), position=1400))
        self.assertFalse(keep_going(self.timeline, 200, True, 1800, busy=(1300, 1400), position=1400))
        # seeking back into the part this run already covers changes nothing
        self.assertTrue(keep_going(self.timeline, 1320, False, 1800, busy=(1300, 1400), position=1400))

    def test_preview_run_stops_at_the_high_water_mark(self):
        self.assertTrue(keep_going(self.timeline, 1300, False, 600, busy=(1300, 1899), position=1899))
        self.assertFalse(keep_going(self.timeline, 1300, False, 600, busy=(1300, 1900), position=1900))
        self.assertTrue(keep_going(self.timeline, 1300, True, 600, busy=(1300, 1900), position=1900))

    def test_batch_run_from_the_start_after_the_tail_is_done(self):
        self.timeline.add(piece(1300, 10_000))
        self.assertTrue(keep_going(self.timeline, 5000, True, 600, busy=(0, 40), position=40))

    def test_a_waiting_player_never_leaves_the_renderer_idle(self):
        """The player may wait for BUFFER_MAX_SECONDS ready past the frame after the playhead (and the playhead it
        last reported can be a little behind): a version one frame short of that must still get work."""
        for rate in (24, 30, 60):
            needed = int(round(player.BUFFER_MAX_SECONDS * rate))
            playhead = 1000
            timeline = Timeline(100_000)
            timeline.add(piece(0, playhead + needed))  # needed - 1 frames from playhead + 1: the player waits
            target = playhead - int(round(0.5 * rate))
            self.assertIsNotNone(next_start(timeline, target, False, int(round(render.PREVIEW_LOW_SECONDS * rate))),
                                 f"{rate} fps")

    def test_piece_lengths_grow(self):
        lengths = piece_lengths(30)
        self.assertEqual([next(lengths) for _ in range(6)], [15, 30, 60, 120, 240, 240])
        self.assertEqual(next(piece_lengths(1)), 1)


if __name__ == "__main__":
    unittest.main()
