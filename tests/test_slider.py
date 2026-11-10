"""The marked sliders (zoom, speed, volume): where labels go, where a click lands, and the volume's decibels."""
import tkinter as tk
import unittest

import needs

if not needs.HAS_OPENCV:
    raise unittest.SkipTest("opencv-python is not installed (the window needs it)")

from fsr4.ui_slider import KNOB_INSET, Mark, MarkedSlider, nearest_mark, place_labels
from fsr4.ui_theme import Theme
from fsr4.ui_transport import (QUIETEST_VOLUME, SILENT_SHARE, SPEED_MARKS, VOLUME_MARKS, VOLUME_TICKS,
                               format_volume, step_volume, volume_at, volume_fraction, volume_icon)

THREE_MARKS = (Mark(0.0, "0", 0.0), Mark(0.5, "50", 0.5), Mark(1.0, "100", 1.0))


class PureTest(unittest.TestCase):
    def test_labels_give_way_by_order(self):
        centres, widths, ranks = [10, 20, 60], [16, 16, 16], [0, 0, 0]
        self.assertEqual(place_labels(centres, widths, ranks, [0, 1, 2], gap=4), [0, 2])
        self.assertEqual(place_labels(centres, widths, ranks, [1, 0, 2], gap=4), [1, 2])
        self.assertEqual(place_labels(centres, widths, ranks, [2, 1, 0], gap=30), [2, 0])
        self.assertEqual(place_labels(centres, widths, ranks, [2, 1, 0], gap=40), [2])

    def test_once_a_label_gives_way_the_less_useful_ones_do_too(self):
        centres, widths = [0, 40, 50, 100], [10, 10, 10, 10]
        self.assertEqual(place_labels(centres, widths, [0, 0, 1, 2], [0, 3, 1, 2], gap=4), [0, 3, 1])
        self.assertEqual(place_labels([0, 3, 80], [10, 10, 10], [0, 1, 2], [0, 1, 2], gap=4), [0],
                         "the rank 2 label would fit, but a rank 1 label gave way")

    def test_nearest_mark_within_reach(self):
        marks = [Mark(0.0, "0", 0), Mark(0.5, "50", 0.5), Mark(1.0, "100", 1)]
        self.assertEqual(nearest_mark(marks, 0.52, 0.03).value, 0.5)
        self.assertIsNone(nearest_mark(marks, 0.4, 0.03))
        self.assertIsNone(nearest_mark([], 0.4, 0.03))

    def test_the_volume_is_laid_out_in_decibels(self):
        self.assertEqual(volume_fraction(0.0), 0.0)
        self.assertEqual(volume_fraction(QUIETEST_VOLUME), SILENT_SHARE)
        self.assertAlmostEqual(volume_fraction(1.0), 1.0)
        self.assertAlmostEqual(volume_fraction(1.0) - volume_fraction(0.1), volume_fraction(0.1) - volume_fraction(0.01))

    def test_every_volume_tick_reads_back(self):
        for volume, _, _ in VOLUME_TICKS:
            self.assertEqual(volume_at(volume_fraction(volume)), volume)

    def test_nothing_lies_between_silence_and_the_quietest_volume(self):
        self.assertEqual(volume_at(SILENT_SHARE * 0.4), 0.0)
        self.assertEqual(volume_at(SILENT_SHARE * 0.6), QUIETEST_VOLUME)
        self.assertEqual(volume_at(-1.0), 0.0)
        self.assertEqual(volume_at(2.0), 1.0)

    def test_volume_steps_in_decibels(self):
        self.assertEqual(step_volume(0.0, 4), QUIETEST_VOLUME)
        self.assertEqual(step_volume(0.0, -4), 0.0)
        self.assertEqual(step_volume(QUIETEST_VOLUME, -2), 0.0)
        self.assertEqual(step_volume(0.5, -4), 0.32)
        self.assertEqual(step_volume(0.8, 4), 1.0)
        self.assertEqual(step_volume(1.0, -2), 0.79)

    def test_volume_percentages(self):
        self.assertEqual([format_volume(volume) for volume in (0.0, 0.013, 0.01, 0.1, 0.8, 1.0)],
                         ["0%", "1.3%", "1%", "10%", "80%", "100%"])

    def test_speaker_shows_the_level(self):
        self.assertEqual(volume_icon(0.8, muted=True), "mute")
        self.assertEqual(volume_icon(0.0, muted=False), "volume_none")
        self.assertEqual(volume_icon(0.2, muted=False), "volume_low")
        self.assertEqual(volume_icon(0.25, muted=False), "volume")


class SliderTest(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.geometry("+-3000+-3000")  # mapped (sizes are real) but off every screen
        self.theme = Theme(self.root)
        self.picks: list[tuple[float, Mark | None]] = []

    def tearDown(self):
        self.root.destroy()

    def slider(self, width: int, marks) -> MarkedSlider:
        slider = MarkedSlider(self.root, self.theme, width, on_pick=lambda *pick: self.picks.append(pick),
                              on_wheel=lambda _: None, marks=marks)
        slider.pack()
        self.root.update()
        slider.redraw()
        return slider

    def labels(self, slider: MarkedSlider) -> list[str]:
        return [slider.itemcget(item, "text") for item in slider.find_all() if slider.type(item) == "text"]

    def test_a_narrow_speed_slider_keeps_the_main_labels(self):
        self.assertEqual(self.labels(self.slider(84, SPEED_MARKS)), ["0", "¼", "1×", "5"])
        self.assertEqual(len(self.labels(self.slider(176, SPEED_MARKS))), len(SPEED_MARKS))

    def test_the_volume_slider_labels_every_tick_and_keeps_the_ends_when_narrow(self):
        self.assertEqual(self.labels(self.slider(150, VOLUME_MARKS)), ["0", "1", "3", "10", "30", "100"])
        self.assertEqual(self.labels(self.slider(48, VOLUME_MARKS)), ["0", "100"])

    def test_labels_stay_inside_the_slider(self):
        for marks in (VOLUME_MARKS, (Mark(0.0, "Original", "source"), Mark(1.0, "800%", 8))):
            slider = self.slider(150, marks)
            boxes = [slider.bbox(item) for item in slider.find_all() if slider.type(item) == "text"]
            # A text's box is a pixel wider than its glyphs on each side.
            inside = [0 <= left + 1 and right - 1 <= slider.winfo_width() and bottom - 1 <= slider.winfo_height()
                      for left, _, right, bottom in boxes]
            self.assertTrue(all(inside), (boxes, slider.winfo_width(), slider.winfo_height()))

    def test_the_label_under_the_knob_always_shows(self):
        slider = self.slider(84, SPEED_MARKS)
        slider.show(SPEED_MARKS[1].fraction, SPEED_MARKS[1].value)
        slider.redraw()
        self.assertIn(".05", self.labels(slider))

    def test_clicks_land_on_marks_or_anywhere(self):
        slider = self.slider(96, THREE_MARKS)
        middle = slider.winfo_height() // 2
        fraction, mark = slider.pick_at(slider.x_of(0.5) + self.theme.px(3), middle, on_label=False)
        self.assertEqual((fraction, mark.value), (0.5, 0.5))
        fraction, mark = slider.pick_at(slider.x_of(0.25), middle, on_label=False)
        self.assertIsNone(mark)
        self.assertAlmostEqual(fraction, 0.25, delta=0.02)
        label_y = middle + self.theme.px(13)
        _, mark = slider.pick_at(slider.x_of(0.5) + self.theme.px(6), label_y, on_label=True)
        self.assertEqual(mark.value, 0.5, "a click on a label lands on its mark")
        _, mark = slider.pick_at(slider.x_of(0.5) + self.theme.px(6), label_y, on_label=False)
        self.assertIsNone(mark, "a drag along the labels does not jump to them")

    def test_a_disabled_slider_takes_no_click(self):
        slider = self.slider(96, THREE_MARKS)
        slider.set_enabled(False)
        slider.event_generate("<ButtonPress-1>", x=self.theme.px(KNOB_INSET) + 10, y=slider.winfo_height() // 2)
        self.assertEqual(self.picks, [])
        slider.set_enabled(True)
        slider.event_generate("<ButtonPress-1>", x=self.theme.px(KNOB_INSET) + 10, y=slider.winfo_height() // 2)
        self.assertEqual(len(self.picks), 1)


if __name__ == "__main__":
    unittest.main()
