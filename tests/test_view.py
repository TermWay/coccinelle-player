"""The video view's geometry and the left side's alignment with the upscaled picture (no window needed)."""
import unittest

import needs

if not needs.HAS_OPENCV:
    raise unittest.SkipTest("opencv-python is not installed (the view draws with it)")

import cv2
import numpy as np

from fsr4.ui_transport import (FASTEST_SPEED, SLOWEST_SPEED, SPEED_STOPS, STOPPED_SHARE, format_precise, format_speed,
                               format_time, next_stop, speed_at, speed_fraction)
from fsr4.ui_art import Sketch
from fsr4.ui_video import (VideoView, blend, blend_outline, block_view, compute_geometry, decode_request,
                           minimap_box, over, reference_view, reframe, step_zoom, upscaled_view)


def gradient(width: int, height: int) -> np.ndarray:
    """A BGRA picture where every pixel differs from its neighbours."""
    ys, xs = np.mgrid[0:height, 0:width]
    picture = np.zeros((height, width, 4), np.uint8)
    picture[..., 0] = (xs * 7 + ys * 3) % 256
    picture[..., 1] = (xs * 13) % 256
    picture[..., 2] = (ys * 17) % 256
    picture[..., 3] = 255
    return picture


class SpeedSliderTest(unittest.TestCase):
    def test_the_ticks_fill_the_slider_in_order(self):
        fractions = [speed_fraction(stop) for stop in SPEED_STOPS]
        self.assertEqual(fractions, sorted(set(fractions)))
        self.assertEqual(fractions[0], 0.0)
        self.assertAlmostEqual(fractions[-1], 1.0)

    def test_the_scale_is_logarithmic(self):
        self.assertAlmostEqual(speed_fraction(1.0) - speed_fraction(0.5), speed_fraction(0.5) - speed_fraction(0.25))

    def test_every_tick_reads_back(self):
        for stop in SPEED_STOPS:
            self.assertEqual(speed_at(speed_fraction(stop)), stop)

    def test_between_ticks_the_speed_is_what_the_value_shows(self):
        between = speed_at((speed_fraction(0.25) + speed_fraction(0.5)) / 2)
        self.assertEqual(between, 0.35)
        self.assertEqual(speed_at((speed_fraction(2.0) + speed_fraction(5.0)) / 2), 3.2)

    def test_nothing_lies_between_0_and_the_slowest_speed(self):
        self.assertEqual(speed_at(STOPPED_SHARE * 0.4), 0.0)
        self.assertEqual(speed_at(STOPPED_SHARE * 0.6), SLOWEST_SPEED)

    def test_past_the_ends_stays_at_the_ends(self):
        self.assertEqual(speed_at(-0.5), 0.0)
        self.assertEqual(speed_at(1.5), FASTEST_SPEED)

    def test_the_keys_go_from_tick_to_tick(self):
        self.assertEqual(next_stop(1.0, True), 2.0)
        self.assertEqual(next_stop(1.0, False), 0.5)
        self.assertEqual(next_stop(0.37, True), 0.5)
        self.assertEqual(next_stop(0.37, False), 0.25)
        self.assertEqual(next_stop(FASTEST_SPEED, True), FASTEST_SPEED)
        self.assertEqual(next_stop(0.0, False), 0.0)


class GeometryTest(unittest.TestCase):
    def test_fit_centres_the_whole_picture(self):
        geometry = compute_geometry((1000, 800), (3840, 2160), "fit", None)
        self.assertTrue(geometry.fit)
        self.assertEqual(geometry.region, (0, 0, 3840, 2160))
        self.assertEqual(geometry.shown, (0, 119, 1000, 562))
        self.assertEqual(decode_request(geometry), (None, (1000, 562), "area"))

    def test_zoom_shows_a_clamped_region(self):
        geometry = compute_geometry((1000, 800), (3840, 2160), 2, (100, 100))
        self.assertEqual(geometry.region, (0, 0, 500, 400), "the region stays inside the picture")
        self.assertEqual(geometry.shown, (0, 0, 1000, 800))
        self.assertEqual(decode_request(geometry), ((0, 0, 500, 400), (500, 400), "neighbor"))
        geometry = compute_geometry((1000, 800), (3840, 2160), 1, (1920, 1080))
        self.assertEqual(geometry.region, (1420, 680, 1000, 800))

    def test_small_picture_zoomed_out_is_centred(self):
        geometry = compute_geometry((1000, 800), (640, 360), 1, None)
        self.assertEqual(geometry.region, (0, 0, 640, 360))
        self.assertEqual(geometry.shown, (180, 220, 640, 360))


class MinimapTest(unittest.TestCase):
    def test_tall_picture_in_a_short_view_fits(self):
        """A 9:16 output in a short view at 150% gets a narrower map that stays inside the view."""
        px = lambda value: int(round(value * 1.5))  # noqa: E731 - the theme's DPI scaling
        for view_size in ((1400, 455), (700, 300), (300, 120)):
            box = minimap_box(view_size, (2160, 3840), px)
            self.assertIsNotNone(box, view_size)
            left, top, right, bottom = box
            self.assertTrue(0 <= left < right <= view_size[0] and 0 <= top < bottom <= view_size[1], (view_size, box))
            self.assertAlmostEqual((right - left) / (bottom - top), 2160 / 3840, delta=0.05)

    def test_landscape_map_keeps_its_width(self):
        self.assertEqual(minimap_box((1600, 900), (3840, 2160), lambda value: int(round(value))),
                         (1408, 787, 1588, 888))

    def test_no_room_no_map(self):
        self.assertIsNone(minimap_box((40, 30), (3840, 2160), lambda value: int(round(value * 1.5))))


class ReferenceTest(unittest.TestCase):
    """The left of the divider must line up pixel for pixel with an upscale of the whole source."""

    def setUp(self):
        self.source = gradient(320, 180)
        self.output_size = (640, 360)

    def test_zoomed_region_matches_a_full_upscale(self):
        for interpolation in (cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC):
            full = cv2.resize(self.source, self.output_size, interpolation=interpolation)
            geometry = compute_geometry((200, 120), self.output_size, 1, (301, 151))
            x0, y0, width, height = geometry.region
            left = reference_view(self.source, geometry, interpolation, 150)
            expected = full[y0:y0 + height, x0:x0 + 150]
            difference = np.abs(left.astype(int) - expected.astype(int)).max()
            self.assertLessEqual(difference, 1, f"interpolation {interpolation}")

    def test_zoom_two_repeats_output_pixels(self):
        geometry = compute_geometry((200, 120), self.output_size, 2, (320, 180))
        left = reference_view(self.source, geometry, cv2.INTER_NEAREST, 200)
        np.testing.assert_array_equal(left[0::2, 0::2], left[1::2, 1::2])

    def test_fit_enlarging_uses_the_filter(self):
        geometry = compute_geometry((960, 540), self.output_size, "fit", None)
        left = reference_view(self.source, geometry, cv2.INTER_NEAREST, 480)
        expected = cv2.resize(self.source, (960, 540), interpolation=cv2.INTER_NEAREST)[:, :480]
        np.testing.assert_array_equal(left, expected)

    def test_original_size_copies_the_source_pixels(self):
        geometry = compute_geometry((200, 120), self.output_size, "source", (301, 151), (320, 180))
        self.assertEqual(geometry.shown[2:], (200, 120), "one view pixel per source pixel")
        x0, y0, width, height = geometry.region
        self.assertEqual((width, height), (400, 240))
        for interpolation in (cv2.INTER_NEAREST, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4):
            left = reference_view(self.source, geometry, interpolation, 200)
            np.testing.assert_array_equal(left, self.source[y0 // 2:y0 // 2 + 120, x0 // 2:x0 // 2 + 200])
        upscaled = gradient(*self.output_size)
        expected = cv2.resize(upscaled[y0:y0 + height, x0:x0 + width], (200, 120), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(upscaled_view(upscaled, geometry), expected)
        self.assertEqual(decode_request(geometry), ((x0, y0, 400, 240), (200, 120), "area"))

    def test_right_side_columns_match_a_full_width_render(self):
        """The right side computes only its own columns; they must equal the same columns of a full render."""
        cases = [(compute_geometry((960, 540), self.output_size, "fit", None), 301),
                 (compute_geometry((200, 120), self.output_size, 2, (301, 151)), 77),
                 (compute_geometry((200, 120), self.output_size, "source", (301, 151), (320, 180)), 63)]
        for geometry, first in cases:
            width = geometry.shown[2]
            for interpolation in (cv2.INTER_NEAREST, cv2.INTER_CUBIC):
                full = reference_view(self.source, geometry, interpolation, width)
                part = reference_view(self.source, geometry, interpolation, width - first, first)
                difference = np.abs(part.astype(int) - full[:, first:].astype(int)).max()
                self.assertLessEqual(difference, 1, f"{geometry.fit} {geometry.scale} from column {first}")

    def test_upscaled_view_crops_the_same_region(self):
        upscaled = gradient(*self.output_size)
        geometry = compute_geometry((200, 120), self.output_size, 1, (301, 151))
        x0, y0, width, height = geometry.region
        np.testing.assert_array_equal(upscaled_view(upscaled, geometry), upscaled[y0:y0 + height, x0:x0 + width])


class AnyZoomTest(unittest.TestCase):
    """The slider zooms to any scale: from 100% up every view pixel is a whole output pixel on both sides."""

    def setUp(self):
        self.source = gradient(320, 180)
        self.output_size = (640, 360)

    def test_upscaled_pixels_are_blocks(self):
        upscaled = gradient(*self.output_size)
        geometry = compute_geometry((300, 200), self.output_size, 1.5, (321, 181))
        x0, y0, _, _ = geometry.region
        _, _, width, height = geometry.shown
        self.assertEqual((width, height), (300, 200))
        columns = ((np.arange(width) + 0.5) / 1.5).astype(int)
        rows = ((np.arange(height) + 0.5) / 1.5).astype(int)
        np.testing.assert_array_equal(upscaled_view(upscaled, geometry), upscaled[y0 + rows][:, x0 + columns])

    def test_both_sides_line_up(self):
        full = cv2.resize(self.source, self.output_size, interpolation=cv2.INTER_NEAREST)
        for zoom in (1.5, 2.75):
            geometry = compute_geometry((300, 200), self.output_size, zoom, (301, 151))
            left = reference_view(self.source, geometry, cv2.INTER_NEAREST, geometry.shown[2])
            difference = np.abs(left.astype(int) - upscaled_view(full, geometry).astype(int)).max()
            self.assertLessEqual(difference, 1, f"zoom {zoom}")

    def test_right_side_columns_match_at_any_zoom(self):
        for zoom in (0.7, 1.37, 2.5):
            geometry = compute_geometry((200, 120), self.output_size, zoom, (301, 151))
            width = geometry.shown[2]
            for interpolation in (cv2.INTER_NEAREST, cv2.INTER_CUBIC):
                full = reference_view(self.source, geometry, interpolation, width)
                part = reference_view(self.source, geometry, interpolation, width - 61, 61)
                difference = np.abs(part.astype(int) - full[:, 61:].astype(int)).max()
                self.assertLessEqual(difference, 1, f"zoom {zoom}")

    def test_playing_draws_the_same_blocks_as_paused(self):
        """Playing decodes the shown region at its own pixels and the view draws the blocks, as when paused."""
        upscaled = gradient(*self.output_size)
        for zoom in (1.5, 2.75, 3.3):
            geometry = compute_geometry((300, 200), self.output_size, zoom, (301, 151))
            crop, size, _ = decode_request(geometry)
            self.assertEqual((crop, size), (geometry.region, geometry.region[2:]))
            x0, y0, width, height = crop
            decoded = upscaled[y0:y0 + height, x0:x0 + width]
            paused = upscaled_view(upscaled, geometry)
            right = block_view(decoded, geometry.scale, 113, geometry.shown[2] - 113, geometry.shown[3])
            np.testing.assert_array_equal(right, paused[:, 113:], f"zoom {zoom}")

    def test_a_picture_decoded_for_the_last_zoom_stays_in_place(self):
        """While the decoder restarts after a zoom or pan, its last picture is redrawn where it belongs in the new
        geometry, so the upscaled side lines up with the source side at once."""
        ys, xs = np.mgrid[0:360, 0:640]
        upscaled = np.zeros((360, 640, 4), np.uint8)
        upscaled[..., 0] = (xs * 0.35) % 256
        upscaled[..., 1] = (ys * 0.6) % 256
        upscaled[..., 3] = 255
        cases = [(compute_geometry((300, 200), (640, 360), 2, (320, 180)),
                  compute_geometry((300, 200), (640, 360), 2.5, (330, 170))),  # zoom in and pan, playing at 2x
                 (compute_geometry((300, 200), (640, 360), "fit", None),
                  compute_geometry((300, 200), (640, 360), 1.5, (300, 200)))]  # from Fit
        for old, new in cases:
            crop, size, _ = decode_request(old)
            if crop is None:
                decoded = upscaled_view(upscaled, old)
            else:
                x0, y0, width, height = crop
                decoded = upscaled[y0:y0 + height, x0:x0 + width]
                if size != (width, height):
                    decoded = upscaled_view(upscaled, old)
            shown = reframe(decoded, old, new, 40, new.shown[2] - 40)
            exact = upscaled_view(upscaled, new)[:, 40:]
            difference = np.abs(shown[..., :2].astype(int) - exact[..., :2].astype(int))
            self.assertLess(difference.mean(), 2.0, f"{old.scale} -> {new.scale}")

    def test_wheel_stops_at_whole_zooms(self):
        stops = [0.3, 0.5, 1.0, 2.0, 4.0, 8.0]
        scale, seen = 0.3, []
        for _ in range(30):
            scale = step_zoom(scale, 1, stops, 0.3, 8.0)
            seen.append(scale)
        for stop in stops[1:]:
            self.assertTrue(any(abs(value - stop) < 1e-9 for value in seen), f"stops at {stop}")
        self.assertEqual(seen[-1], 8.0)
        self.assertAlmostEqual(step_zoom(1.0, -1, stops, 0.3, 8.0), 2 ** -0.25)
        self.assertEqual(step_zoom(1.9, 1, stops, 0.3, 8.0), 2.0)
        self.assertEqual(step_zoom(0.32, -2, stops, 0.3, 8.0), 0.3)


class CompositingTest(unittest.TestCase):
    def test_blend_premultiplied_sprite(self):
        canvas = np.full((4, 4, 4), 200, np.uint8)
        sprite = np.zeros((2, 2, 4), np.uint8)
        sprite[...] = (50, 50, 50, 128)  # white at about half opacity, premultiplied would be 128; 50 is dimmer
        blend(canvas, sprite, 3, 3)
        self.assertEqual(int(canvas[3, 3, 0]), 50 + (200 * 127 + 127) // 255)
        self.assertEqual(int(canvas[0, 0, 0]), 200, "outside the sprite is untouched")

    def test_over_combines_alpha(self):
        target = np.zeros((2, 2, 4), np.uint8)
        target[...] = (0, 0, 0, 153)
        sprite = np.zeros((2, 2, 4), np.uint8)
        sprite[...] = (255, 255, 255, 255)
        over(target, sprite, 0, 0)
        self.assertEqual(target[0, 0].tolist(), [255, 255, 255, 255])

    def test_outline_from_a_small_piece_matches_one_drawn_full_size(self):
        width, height, radius = 300, 120, 12
        full = Sketch(width, height)
        full.fill(np.clip(full.rounded_rect(0, 0, width, height, radius)
                          - full.rounded_rect(1.5, 1.5, width - 1.5, height - 1.5, radius - 1.5), 0, 1), "#ff6a3d")
        expected = np.full((height + 20, width + 20, 4), 30, np.uint8)
        blend(expected, full.premultiplied_bgra(), 10, 10)
        canvas = np.full_like(expected, 30)
        blend_outline(canvas, VideoView._make_outline(radius, "#ff6a3d"), 10, 10, width, height)
        self.assertLessEqual(int(np.abs(canvas.astype(int) - expected.astype(int)).max()), 1)

    def test_format_time(self):
        self.assertEqual(format_time(0), "0:00")
        self.assertEqual(format_time(65.9), "1:05")
        self.assertEqual(format_time(3725), "1:02:05")
        self.assertEqual(format_precise(65.25), "1:05.250")
        self.assertEqual(format_precise(3725.0006), "1:02:05.001")
        self.assertEqual(format_speed(1.0), "1×")
        self.assertEqual(format_speed(-0.25), "−0.25×")
        self.assertEqual(format_speed(-0.0), "0×")


if __name__ == "__main__":
    unittest.main()
