"""The video view's geometry and the left side's alignment with the upscaled picture (no window needed)."""
import unittest

import cv2
import numpy as np

from fsr4.ui_transport import format_time
from fsr4.ui_video import blend, compute_geometry, decode_request, over, reference_view, upscaled_view


def gradient(width: int, height: int) -> np.ndarray:
    """A BGRA picture where every pixel differs from its neighbours."""
    ys, xs = np.mgrid[0:height, 0:width]
    picture = np.zeros((height, width, 4), np.uint8)
    picture[..., 0] = (xs * 7 + ys * 3) % 256
    picture[..., 1] = (xs * 13) % 256
    picture[..., 2] = (ys * 17) % 256
    picture[..., 3] = 255
    return picture


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
        self.assertEqual(decode_request(geometry), ((0, 0, 500, 400), (1000, 800), "neighbor"))
        geometry = compute_geometry((1000, 800), (3840, 2160), 1, (1920, 1080))
        self.assertEqual(geometry.region, (1420, 680, 1000, 800))

    def test_small_picture_zoomed_out_is_centred(self):
        geometry = compute_geometry((1000, 800), (640, 360), 1, None)
        self.assertEqual(geometry.region, (0, 0, 640, 360))
        self.assertEqual(geometry.shown, (180, 220, 640, 360))


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

    def test_format_time(self):
        self.assertEqual(format_time(0), "0:00")
        self.assertEqual(format_time(65.9), "1:05")
        self.assertEqual(format_time(3725), "1:02:05")


if __name__ == "__main__":
    unittest.main()
