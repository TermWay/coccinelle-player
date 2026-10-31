"""The theme's images: opaque over their container, 9-slice middles stretched without changing the picture."""
import unittest

import numpy as np

from fsr4.ui_art import Sketch, stretch_middle


def rounded_box(width: int, height: int) -> Sketch:
    sketch = Sketch(width, height)
    sketch.fill(sketch.rounded_rect(0, 0, width, height, 6), "#2d3036")
    sketch.fill(sketch.rounded_rect(1, 1, width - 1, height - 1, 5), "#1e2024")
    return sketch


class ArtTest(unittest.TestCase):
    def test_opaque_composites_over_the_container(self):
        rgba = rounded_box(24, 24).opaque("#141518")
        self.assertTrue((rgba[..., 3] == 255).all(), "no partly transparent pixel: Tk draws those slowly")
        self.assertEqual(rgba[0, 0, :3].tolist(), [0x14, 0x15, 0x18], "outside the rounded corner: the container")
        self.assertEqual(rgba[12, 12, :3].tolist(), [0x1e, 0x20, 0x24])

    def test_stretched_middle_is_the_box_drawn_larger(self):
        stretched = stretch_middle(rounded_box(24, 24).opaque("#141518"), 320, 64)
        self.assertEqual(stretched.shape, (64, 320, 4))
        direct = rounded_box(320, 64).opaque("#141518")
        self.assertLessEqual(int(np.abs(stretched.astype(int) - direct.astype(int)).max()), 1)

    def test_never_shrinks(self):
        rgba = rounded_box(24, 24).rgba()
        self.assertIs(stretch_middle(rgba, 10, 0), rgba)


if __name__ == "__main__":
    unittest.main()
