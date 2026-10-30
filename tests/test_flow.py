"""Optical flow must follow FSR's convention: vector = previous position - current position, in pixels."""
import unittest

import numpy as np

from fsr4 import flow


def textured_image(width: int, height: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(height // 4, width // 4), dtype=np.uint8)
    gray = np.kron(noise, np.ones((4, 4), dtype=np.uint8))
    rgba = np.stack([gray, gray, gray, np.full_like(gray, 255)], axis=-1)
    return np.ascontiguousarray(rgba)


class FlowTest(unittest.TestCase):
    def test_zero_flow(self):
        provider = flow.create_flow("zero", 32, 16)
        vectors = provider.compute(np.zeros((16, 32, 4), dtype=np.uint8))
        self.assertEqual(vectors.shape, (16, 32, 2))
        self.assertEqual(vectors.dtype, np.float16)
        self.assertEqual(float(np.abs(vectors).max()), 0.0)

    def test_dis_flow_sign_and_magnitude(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("opencv-python not installed")
        width, height, shift = 256, 128, 5
        previous = textured_image(width + 2 * shift, height)
        current = np.ascontiguousarray(previous[:, 2 * shift:2 * shift + width])
        previous = np.ascontiguousarray(previous[:, shift:shift + width])
        # The content in `current` sits `shift` pixels further left than in `previous`:
        # a pixel at x in current was at x + shift in previous, so FSR's vector is (+shift, 0).
        provider = flow.create_flow("dis", width, height, preset="medium")
        first = provider.compute(previous)
        self.assertEqual(float(np.abs(first).max()), 0.0, "first frame has no previous frame")
        vectors = provider.compute(current).astype(np.float32)
        center = vectors[height // 4: -height // 4, width // 4: -width // 4]
        self.assertAlmostEqual(float(np.median(center[..., 0])), shift, delta=0.75)
        self.assertAlmostEqual(float(np.median(center[..., 1])), 0.0, delta=0.5)

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            flow.create_flow("magic", 8, 8)


if __name__ == "__main__":
    unittest.main()
