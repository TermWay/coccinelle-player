"""Vulkan round trip on the real GPU (skipped when no Vulkan device is available)."""
import unittest

from fsr4 import selftest


class GpuTest(unittest.TestCase):
    def test_roundtrip(self):
        try:
            from fsr4.gpu import Gpu
            Gpu(verbose=False).destroy()
        except Exception as error:  # noqa: BLE001 - any failure here means no usable device
            self.skipTest(f"no Vulkan device: {error}")
        self.assertEqual(selftest.run(validate=False, verbose=False, width=256, height=128), 0)


if __name__ == "__main__":
    unittest.main()
