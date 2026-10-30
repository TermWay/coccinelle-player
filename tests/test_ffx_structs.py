"""Pin the ctypes FSR API structures to the MSVC x64 layout of the SDK headers."""
import ctypes
import unittest

from fsr4 import ffx


class StructLayoutTest(unittest.TestCase):
    def test_sizes(self):
        expected = {
            ffx.ffxApiHeader: 16,
            ffx.FfxApiDimensions2D: 8,
            ffx.FfxApiFloatCoords2D: 8,
            ffx.FfxApiResourceDescription: 32,
            ffx.FfxApiResource: 48,
            ffx.VkDeviceContext: 24,
            ffx.ffxCreateBackendVKDesc: 40,
            ffx.ffxCreateContextDescUpscale: 48,
            ffx.ffxCreateContextDescUpscaleVersion: 24,
            ffx.ffxDispatchDescUpscale: 432,
            ffx.ffxQueryDescGetVersions: 56,
            ffx.ffxQueryGetProviderVersion: 32,
            ffx.ffxOverrideVersion: 24,
            ffx.ffxConfigureDescGlobalDebug: 40,
            ffx.ffxQueryDescUpscaleGetJitterPhaseCount: 32,
            ffx.ffxQueryDescUpscaleGetJitterOffset: 40,
            ffx.ffxQueryDescUpscaleGetGPUMemoryUsage: 24,
        }
        for structure, size in expected.items():
            with self.subTest(structure=structure.__name__):
                self.assertEqual(ctypes.sizeof(structure), size)

    def test_dispatch_offsets(self):
        self.assertEqual(ffx.ffxDispatchDescUpscale.commandList.offset, 16)
        self.assertEqual(ffx.ffxDispatchDescUpscale.color.offset, 24)
        self.assertEqual(ffx.ffxDispatchDescUpscale.output.offset, 24 + 6 * 48)
        self.assertEqual(ffx.ffxDispatchDescUpscale.jitterOffset.offset, 360)
        self.assertEqual(ffx.ffxDispatchDescUpscale.enableSharpening.offset, 392)
        self.assertEqual(ffx.ffxDispatchDescUpscale.sharpness.offset, 396)
        self.assertEqual(ffx.ffxDispatchDescUpscale.reset.offset, 408)
        self.assertEqual(ffx.ffxDispatchDescUpscale.cameraNear.offset, 412)
        self.assertEqual(ffx.ffxDispatchDescUpscale.flags.offset, 428)

    def test_constants(self):
        self.assertEqual(ffx.FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE, 0x00010000)
        self.assertEqual(ffx.FFX_API_DISPATCH_DESC_TYPE_UPSCALE, 0x00010001)
        self.assertEqual(ffx.FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK, 0x01000003)
        self.assertEqual(ffx.FFX_UPSCALER_VERSION, (4 << 22) | (1 << 12) | 2)
        self.assertEqual(ffx.FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ, 12)


if __name__ == "__main__":
    unittest.main()
