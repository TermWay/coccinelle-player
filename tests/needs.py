"""What tests need from the machine: without it they skip, saying why, instead of failing.

ffmpeg and ffprobe on PATH, OpenCV (`opencv-python`), and FSR 4 itself: an AMD Radeon RX 9000 card, a Vulkan
driver and the DLLs in `bin/` (`python tools/fetch_sdk.py`).
"""
import functools
import importlib.util
import shutil
import unittest

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
HAS_OPENCV = importlib.util.find_spec("cv2") is not None
ffmpeg = unittest.skipUnless(HAS_FFMPEG, "ffmpeg and ffprobe are not on PATH")
opencv = unittest.skipUnless(HAS_OPENCV, "opencv-python is not installed")


@functools.cache
def fsr_unavailable() -> str | None:
    """Why FSR 4 cannot run on this machine, or None: the window's GPU check (a small upscaler created)."""
    try:
        from fsr4.ffx import FFX_UPSCALE_ENABLE_AUTO_EXPOSURE, Upscaler, find_sdk_dir
        from fsr4.gpu import Gpu
        gpu = Gpu(verbose=False)
        try:
            Upscaler(gpu, (256, 144), (512, 288), FFX_UPSCALE_ENABLE_AUTO_EXPOSURE, find_sdk_dir(None)).destroy()
        finally:
            gpu.destroy()
    except Exception as error:  # noqa: BLE001 - any failure means no FSR here; it is the skip's reason
        return str(error)
    return None


def require_fsr() -> None:
    """Skip the calling test (or class, from `setUpClass`) unless FSR 4 runs on this machine."""
    reason = fsr_unavailable()
    if reason is not None:
        raise unittest.SkipTest(f"FSR 4 cannot run here: {reason}")
