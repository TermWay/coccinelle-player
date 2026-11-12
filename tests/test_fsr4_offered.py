"""FSR 4 is asked for, never quietly replaced: on a card the DLL offers no FSR 4 for (anything but RDNA 4), the
DLL would run its newest upscaler, FSR 3.1, so the upscaler refuses and the command line says what runs instead.

The refusal on a real card needs a Vulkan device without FSR 4, such as the integrated Radeon of a Ryzen
processor beside an RX 9000; without one it skips."""
import contextlib
import io
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import needs
from fsr4 import cli
from fsr4.ffx import NoFsr4Error, offers_fsr4


class Fsr4OfferedTest(unittest.TestCase):
    def test_only_a_version_4_counts(self):
        self.assertTrue(offers_fsr4(["4.1.2", "3.1.6", "2.3.4"]))
        self.assertTrue(offers_fsr4(["4.0.0"]))
        self.assertFalse(offers_fsr4(["3.1.6", "2.3.4"]))
        self.assertFalse(offers_fsr4([]))
        self.assertFalse(offers_fsr4(["14.0.0"]))

    def test_the_reason_names_the_card_and_what_it_offers(self):
        error = NoFsr4Error("AMD Radeon RX 7900 XTX", ["3.1.6", "2.3.4"])
        self.assertEqual(str(error), "FSR 4 needs a Radeon RX 9000 card: the FSR DLL offers only FSR 3.1.6 and "
                                     "FSR 2.3.4 on the AMD Radeon RX 7900 XTX")
        self.assertEqual(error.versions, ["3.1.6", "2.3.4"])
        self.assertIsInstance(error, ValueError)  # a wrong request for this card: exit code 2

    def test_the_command_line_says_what_runs_instead(self):
        def refuse(options, progress, **_):
            raise NoFsr4Error("AMD Radeon RX 7900 XTX", ["3.1.6", "2.3.4"])
        stderr = io.StringIO()
        with mock.patch.object(cli, "run_job", refuse), contextlib.redirect_stderr(stderr):
            code = cli.main(["upscale", "clip.mp4"])
        self.assertEqual(code, 2)
        self.assertEqual(stderr.getvalue(), "fsr4: FSR 4 needs a Radeon RX 9000 card: the FSR DLL offers only "
                                            "FSR 3.1.6 and FSR 2.3.4 on the AMD Radeon RX 7900 XTX; "
                                            "--upscaler 3.1 or 2.3 runs one of those\n")


def device_without_fsr4() -> tuple[int, str] | None:
    """The index and name of a Vulkan device the FSR DLL offers no FSR 4 for, or None (no such device, no Vulkan,
    no DLLs)."""
    try:
        from fsr4.ffx import FFX_UPSCALE_ENABLE_AUTO_EXPOSURE, Upscaler, find_sdk_dir
        from fsr4.gpu import Gpu
        sdk_dir = find_sdk_dir(None)
        probe = Gpu(verbose=False)
        try:
            count = len(probe.physical_devices())
        finally:
            probe.destroy()
    except Exception:  # noqa: BLE001 - no Vulkan or no DLLs: nothing to look for; the tests skip
        return None
    for index in range(count):
        gpu = Gpu(gpu_index=index, verbose=False)
        try:
            Upscaler(gpu, (256, 144), (512, 288), FFX_UPSCALE_ENABLE_AUTO_EXPOSURE, sdk_dir).destroy()
        except NoFsr4Error:
            return index, gpu.device_name
        finally:
            gpu.destroy()
    return None


class CardWithoutFsr4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        found = device_without_fsr4()
        if found is None:
            raise unittest.SkipTest("no Vulkan device here lacks FSR 4 (or no Vulkan, or no FSR DLLs)")
        cls.index, cls.name = found

    def test_fsr4_is_refused_and_the_versions_offered_still_run(self):
        from fsr4.ffx import FFX_UPSCALE_ENABLE_AUTO_EXPOSURE, Upscaler, find_sdk_dir
        from fsr4.gpu import Gpu
        gpu = Gpu(gpu_index=self.index, verbose=False)
        try:
            with self.assertRaises(NoFsr4Error) as refused:
                Upscaler(gpu, (256, 144), (512, 288), FFX_UPSCALE_ENABLE_AUTO_EXPOSURE, find_sdk_dir(None))
            self.assertIn(self.name, str(refused.exception))
            self.assertTrue(refused.exception.versions)
            newest = refused.exception.versions[0]
            upscaler = Upscaler(gpu, (256, 144), (512, 288), FFX_UPSCALE_ENABLE_AUTO_EXPOSURE, find_sdk_dir(None),
                                provider=newest)
            self.assertEqual(upscaler.version_name, newest)
            upscaler.destroy()
        finally:
            gpu.destroy()

    @needs.ffmpeg
    def test_the_command_line_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            picture, output = Path(folder) / "picture.png", Path(folder) / "picture.out.png"
            subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90",
                            "-frames:v", "1", str(picture)], check=True)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = cli.main(["upscale", str(picture), "--gpu", str(self.index), "-q", "-o", str(output)])
            self.assertEqual(code, 2)
            self.assertIn("FSR 4 needs a Radeon RX 9000 card", stderr.getvalue())
            self.assertEqual(sorted(path.name for path in Path(folder).iterdir()), ["picture.png"])


if __name__ == "__main__":
    unittest.main()
