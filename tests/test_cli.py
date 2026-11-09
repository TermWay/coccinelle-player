"""The command line around a job: an error starts a line of its own (exit code 2 for a wrong request, 1 for a
failure), and the optical flow gets at most 8 threads (no GPU needed: a stand-in replaces the job)."""
import contextlib
import io
import unittest
from pathlib import Path
from unittest import mock

import needs
from fsr4 import cli, usage
from fsr4.job import JobResult


def failing_job(error: Exception):
    """A job that reports progress, which leaves the progress line open, then raises `error`."""
    def run(options, progress, **_):
        progress(10, 100, 30.0)
        raise error
    return run


class CommandLineTest(unittest.TestCase):
    def setUp(self):
        previous = usage.current()
        self.addCleanup(usage.apply, previous.threads, previous.priority)  # the command line applies its own

    def upscale(self, job) -> tuple[int, str]:
        """Run `fsr4 upscale clip.mp4` with `job` in place of the real one; return the exit code and stderr."""
        stderr = io.StringIO()
        with mock.patch.object(cli, "run_job", job), contextlib.redirect_stderr(stderr):
            code = cli.main(["upscale", "clip.mp4"])
        return code, stderr.getvalue()

    def test_an_error_starts_a_line_of_its_own(self):
        for error, expected_code in ((ValueError("sharpening 2 is not between 0 and 1"), 2),
                                     (RuntimeError("ffmpeg failed"), 1)):
            with self.subTest(error=error):
                code, stderr = self.upscale(failing_job(error))
                self.assertEqual(code, expected_code)
                self.assertTrue(stderr.endswith(f" s left   \nfsr4: {error}\n"), repr(stderr))

    @needs.opencv
    def test_optical_flow_gets_at_most_8_threads(self):
        import cv2
        threads = []

        def job(options, progress, **_):
            threads.append(cv2.getNumThreads())
            return JobResult(output=Path("clip.fsr4.mp4"), frames=1, seconds=0.1, cancelled=False, timings="")

        self.assertEqual(self.upscale(job), (0, ""))
        self.assertEqual(threads, [min(8, usage.CPU_THREADS)])


if __name__ == "__main__":
    unittest.main()
