"""The version shown in the window and by --version: the commit and, for the executables, the build's time."""
import contextlib
import datetime
import io
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fsr4 import __version__, build_info, cli

HAS_GIT = subprocess.run(["git", "--version"], capture_output=True).returncode == 0
# A source download (GitHub's "Source code" zip, git archive) has no history to name a commit from.
IN_REPOSITORY = (build_info.SOURCE_ROOT / ".git").exists()


@unittest.skipUnless(HAS_GIT and IN_REPOSITORY, "needs git and a checkout of the repository")
class FromTheSourcesTest(unittest.TestCase):
    def test_names_the_commit_checked_out(self):
        head = subprocess.run(["git", "-C", str(build_info.SOURCE_ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        commit = build_info.commit()
        self.assertIsNotNone(commit)
        self.assertTrue(head.startswith(commit.removesuffix(build_info.CHANGED_MARK)), f"{commit} vs {head}")

    def test_says_it_runs_from_the_sources(self):
        self.assertRegex(build_info.describe(), rf"^{re.escape(__version__)} · [0-9a-f]{{7,}}( \+ changes)? · from "
                                                rf"the sources$")

    def test_outside_a_repository_the_commit_is_unknown_and_logged(self):
        lines = []
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(build_info.log, "emit", lines.append):
            self.assertIsNone(build_info.commit(Path(folder)))
            self.assertIn("unknown commit · built 2026-11-11 09:05",
                          build_info.stamp(Path(folder), datetime.datetime(2026, 11, 11, 9, 5)))
        self.assertTrue(lines and lines[0].startswith("[build] the commit is unknown"), lines)


class CommandLineTest(unittest.TestCase):
    def test_version_prints_the_same_line(self):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed), self.assertRaises(SystemExit) as raised:
            cli.build_parser().parse_args(["--version"])
        self.assertEqual(raised.exception.code, 0)
        # Outside a repository a "[build] the commit is unknown" line comes first (logged, not an error).
        self.assertEqual(printed.getvalue().splitlines()[-1], f"fsr4 {build_info.describe()}")


class PackagedTest(unittest.TestCase):
    def test_reads_the_stamp_written_by_the_build(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / build_info.BUILD_FILE).write_text("0e9ae93 · built 2026-11-11 09:05\n", encoding="utf-8")
            with mock.patch.object(sys, "frozen", True, create=True), \
                    mock.patch.object(sys, "_MEIPASS", folder, create=True):
                self.assertEqual(build_info.describe(), f"{__version__} · 0e9ae93 · built 2026-11-11 09:05")

    def test_a_missing_stamp_is_logged(self):
        lines = []
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(build_info.log, "emit", lines.append), \
                mock.patch.object(sys, "frozen", True, create=True), \
                mock.patch.object(sys, "_MEIPASS", folder, create=True):
            self.assertEqual(build_info.describe(), f"{__version__} · unknown build")
        self.assertTrue(lines and "build stamp" in lines[0], lines)


if __name__ == "__main__":
    unittest.main()
