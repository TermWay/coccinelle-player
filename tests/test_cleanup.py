"""The temp folder cleanup: only this tool's piece folders, and only those no running session holds (no GPU)."""
import tempfile
import unittest
from pathlib import Path

from fsr4 import render


class CleanupTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.handles = []

    def tearDown(self):
        for handle in self.handles:
            handle.close()
        self.directory.cleanup()

    def folder(self, name: str, marked: bool, files: tuple[str, ...] = ("piece_1_00000000.mp4",)) -> Path:
        folder = self.root / name
        folder.mkdir()
        for file in files:
            (folder / file).write_bytes(b"piece")
        if marked:
            (folder / render.SESSION_MARKER).write_text("session\n", encoding="utf-8")
        return folder

    def test_only_abandoned_marked_folders_go(self):
        foreign = self.folder("fsr4-notes", marked=False)
        abandoned = self.folder("fsr4-abandoned", marked=True)
        live = self.folder("fsr4-live", marked=True)
        self.handles.append(open(live / render.SESSION_MARKER, "a", encoding="utf-8"))  # a running session's
        self.assertEqual(render.clean_stale_directories([self.root]), [abandoned])
        self.assertFalse(abandoned.exists())
        self.assertTrue((foreign / "piece_1_00000000.mp4").is_file(), "a folder without the marker is not ours")
        self.assertTrue((live / "piece_1_00000000.mp4").is_file(), "a running session's pieces stay")

    def test_a_folder_that_cannot_be_emptied_stays_marked(self):
        busy = self.folder("fsr4-busy", marked=True)
        handle = open(busy / "piece_1_00000000.mp4", "rb")  # a player still reads it
        self.assertEqual(render.clean_stale_directories([self.root]), [])
        self.assertTrue((busy / render.SESSION_MARKER).is_file(), "marked again for the next start")
        handle.close()
        self.assertEqual(render.clean_stale_directories([self.root]), [busy])
        self.assertFalse(busy.exists())


if __name__ == "__main__":
    unittest.main()
