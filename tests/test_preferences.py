"""The window's preferences file, and where temporary files go: beside the output (no GPU)."""
import json
import tempfile
import unittest
from pathlib import Path

from fsr4 import temporary
from fsr4.preferences import Preferences, PreferenceStore


class PreferencesTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "preferences.json"

    def test_defaults_without_a_file(self):
        store = PreferenceStore(self.path)
        self.assertEqual(store.preferences, Preferences(upscale_on_open=True))
        self.assertIsNone(store.load_error)
        self.assertFalse(self.path.exists(), "reading never writes")

    def test_a_change_is_kept_for_the_next_run(self):
        PreferenceStore(self.path).change(upscale_on_open=False)
        self.assertEqual(PreferenceStore(self.path).preferences, Preferences(False))

    def test_a_file_with_a_field_this_version_dropped_still_reads(self):
        self.path.write_text('{"version": 1, "upscale_on_open": false, "temporary_folder": "D:\\\\pieces"}',
                             encoding="utf-8")
        store = PreferenceStore(self.path)
        self.assertIsNone(store.load_error)
        self.assertEqual(store.preferences, Preferences(False))
        store.change(upscale_on_open=True)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), {"version": 1, "upscale_on_open": True})

    def test_an_unreadable_file_gives_the_defaults_and_is_kept_aside(self):
        for content in ("{not json", '{"upscale_on_open": "yes"}', "[]"):
            with self.subTest(content=content):
                self.path.write_text(content, encoding="utf-8")
                store = PreferenceStore(self.path)
                self.assertIsNotNone(store.load_error)
                self.assertEqual(store.preferences, Preferences())
                store.change(upscale_on_open=False)
                self.assertEqual(self.path.with_suffix(".unreadable").read_text(encoding="utf-8"), content)
                self.assertFalse(json.loads(self.path.read_text(encoding="utf-8"))["upscale_on_open"])


class TemporaryFolderTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.addCleanup(temporary.use, temporary.folder())

    def test_files_go_beside_the_output(self):
        output = self.root / "clip.fsr4.mp4"
        self.assertEqual(temporary.use_beside(output), self.root)
        self.assertEqual(temporary.folder(), self.root)
        self.assertEqual(list(self.root.iterdir()), [], "the write check leaves nothing behind")
        self.assertIn(self.root, temporary.folders_to_clean())

    def test_an_output_folder_that_cannot_take_files_falls_back_to_the_system_one(self):
        blocker = self.root / "a file"
        blocker.write_text("not a folder", encoding="utf-8")
        self.assertEqual(temporary.use_beside(blocker / "clip.fsr4.mp4"), temporary.system_folder().resolve())
        missing = self.root / "not made yet" / "clip.fsr4.mp4"
        self.assertEqual(temporary.use_beside(missing), temporary.system_folder().resolve())
        self.assertFalse(missing.parent.exists(), "an output folder is never made for temporary files")

    def test_use_keeps_the_previous_folder_when_the_new_one_fails(self):
        temporary.use(self.root)
        with self.assertRaises(OSError):
            temporary.use(self.root / "missing")
        self.assertEqual(temporary.folder(), self.root)


if __name__ == "__main__":
    unittest.main()
