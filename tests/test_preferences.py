"""The window's preferences file and the temporary folder they choose (no GPU)."""
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
        self.assertEqual(store.preferences, Preferences(upscale_on_open=True, temporary_folder=None))
        self.assertIsNone(store.load_error)
        self.assertFalse(self.path.exists(), "reading never writes")

    def test_a_change_is_kept_for_the_next_run(self):
        PreferenceStore(self.path).change(upscale_on_open=False, temporary_folder=self.root / "pieces")
        self.assertEqual(PreferenceStore(self.path).preferences, Preferences(False, self.root / "pieces"))
        PreferenceStore(self.path).change(temporary_folder=None)
        self.assertEqual(PreferenceStore(self.path).preferences, Preferences(False, None))

    def test_an_unreadable_file_gives_the_defaults_and_is_kept_aside(self):
        for content in ("{not json", '{"upscale_on_open": "yes"}', '{"temporary_folder": 3}', "[]"):
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
        self.root = Path(directory.name)
        previous = temporary.folder()
        self.addCleanup(temporary.use, previous)

    def test_a_new_folder_is_made_and_used(self):
        folder = (self.root / "a" / "b").resolve()
        self.assertEqual(temporary.use(folder), folder)
        self.assertEqual(temporary.folder(), folder)
        self.assertEqual(list(folder.iterdir()), [], "the write check leaves nothing behind")
        self.assertIn(folder, temporary.folders_to_clean())

    def test_a_folder_that_cannot_be_made_keeps_the_previous_one(self):
        self.root = temporary.use(self.root)
        blocker = self.root / "a file"
        blocker.write_text("not a folder", encoding="utf-8")
        with self.assertRaises(OSError):
            temporary.use(blocker / "pieces")
        self.assertEqual(temporary.folder(), self.root)

    def test_the_default_is_next_to_the_program(self):
        self.assertEqual(temporary.default_folder(), Path(__file__).resolve().parent.parent / temporary.DEFAULT_NAME)


if __name__ == "__main__":
    unittest.main()
