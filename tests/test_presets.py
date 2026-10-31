"""Presets: what they change, how the user's are saved, and that a broken file is never overwritten."""
import json
import tempfile
import unittest
from pathlib import Path

from fsr4.presets import BUILTIN, PresetStore
from fsr4.render import RenderSettings


class PresetTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "presets.json"

    def tearDown(self):
        self.directory.cleanup()

    def test_a_preset_keeps_the_size_and_the_encoder(self):
        current = RenderSettings((3840, 2160), sharpness=0.3, encoder="x265")
        fsr3 = next(preset for preset in BUILTIN if preset.upscaler == "3.1.6")
        shown = fsr3.applied_to(current)
        self.assertEqual((shown.output_size, shown.encoder), ((3840, 2160), "x265"))
        self.assertEqual((shown.upscaler, shown.sharpness), ("3.1.6", None))
        self.assertTrue(fsr3.matches(shown))
        self.assertFalse(fsr3.matches(current))

    def test_saved_presets_come_back_first(self):
        store = PresetStore(self.path)
        store.save("Crisp", RenderSettings((1, 1), flow_preset="medium", sharpness=0.35))
        again = PresetStore(self.path)
        self.assertIsNone(again.load_error)
        self.assertEqual(again.all[0].name, "Crisp")
        self.assertEqual(again.matching(RenderSettings((640, 360), flow_preset="medium", sharpness=0.35)).name, "Crisp")
        again.save("crisp", RenderSettings((1, 1)))  # same name, other case: replaces it
        self.assertEqual([preset.name for preset in again.user], ["Crisp"])
        again.delete("Crisp")
        self.assertEqual(PresetStore(self.path).user, [])

    def test_names(self):
        store = PresetStore(self.path)
        self.assertIsNotNone(store.check_name("  "))
        self.assertIsNotNone(store.check_name("fsr 4"), "a built-in name")
        self.assertIsNotNone(store.check_name("x" * 41))
        self.assertIsNone(store.check_name("Mine"))
        with self.assertRaises(ValueError):
            store.save("FSR 3.1", RenderSettings((1, 1)))

    def test_an_unreadable_file_is_kept_aside(self):
        self.path.write_text('{"presets": [{"name": "Odd", "upscaler": "9.9"}]}', encoding="utf-8")
        store = PresetStore(self.path)
        self.assertIsNotNone(store.load_error)
        self.assertEqual(store.user, [])
        store.save("New", RenderSettings((1, 1)))
        self.assertIn("Odd", self.path.with_suffix(".unreadable").read_text(encoding="utf-8"))
        self.assertEqual([entry["name"] for entry in json.loads(self.path.read_text(encoding="utf-8"))["presets"]],
                         ["New"])

    def test_files_of_other_shapes_are_unreadable_not_fatal(self):
        for content in ('["abc"]', '{"presets": ["abc"]}', '{"presets": [null]}', '{"presets": {"a": 1}}', "[]"):
            self.path.write_text(content, encoding="utf-8")
            store = PresetStore(self.path)
            self.assertIsNotNone(store.load_error, content)
            self.assertEqual(store.user, [])

    def test_saving_again_keeps_the_spelling(self):
        store = PresetStore(self.path)
        store.save("Crisp", RenderSettings((1, 1), sharpness=0.3))
        saved = store.save("crisp", RenderSettings((1, 1), sharpness=0.9))
        self.assertEqual((saved.name, [preset.name for preset in store.user]), ("Crisp", ["Crisp"]))
        self.assertEqual(store.find("Crisp").sharpness, 0.9)

    def test_a_failed_write_changes_nothing(self):
        store = PresetStore(self.path)
        store.save("Kept", RenderSettings((1, 1)))
        self.path.with_suffix(".tmp").mkdir()  # the temporary file cannot be written
        with self.assertRaises(OSError):
            store.save("Lost", RenderSettings((1, 1), sharpness=0.2))
        self.assertEqual([preset.name for preset in store.user], ["Kept"])
        self.assertEqual([preset.name for preset in PresetStore(self.path).user], ["Kept"])

    def test_descriptions(self):
        self.assertEqual(RenderSettings((1, 1)).describe(), "FSR 4")
        self.assertEqual(RenderSettings((1, 1), flow="zero", sharpness=0.5, upscaler="2.3.4").describe(),
                         "FSR 2.3 · no motion · sharpening 0.50")
        self.assertEqual(RenderSettings((1, 1), flow_preset="medium").describe(), "FSR 4 · best motion")


if __name__ == "__main__":
    unittest.main()
