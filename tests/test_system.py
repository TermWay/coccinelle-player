"""What the window shows about the machine, and the monitors full screen chooses from (Windows)."""
import unittest

from fsr4 import system, ui_window


class MachineTest(unittest.TestCase):
    def test_processor_and_memory(self):
        name = system.processor_name()
        self.assertTrue(name and "(R)" not in name and "Processor" not in name, name)
        used, total = system.main_memory()
        self.assertGreater(used, 0)
        self.assertGreater(total, used)

    def test_graphics_card_memory(self):
        video = system.VideoMemory("")  # no name given: the card with the most memory
        try:
            used, total = video.read()
            if total == 0:
                self.skipTest(f"{video.description} has no memory of its own")
            self.assertGreater(total, 0)
            self.assertGreaterEqual(used, 0)
        finally:
            video.close()

    def test_monitors(self):
        screens = ui_window.monitors()
        self.assertEqual(sum(primary for *_, primary in screens), 1, "one primary monitor")
        self.assertTrue(all(width > 0 and height > 0 for _, _, width, height, _ in screens))


if __name__ == "__main__":
    unittest.main()
