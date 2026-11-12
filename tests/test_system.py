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



class PlacementTest(unittest.TestCase):
    """Where a window left on some monitor comes back (`ui_window.placement_on_screen`)."""

    SCREENS = [(0, 0, 1920, 1080, True), (-2560, -200, 2560, 1440, False)]

    def test_a_place_on_a_monitor_is_kept(self):
        for place in ((100, 50, 1280, 860), (-2400, -100, 1600, 900)):
            self.assertEqual(ui_window.placement_on_screen(place, self.SCREENS, 32), place)

    def test_a_window_hanging_off_its_monitor_is_moved_onto_it(self):
        self.assertEqual(ui_window.placement_on_screen((1000, 900, 1280, 860), self.SCREENS, 32), (640, 220, 1280, 860))

    def test_a_window_larger_than_its_monitor_is_made_to_fit(self):
        self.assertEqual(ui_window.placement_on_screen((10, 10, 2560, 1600), self.SCREENS, 32), (0, 0, 1920, 1080))

    def test_a_monitor_that_is_gone_gives_nothing(self):
        self.assertIsNone(ui_window.placement_on_screen((4000, 100, 1280, 860), self.SCREENS, 32))
        self.assertIsNone(ui_window.placement_on_screen((100, -400, 1280, 860), self.SCREENS[:1], 32))


if __name__ == "__main__":
    unittest.main()
