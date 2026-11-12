"""The sidebar's settings: what shows for each upscaler, and what the block reads back."""
import tkinter as tk
import unittest

from fsr4.hq_scaler import HQ_SCALER
from fsr4.presets import BEST, BUILTIN, Preset
from fsr4.realesrgan import REAL_ESRGAN, REAL_ESRGAN_ANIME, STILL_ONLY
from fsr4.render import UPSCALER_NAMES
from fsr4.ui_sides import MOTION_CHOICES, NO_FSR4_NOTE, NOT_HERE, UPSCALER_CHOICES, SideBlock
from fsr4.ui_theme import Theme


def laid_out(widget: tk.Misc) -> bool:
    """Whether `widget` has a place in its grid (a withdrawn window maps nothing)."""
    return bool(widget.grid_info())


def preset(upscaler: str | None) -> Preset:
    return next(entry for entry in BUILTIN if entry.upscaler == upscaler)


class SideBlockTest(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        image = tk.PhotoImage(master=self.root, width=8, height=8)
        self.block = SideBlock(self.root, Theme(self.root), 300, lambda *_: None, (image,), (image,),
                               on_changed=lambda: None, on_preset=lambda: None, on_add=lambda: None,
                               on_delete=lambda: None, on_name=lambda: None, on_cancel_name=lambda _event=None: None)
        self.block.grid(row=0, column=0)

    def tearDown(self):
        self.root.destroy()

    def show(self, how: Preset, image: bool = False) -> None:
        self.block.load(how, image)
        self.block.refresh(True, image, True, "A preset", False)

    def rows(self) -> tuple[bool, bool]:
        """Whether the motion and the sharpening are shown, each with everything that goes with it."""
        block = self.block
        motion = [laid_out(widget) for widget in (block.motion_title, block.motion_choice, block.motion_note)]
        sharpening = [laid_out(widget) for widget in (block.sharpen_title, block.sharpen_row)]
        self.assertEqual(len(set(motion)), 1)
        self.assertEqual(len(set(sharpening)), 1)
        return motion[0], sharpening[0]

    def test_the_preset_and_the_upscaler_always_show(self):
        for upscaler in UPSCALER_CHOICES:
            self.show(Preset("", upscaler, "zero", "fast"), image=True)
            self.assertTrue(laid_out(self.block.preset_choice) and laid_out(self.block.upscaler_choice))

    def test_fsr_on_a_video_shows_everything_at_its_best(self):
        self.show(BEST)
        self.assertEqual(self.rows(), (True, True))
        self.assertEqual(self.block.how(), Preset("", None, "dis", "medium"))
        self.assertEqual(self.block.motion_choice.get(), "Optical flow · medium, best")
        self.assertEqual(self.block.upscaler_choice.get(), "FSR 4 · machine learning")
        self.assertEqual(self.block.sharpness_label.cget("text"), "off")

    def test_older_fsr_shows_the_same_rows(self):
        for upscaler in ("3.1.6", "2.3.4"):
            self.show(preset(upscaler))
            self.assertEqual(self.rows(), (True, True))

    def test_fsr_on_an_image_has_no_motion(self):
        self.show(BEST, image=True)
        self.assertEqual(self.rows(), (False, True))

    def test_the_hq_scaler_has_only_sharpening(self):
        self.show(preset(HQ_SCALER))
        self.assertEqual(self.rows(), (False, True))
        self.block.sharpen_var.set(True)
        self.block.sharpness_var.set(0.3)
        self.assertEqual(self.block.how(), Preset("", HQ_SCALER, "zero", "fast", 0.3))

    def test_real_esrgan_has_neither(self):
        self.show(preset(REAL_ESRGAN))
        self.assertEqual(self.rows(), (False, False))
        self.block.sharpen_var.set(True)
        self.assertEqual(self.block.how(), Preset("", REAL_ESRGAN, "zero", "fast"))

    def test_the_rows_follow_the_upscaler_picked_in_the_list(self):
        self.show(BEST)
        self.block.upscaler_choice.current(self.block.upscalers.index(REAL_ESRGAN))
        self.block.refresh(True, False, True, "A preset", False)
        self.assertEqual(self.rows(), (False, False))
        self.block.upscaler_choice.current(self.block.upscalers.index(None))
        self.block.refresh(True, False, True, "A preset", False)
        self.assertEqual(self.rows(), (True, True))

    def test_going_through_an_upscaler_without_motion_keeps_the_motion(self):
        self.show(BEST)
        self.block.load(preset(HQ_SCALER), False)
        self.block.load(self.block.how(), False)
        self.block.upscaler_choice.current(self.block.upscalers.index(None))
        self.assertEqual(self.block.motion_choice.get(), "Optical flow · medium, best")

    def test_a_model_for_still_images_leaves_a_video(self):
        self.show(preset(REAL_ESRGAN_ANIME), image=True)
        self.assertEqual(self.block.how().upscaler, REAL_ESRGAN_ANIME)
        self.show(self.block.how())
        self.assertEqual(self.block.how().upscaler, REAL_ESRGAN)
        self.assertFalse(STILL_ONLY & set(self.block.upscalers))

    def test_the_block_goes_when_no_side_is_upscaled(self):
        self.show(BEST)
        self.block.refresh(False, False, True, "A preset", False)
        self.assertFalse(laid_out(self.block))

    def test_the_bin_is_for_the_users_own_presets(self):
        self.show(BEST)
        self.assertFalse(laid_out(self.block.delete_preset_button))
        self.block.refresh(True, False, True, "Mine", True)
        self.assertTrue(laid_out(self.block.delete_preset_button))
        self.assertEqual(self.block.preset_choice.get(), "Mine")

    def test_a_preset_comes_back_as_it_was_saved(self):
        for flow, flow_preset in [(flow, speed) for _, flow, speed, _ in MOTION_CHOICES]:
            for upscaler in (None, "3.1.6", "2.3.4"):
                saved = Preset("", upscaler, flow, flow_preset, 0.65)
                self.show(saved)
                self.assertEqual(self.block.how(), saved)

    def test_the_settings_lock_during_work(self):
        self.show(BEST)
        self.block.refresh(True, False, False, "A preset", False)
        for widget in (self.block.preset_choice, self.block.upscaler_choice, self.block.motion_choice,
                       self.block.sharpen_switch, self.block.sharpness_scale, self.block.save_preset_button):
            self.assertTrue(widget.instate(["disabled"]))

    def test_fsr4_is_marked_on_a_card_without_it(self):
        self.show(BEST)
        self.assertEqual(self.block.upscaler_choice.get(), UPSCALER_CHOICES[None][0])
        self.block.fsr4_missing = True
        self.show(BEST)
        self.assertEqual(self.block.upscaler_choice.get(), f"FSR 4 · {NOT_HERE}")
        self.assertEqual(self.block.upscaler_note.cget("text"), NO_FSR4_NOTE)
        self.assertIsNone(self.block.how().upscaler)  # still reads back as FSR 4: the window refuses it, not the block
        self.show(preset("3.1.6"))
        self.assertEqual(self.block.upscaler_choice.get(), UPSCALER_CHOICES["3.1.6"][0])
        self.assertEqual(self.block.upscaler_note.cget("text"), UPSCALER_CHOICES["3.1.6"][1])

    def test_every_upscaler_has_a_label_and_a_note(self):
        self.assertCountEqual(UPSCALER_CHOICES, UPSCALER_NAMES)


if __name__ == "__main__":
    unittest.main()
