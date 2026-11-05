"""A side's settings in the sidebar: what shows for each upscaler, and what the block reads back."""
import tkinter as tk
import unittest

from fsr4.hq_scaler import HQ_SCALER
from fsr4.presets import FAMILIES, FAMILY_BY_NAME, Preset
from fsr4.realesrgan import REAL_ESRGAN, REAL_ESRGAN_ANIME, REAL_ESRGAN_X4PLUS
from fsr4.ui_sides import MOTION_CHOICES, UPSCALER_CHOICES, SideBlock
from fsr4.ui_theme import Theme


def laid_out(widget: tk.Misc) -> bool:
    """Whether `widget` has a place in its grid (a withdrawn window maps nothing)."""
    return bool(widget.grid_info())


class SideBlockTest(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        image = tk.PhotoImage(master=self.root, width=8, height=8)
        self.block = SideBlock(self.root, Theme(self.root), "Right", 300, lambda *_: None, (image,), (image,),
                               on_changed=lambda: None, on_add=lambda: None, on_delete=lambda: None,
                               on_name=lambda: None, on_cancel_name=lambda _event=None: None)
        self.block.grid(row=0, column=0)

    def tearDown(self):
        self.root.destroy()

    def show(self, family: str, image: bool = False, still: bool = False, how: Preset | None = None) -> None:
        self.block.load(FAMILY_BY_NAME[family], how or FAMILY_BY_NAME[family].best, still or image)
        self.block.refresh(True, image, True, False)

    def test_fsr_on_a_video_shows_everything_at_its_best(self):
        self.show("FSR")
        block = self.block
        self.assertTrue(laid_out(block) and laid_out(block.version_choice) and laid_out(block.motion_choice)
                        and laid_out(block.sharpen_row))
        self.assertEqual(block.how(), FAMILY_BY_NAME["FSR"].best)
        self.assertEqual(block.motion_choice.get(), "Optical flow · medium, best")
        self.assertEqual(block.version_choice.get(), "FSR 4 · machine learning")
        self.assertEqual(block.sharpness_label.cget("text"), "off")

    def test_fsr_on_an_image_has_no_motion(self):
        self.show("FSR", image=True)
        self.assertTrue(laid_out(self.block.version_choice))
        self.assertFalse(laid_out(self.block.motion_choice) or laid_out(self.block.motion_note))

    def test_the_hq_scaler_has_only_sharpening(self):
        self.show("AMD HQ scaler")
        block = self.block
        self.assertTrue(laid_out(block) and laid_out(block.sharpen_row))
        self.assertFalse(laid_out(block.version_choice) or laid_out(block.motion_choice))
        block.sharpen_var.set(True)
        block.sharpness_var.set(0.3)
        self.assertEqual(block.how(), Preset("AMD HQ scaler", HQ_SCALER, "zero", "fast", 0.3))

    def test_real_esrgan_on_a_video_shows_nothing(self):
        self.show("Real-ESRGAN")
        self.assertFalse(laid_out(self.block))
        self.assertEqual(self.block.how(), FAMILY_BY_NAME["Real-ESRGAN"].best)

    def test_real_esrgan_on_an_image_offers_its_models(self):
        self.show("Real-ESRGAN", image=True)
        block = self.block
        self.assertTrue(laid_out(block) and laid_out(block.version_choice))
        self.assertFalse(laid_out(block.motion_choice) or laid_out(block.sharpen_row))
        self.assertEqual(block.version_label.cget("text"), "Model")
        self.assertEqual(block.upscalers, [REAL_ESRGAN, REAL_ESRGAN_X4PLUS, REAL_ESRGAN_ANIME])

    def test_a_model_for_still_images_leaves_a_video(self):
        anime = Preset("Mine", REAL_ESRGAN_ANIME, "zero", "fast")
        self.show("Real-ESRGAN", image=True, how=anime)
        self.assertEqual(self.block.how().upscaler, REAL_ESRGAN_ANIME)
        self.show("Real-ESRGAN", how=self.block.how())
        self.assertEqual(self.block.how().upscaler, REAL_ESRGAN)

    def test_a_side_that_shows_no_upscaler_shows_no_settings(self):
        self.show("FSR")
        self.block.refresh(False, False, True, False)
        self.assertFalse(laid_out(self.block))

    def test_a_preset_comes_back_as_it_was_saved(self):
        for flow, flow_preset in [(flow, preset) for _, flow, preset, _ in MOTION_CHOICES]:
            for upscaler in FAMILIES[0].upscalers:
                preset = Preset("FSR", upscaler, flow, flow_preset, 0.65)
                self.show("FSR", how=preset)
                self.assertEqual(self.block.how(), preset)

    def test_the_settings_lock_during_work(self):
        self.show("FSR")
        self.block.refresh(True, False, False, False)
        for widget in (self.block.version_choice, self.block.motion_choice, self.block.sharpen_switch,
                       self.block.sharpness_scale, self.block.save_preset_button):
            self.assertTrue(widget.instate(["disabled"]))

    def test_every_upscaler_has_a_label_and_a_note(self):
        for family in FAMILIES:
            for upscaler in family.upscalers:
                self.assertEqual(len(UPSCALER_CHOICES[upscaler]), 2)


if __name__ == "__main__":
    unittest.main()
