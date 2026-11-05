"""What can be set for the upscaler one side of the picture shows: a block of the sidebar, absent when there is nothing.

The block only holds widgets: it reads back as a `Preset` (`how`) and takes one (`load`). What a side shows, what is
saved and the user's presets are the window's business.
"""
from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from .hq_scaler import HQ_SCALER
from .presets import FAMILIES, Family, Preset
from .realesrgan import REAL_ESRGAN, REAL_ESRGAN_ANIME, REAL_ESRGAN_X4PLUS, STILL_ONLY
from .ui_theme import Theme

DEFAULT_SHARPNESS = 0.4
UPSCALER_CHOICES = {
    None: ("FSR 4 · machine learning",
           "AMD's newest upscaler, a neural network made for RDNA 4. The best results here."),
    "3.1.6": ("FSR 3.1 · older, hand-tuned",
              "AMD's previous upscaler, kept in the same DLL. Softer and less stable on video; here to compare."),
    "2.3.4": ("FSR 2.3 · older, hand-tuned", "The upscaler before that, also in the DLL. Here to compare."),
    HQ_SCALER: ("AMD HQ scaler",
                "AMD's video scaler, run by ffmpeg 8.0 or later. Faster than FSR 4 with motion, but each frame stands "
                "alone: it gathers no detail across frames."),
    REAL_ESRGAN: ("Video model · fastest",
                  "A neural network that draws the picture again: crisp, and it can change faces and small text. A "
                  "separate program: see the README."),
    REAL_ESRGAN_X4PLUS: ("x4plus · general, slow",
                         "Its general model: about 10 s for a 1080p picture. It can change faces and small text."),
    REAL_ESRGAN_ANIME: ("anime · drawings, slow",
                        "Its model for drawings: about 4 s for a 1080p picture, and it keeps the colours better."),
}
VERSION_CAPTIONS = {"Real-ESRGAN": "Model"}
VERSION_TIP = ("AMD's FSR DLL holds three upscalers. FSR 4 is the one this tool is for; FSR 3.1 and 2.3 are there to "
               "compare. Real-ESRGAN's two slower models are for still images. Measured on a test clip (PSNR against "
               "the original, higher is better): FSR 4 26.8 dB, FSR 2.3 26.5, Lanczos 26.4, HQ scaler 26.3, FSR 3.1 "
               "25.9, Real-ESRGAN 24.4 (crisper, but what it draws is not the original). To see two side by side, "
               "show an upscaler on each side of the picture.")
MOTION_CHOICES = [
    ("Optical flow · medium, best", "dis", "medium",
     "The most accurate motion, so FSR keeps the most detail from earlier frames. About 20 frames a second at 4K."),
    ("Optical flow · fast", "dis", "fast",
     "Follows how things move between frames, a little less exactly, at twice the speed: about 45 frames a second "
     "at 4K."),
    ("Optical flow · ultrafast", "dis", "ultrafast",
     "A rougher motion estimate, a little faster still. Fine moving detail can come out softer."),
    ("Off", "zero", "fast",
     "No motion: FSR mostly works on each frame alone. Fine for still shots, moving parts can smear."),
]
MOTION_TIP = ("FSR 4 was made for games, which tell it how every pixel moved. A video does not, so the motion is "
              "measured with optical flow. Better motion lets FSR build detail from several frames; faster settings "
              "trade some of it.")
SHARPEN_TIP = ("The upscaler's own sharpening (FSR's RCAS, or the HQ scaler's). Off keeps its natural look; 0.2 to 0.5 "
               "makes edges crisper; high values also sharpen noise and can leave halos.")


def show_widget(widget: tk.Misc, shown: bool) -> None:
    """Grid `widget` where it was, or take it out of the layout."""
    if shown:
        widget.grid()
    else:
        widget.grid_remove()


class SideBlock(ttk.Frame):
    """The settings of one side, for the sidebar. `load` it with a family, then `refresh` it to lay it out."""

    def __init__(self, parent: tk.Misc, theme: Theme, side: str, wrap: int, tip: Callable, add_image: tuple,
                 delete_image: tuple, on_changed: Callable, on_add: Callable, on_delete: Callable, on_name: Callable,
                 on_cancel_name: Callable):
        super().__init__(parent)
        px = theme.px
        self.side = side
        self.columnconfigure(1, weight=1)
        header = ttk.Frame(self)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(px(10), px(4)))
        header.columnconfigure(0, weight=1)
        self.title = ttk.Label(header, text="", style="Section.TLabel")
        self.title.grid(row=0, column=0, sticky="w")
        self.save_preset_button = ttk.Button(header, image=add_image, style="Icon.TButton", takefocus=False,
                                             command=on_add)
        self.save_preset_button.grid(row=0, column=1, padx=(px(4), 0))
        tip(self.save_preset_button, f"Save these settings as a preset of your own: it joins the Left and Right lists.")
        self.delete_preset_button = ttk.Button(header, image=delete_image, style="Icon.TButton", takefocus=False,
                                               command=on_delete)
        self.delete_preset_button.grid(row=0, column=2, padx=(px(2), 0))
        tip(self.delete_preset_button, "Delete this preset. The side keeps these settings.")

        self.name_row = ttk.Frame(self)
        self.name_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, px(4)))
        self.name_row.columnconfigure(0, weight=1)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(self.name_row, textvariable=self.name_var)
        self.name_entry.grid(row=0, column=0, sticky="ew")
        self.name_entry.bind("<Return>", lambda _: on_name())
        self.name_entry.bind("<Escape>", on_cancel_name)
        ttk.Button(self.name_row, text="Save", style="Secondary.TButton", width=0,
                   command=on_name).grid(row=0, column=1, padx=(px(6), 0))
        ttk.Button(self.name_row, text="Cancel", style="Secondary.TButton", width=0,
                   command=on_cancel_name).grid(row=0, column=2, padx=(px(6), 0))
        self.name_note = ttk.Label(self, text="", style="Faint.TLabel", wraplength=wrap, justify="left")
        self.name_note.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, px(4)))
        self.hide_name()

        caption_gap = (0, px(10))
        self.version_label = ttk.Label(self, text="Version", style="Small.TLabel")
        self.version_label.grid(row=3, column=0, sticky="w", padx=caption_gap)
        self.version_choice = ttk.Combobox(self, state="readonly")
        self.version_choice.grid(row=3, column=1, sticky="ew")
        self.version_choice.bind("<<ComboboxSelected>>", lambda _: on_changed())
        tip(self.version_choice, VERSION_TIP)
        self.upscaler_note = ttk.Label(self, text="", style="Faint.TLabel", wraplength=wrap, justify="left")
        self.upscaler_note.grid(row=4, column=0, columnspan=2, sticky="w", pady=(px(4), 0))

        self.motion_label = ttk.Label(self, text="Motion", style="Small.TLabel")
        self.motion_label.grid(row=5, column=0, sticky="w", padx=caption_gap, pady=(px(8), 0))
        self.motion_choice = ttk.Combobox(self, state="readonly", values=[choice[0] for choice in MOTION_CHOICES])
        self.motion_choice.grid(row=5, column=1, sticky="ew", pady=(px(8), 0))
        self.motion_choice.bind("<<ComboboxSelected>>", lambda _: on_changed())
        tip(self.motion_choice, MOTION_TIP)
        self.motion_note = ttk.Label(self, text="", style="Faint.TLabel", wraplength=wrap, justify="left")
        self.motion_note.grid(row=6, column=0, columnspan=2, sticky="w", pady=(px(4), 0))

        self.sharpen_label = ttk.Label(self, text="Sharpening", style="Small.TLabel")
        self.sharpen_label.grid(row=7, column=0, sticky="w", padx=caption_gap, pady=(px(8), 0))
        self.sharpen_row = ttk.Frame(self)
        self.sharpen_row.grid(row=7, column=1, sticky="ew", pady=(px(8), 0))
        self.sharpen_row.columnconfigure(1, weight=1)
        self.sharpen_var = tk.BooleanVar(value=False)
        self.sharpen_switch = ttk.Checkbutton(self.sharpen_row, text="", variable=self.sharpen_var,
                                              style="Switch.TCheckbutton", command=on_changed, takefocus=False)
        self.sharpen_switch.grid(row=0, column=0, padx=(0, px(4)))
        self.sharpness_var = tk.DoubleVar(value=DEFAULT_SHARPNESS)
        self.sharpness_scale = ttk.Scale(self.sharpen_row, from_=0.0, to=1.0, variable=self.sharpness_var,
                                         command=lambda _: on_changed())
        self.sharpness_scale.grid(row=0, column=1, sticky="ew")
        self.sharpness_label = ttk.Label(self.sharpen_row, text="off", style="Small.TLabel", width=5, anchor="e")
        self.sharpness_label.grid(row=0, column=2, padx=(px(8), 0))
        tip(self.sharpen_switch, SHARPEN_TIP)
        tip(self.sharpness_scale, SHARPEN_TIP)
        self.load(FAMILIES[0], FAMILIES[0].best, True)

    def load(self, family: Family, how: Preset, still: bool) -> None:
        """Set the widgets to `how`, one of `family`'s; a model only for still images needs `still`, or the family's
        best takes its place."""
        self.family = family
        self.upscalers = [upscaler for upscaler in family.upscalers if still or upscaler not in STILL_ONLY]
        self.version_choice.configure(values=[UPSCALER_CHOICES[upscaler][0] for upscaler in self.upscalers])
        upscaler = how.upscaler if how.upscaler in self.upscalers else family.best.upscaler
        self.version_choice.current(self.upscalers.index(upscaler))
        motion = [(flow, flow_preset) for _, flow, flow_preset, _ in MOTION_CHOICES]
        self.motion_choice.current(motion.index((how.flow, how.flow_preset)))
        self.sharpen_var.set(how.sharpness is not None)
        if how.sharpness is not None:
            self.sharpness_var.set(how.sharpness)

    def how(self) -> Preset:
        """What the widgets say, with only what the family has: no motion or sharpening when it takes none."""
        family = self.family
        flow, flow_preset = MOTION_CHOICES[self.motion_choice.current()][1:3] if family.motion else ("zero", "fast")
        sharpening = family.sharpening and self.sharpen_var.get()
        return Preset(family.name, self.upscalers[self.version_choice.current()], flow, flow_preset,
                      round(float(self.sharpness_var.get()), 2) if sharpening else None)

    def refresh(self, shown: bool, image: bool, idle: bool, own_preset: bool) -> None:
        """Lay out what can be set, or nothing at all; `shown` says the side shows this family, `own_preset` that
        these settings are one of the user's presets, which the bin deletes."""
        family = self.family
        versions, motion = len(self.upscalers) > 1, family.motion and not image
        if not shown or not (versions or motion or family.sharpening):
            self.grid_remove()
            return
        self.grid()
        self.title.configure(text=f"{self.side} · {family.name}".upper())
        self.version_label.configure(text=VERSION_CAPTIONS.get(family.name, "Version"))
        self.upscaler_note.configure(text=UPSCALER_CHOICES[self.upscalers[self.version_choice.current()]][1])
        self.motion_note.configure(text=MOTION_CHOICES[self.motion_choice.current()][3])
        sharpening = self.sharpen_var.get()
        self.sharpness_label.configure(text=f"{self.sharpness_var.get():.2f}" if sharpening else "off")
        for widget in (self.version_label, self.version_choice):
            show_widget(widget, versions)
        for widget in (self.motion_label, self.motion_choice, self.motion_note):
            show_widget(widget, motion)
        for widget in (self.sharpen_label, self.sharpen_row):
            show_widget(widget, family.sharpening)
        show_widget(self.delete_preset_button, own_preset)
        for widget in (self.version_choice, self.motion_choice, self.sharpen_switch, self.save_preset_button,
                       self.delete_preset_button):
            widget.state(["!disabled"] if idle else ["disabled"])
        self.sharpness_scale.state(["!disabled"] if idle and sharpening else ["disabled"])

    def ask_name(self, suggestion: str) -> None:
        """Open the row that names a new preset."""
        self.name_var.set(suggestion)
        self.name_row.grid()
        self.name_note.configure(text="Name the preset, then press Enter.")
        self.name_note.grid()
        self.name_entry.focus_set()
        self.name_entry.select_range(0, "end")

    def hide_name(self) -> None:
        self.name_row.grid_remove()
        self.name_note.grid_remove()
