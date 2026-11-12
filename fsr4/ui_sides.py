"""The sidebar's settings of one side of the picture: its preset, its upscaler, and what that upscaler can change.

The block only holds widgets: it reads back as a `Preset` (`how`) and takes one (`load`). The window keeps one per side
and shows the one Upscale saves; what a side shows, what is saved and the presets themselves are its business.
"""
from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from .hq_scaler import HQ_SCALER
from .presets import BEST, Preset
from .realesrgan import MODELS, REAL_ESRGAN, REAL_ESRGAN_ANIME, REAL_ESRGAN_X4PLUS, STILL_ONLY
from .ui_theme import Theme
from .upscalers import EXTERNAL_UPSCALERS

DEFAULT_SHARPNESS = 0.4
NOT_HERE = "not on this card"  # follows the name of a preset or an upscaler this graphics card cannot run
NO_FSR4_NOTE = "FSR 4 runs only on a Radeon RX 9000 card. FSR 3.1 and 2.3, from the same DLL, run on this one."
UPSCALER_CHOICES = {
    None: ("FSR 4 · machine learning",
           "AMD's newest upscaler, a neural network made for RDNA 4. The best results here."),
    "3.1.6": ("FSR 3.1 · older, hand-tuned",
              "AMD's previous upscaler, kept in the same DLL. Softer and less stable on video; here to compare."),
    "2.3.4": ("FSR 2.3 · older, hand-tuned", "The upscaler before that, also in the DLL. Here to compare."),
    HQ_SCALER: ("AMD HQ scaler · faster, no motion",
                "AMD's video scaler, run by ffmpeg 8.0 or later. Faster than FSR 4 with motion, but each frame stands "
                "alone: it gathers no detail across frames."),
    REAL_ESRGAN: ("Real-ESRGAN · slow, redraws detail",
                  "A neural network that draws each frame again: crisp, about 5 frames a second at 4K, and it can "
                  "change faces and small text. A separate program: see the README."),
    REAL_ESRGAN_X4PLUS: ("Real-ESRGAN x4plus · general, stills only",
                         "Its general model, for still images: about 10 s for a 1080p picture. It can change faces "
                         "and small text."),
    REAL_ESRGAN_ANIME: ("Real-ESRGAN anime · drawings, stills only",
                        "Its model for drawings, for still images: about 4 s for a 1080p picture, and it keeps the "
                        "colours better."),
}
PRESET_TIP = ("A preset holds how the video is upscaled: the upscaler, the motion vectors and the sharpening (not the "
              "size or the encoder). Choose one to load it into the settings below. The Left and Right lists above "
              "the picture offer the same presets, to compare two of them.")
UPSCALER_TIP = ("AMD's FSR DLL holds three upscalers. FSR 4 is the one this tool is for; FSR 3.1 and 2.3 are there to "
                "compare. The HQ scaler comes with AMD's driver and runs in ffmpeg. Real-ESRGAN is a separate program, "
                "with two slower models for still images. Only what the upscaler can change is shown below it. "
                "Measured on a test clip (PSNR against the original, higher is better): FSR 4 26.8 dB, FSR 2.3 26.5, "
                "Lanczos 26.4, HQ scaler 26.3, FSR 3.1 25.9, Real-ESRGAN 24.4 (crisper, but what it draws is not the "
                "original).")
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
    """The settings of one side, for the sidebar. `load` it with a preset, then `refresh` it to lay it out."""

    def __init__(self, parent: tk.Misc, theme: Theme, wrap: int, tip: Callable, add_image: tuple, delete_image: tuple,
                 on_changed: Callable, on_preset: Callable, on_add: Callable, on_delete: Callable, on_name: Callable,
                 on_cancel_name: Callable):
        super().__init__(parent)
        px = theme.px
        self.fsr4_missing = False  # the graphics card cannot run FSR 4, once the window knows
        self.columnconfigure(0, weight=1)
        self._title(0, "Preset", px)
        preset_row = ttk.Frame(self)
        preset_row.grid(row=1, column=0, sticky="ew")
        preset_row.columnconfigure(0, weight=1)
        self.preset_choice = ttk.Combobox(preset_row, state="readonly", height=14)
        self.preset_choice.grid(row=0, column=0, sticky="ew")
        self.preset_choice.bind("<<ComboboxSelected>>", lambda _: on_preset())
        tip(self.preset_choice, PRESET_TIP)
        self.save_preset_button = ttk.Button(preset_row, image=add_image, style="Icon.TButton", takefocus=False,
                                             command=on_add)
        self.save_preset_button.grid(row=0, column=1, padx=(px(4), 0))
        tip(self.save_preset_button, "Save the upscaler and its settings below as a preset of your own.")
        self.delete_preset_button = ttk.Button(preset_row, image=delete_image, style="Icon.TButton", takefocus=False,
                                               command=on_delete)
        self.delete_preset_button.grid(row=0, column=2, padx=(px(2), 0))
        tip(self.delete_preset_button, "Delete this preset (one of your own; the built-in ones stay).")
        self.name_row = ttk.Frame(self)
        self.name_row.grid(row=2, column=0, sticky="ew", pady=(px(6), 0))
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
        self.name_note.grid(row=3, column=0, sticky="w", pady=(px(4), 0))
        self.hide_name()

        self._title(4, "Upscaler", px)
        self.upscaler_choice = ttk.Combobox(self, state="readonly")
        self.upscaler_choice.grid(row=5, column=0, sticky="ew")
        self.upscaler_choice.bind("<<ComboboxSelected>>", lambda _: on_changed())
        tip(self.upscaler_choice, UPSCALER_TIP)
        self.upscaler_note = ttk.Label(self, text="", style="Faint.TLabel", wraplength=wrap, justify="left")
        self.upscaler_note.grid(row=6, column=0, sticky="w", pady=(px(4), 0))

        self.motion_title = self._title(7, "Motion vectors", px)
        self.motion_choice = ttk.Combobox(self, state="readonly", values=[choice[0] for choice in MOTION_CHOICES])
        self.motion_choice.grid(row=8, column=0, sticky="ew")
        self.motion_choice.bind("<<ComboboxSelected>>", lambda _: on_changed())
        tip(self.motion_choice, MOTION_TIP)
        self.motion_note = ttk.Label(self, text="", style="Faint.TLabel", wraplength=wrap, justify="left")
        self.motion_note.grid(row=9, column=0, sticky="w", pady=(px(4), 0))

        self.sharpen_title = self._title(10, "Sharpening", px)
        self.sharpen_row = ttk.Frame(self)
        self.sharpen_row.grid(row=11, column=0, sticky="ew")
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
        self.motion_choice.current(0)
        self.load(BEST, True)

    def _title(self, row: int, text: str, px: Callable) -> ttk.Label:
        title = ttk.Label(self, text=text.upper(), style="Section.TLabel")
        title.grid(row=row, column=0, sticky="w", pady=(px(10), px(4)))
        return title

    def load(self, how: Preset, still: bool) -> None:
        """Set the widgets to `how`; a model only for still images needs `still`, or the video model takes its place.
        What the upscaler has no use for stays as it was, for the next upscaler that has."""
        self.upscalers = [upscaler for upscaler in UPSCALER_CHOICES if still or upscaler not in STILL_ONLY]
        self.upscaler_choice.configure(values=[self._upscaler_label(upscaler) for upscaler in self.upscalers])
        upscaler = how.upscaler if how.upscaler in self.upscalers else REAL_ESRGAN
        self.upscaler_choice.current(self.upscalers.index(upscaler))
        if upscaler not in EXTERNAL_UPSCALERS:
            motion = [(flow, flow_preset) for _, flow, flow_preset, _ in MOTION_CHOICES]
            self.motion_choice.current(motion.index((how.flow, how.flow_preset)))
        if upscaler not in MODELS:
            self.sharpen_var.set(how.sharpness is not None)
            if how.sharpness is not None:
                self.sharpness_var.set(how.sharpness)

    def _upscaler_label(self, upscaler: str | None) -> str:
        """The upscaler's entry in the list, "FSR 4 · not on this card" when the card cannot run it."""
        label = UPSCALER_CHOICES[upscaler][0]
        if upscaler is None and self.fsr4_missing:
            return f"{label.partition(' · ')[0]} · {NOT_HERE}"
        return label

    def how(self) -> Preset:
        """What the widgets say, with only what the upscaler has: no motion or sharpening when it takes none."""
        upscaler = self.upscalers[self.upscaler_choice.current()]
        flow, flow_preset = (("zero", "fast") if upscaler in EXTERNAL_UPSCALERS
                             else MOTION_CHOICES[self.motion_choice.current()][1:3])
        sharpening = upscaler not in MODELS and self.sharpen_var.get()
        return Preset("", upscaler, flow, flow_preset,
                      round(float(self.sharpness_var.get()), 2) if sharpening else None)

    def set_presets(self, names: list[str]) -> None:
        self.preset_choice.configure(values=names)

    def refresh(self, shown: bool, image: bool, idle: bool, preset: str, own_preset: bool) -> None:
        """Lay out the settings of the upscaler chosen and nothing else, or the whole block out when not `shown`;
        `preset` names these settings, `own_preset` says the bin can delete them."""
        if not shown:
            self.grid_remove()
            return
        self.grid()
        upscaler = self.upscalers[self.upscaler_choice.current()]
        motion, sharpens = upscaler not in EXTERNAL_UPSCALERS and not image, upscaler not in MODELS
        self.preset_choice.set(preset)
        missing = upscaler is None and self.fsr4_missing
        self.upscaler_note.configure(text=NO_FSR4_NOTE if missing else UPSCALER_CHOICES[upscaler][1])
        self.motion_note.configure(text=MOTION_CHOICES[self.motion_choice.current()][3])
        sharpening = self.sharpen_var.get()
        self.sharpness_label.configure(text=f"{self.sharpness_var.get():.2f}" if sharpening else "off")
        for widget in (self.motion_title, self.motion_choice, self.motion_note):
            show_widget(widget, motion)
        for widget in (self.sharpen_title, self.sharpen_row):
            show_widget(widget, sharpens)
        show_widget(self.delete_preset_button, own_preset)
        for widget in (self.preset_choice, self.save_preset_button, self.delete_preset_button, self.upscaler_choice,
                       self.motion_choice, self.sharpen_switch):
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
