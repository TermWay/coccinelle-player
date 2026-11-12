"""The window: open a video or an image, watch it upscaled while it is made, compare it, save it.

Layout: a settings sidebar that folds away (its button at the top left, Ctrl+B) with the preset, the upscaler and what
that upscaler can change and, at its foot, the machine's status; the preview with its toolbar (what each side of the divider shows, zoom), the
playback controls with the filmstrip, an optional log, and a footer with the progress and the actions. Full screen (F11 or F) keeps
the video; its controls come up when the mouse moves near the bottom and hide once it rests.

Each side of the divider shows a preset, with its settings if they were changed, or the source through a filter.
A loaded video gets a Renderer (fsr4.render), which upscales every version shown into pieces around the playhead, and
a Player (fsr4.player), which plays the finished pieces in real time with sound. Upscale makes the renderer do the
whole file the way the saved side says and joins the pieces into the output. An image gets an
ImagePreviewer (fsr4.image_preview) for the versions shown; saving it runs the one-shot job. Workers reach the
window through a queue polled every 30 ms. Files can be dropped anywhere on the window.
"""
from __future__ import annotations

import ctypes
import math
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from . import build_info, log, system, temporary, usage
from .ffx import find_sdk_dir
from .image_preview import ImagePreviewer
from .realesrgan import STILL_ONLY
from .job import JobOptions, JobPlan, JobResult, default_output, plan_from_info, run_job
from .media import (COPY_ANY_AUDIO_SUFFIXES, IMAGE_SUFFIXES, MP4_AUDIO_COPY, VIDEO_SUFFIXES, KeyframeIndex, MediaInfo,
                    is_image, probe, read_frame, read_thumbnail)
from .player import Player
from .preferences import PreferenceStore
from .presets import BEST, OFF_LABEL, SOURCE_LABELS, Preset, PresetStore
from .render import RenderSettings, Renderer, clean_stale_directories
from .timeline import intersect_ranges
from .ui_art import icon, png_base64
from .ui_sides import NOT_HERE, SideBlock, show_widget
from .ui_theme import Theme, enable_dark_dialogs, style_title_bar
from .ui_tooltip import Tooltip
from .ui_transport import VOLUME_KEY_DECIBELS, Transport, format_time
from .ui_slider import Mark, MarkedSlider
from .ui_video import OFF, REFERENCE_FILTERS, UNAVAILABLE, VideoView
from .ui_window import Curtain, Fullscreen, keep_contents_on_resize, repaint

APP_NAME = "Coccinelle Player"
POLL_MS = 30
DRAG_SEEK_SECONDS = 0.12
SIDEBAR_WIDTH = 340
SWITCH_GAP = 12  # between Choose file and its switch, which belong together
SETTINGS_DELAY_MS = 450
HIDE_CONTROLS_MS = 2500
REPAINT_SETTLE_MS = 150
MACHINE_REFRESH_MS = 2000
# Playing backwards, the renderer fills the frames just behind the playhead a chunk at a time, nearest first,
# until this much is ready behind it: more than the player ever waits for (BUFFER_MAX_SECONDS).
REVERSE_CHUNK_SECONDS = 1.0
REVERSE_AHEAD_SECONDS = 12.0
HIDE_LEAVE_MS = 600
CONTROLS_ZONE_MARGIN = 80
# The zoom slider's marks, (zoom, label, rank): a narrow slider keeps the labels of rank 0.
ZOOM_TICKS = (("source", "Original", 1), (1, "100%", 0), (2, "200%", 1), (4, "400%", 1), (8, "800%", 1),
              (16, "1600%", 0))
ZOOM_WIDTH = 480  # 96 DPI pixels, the zoom slider's width when there is room
ZOOM_MIN_WIDTH = 160
ZOOM_SPARE_WEIGHT = 1000  # the room after the zoom slider takes nearly all the spare width
SEEK_STEP_SECONDS = 5
SOURCE_THUMBNAILS = 40
THUMBNAIL_HEIGHT = 108
SIZE_CHOICES = [("1.5×", "1.5"), ("2×", "2"), ("3×", "3"), ("1440p", "1440"), ("4K", "2160"), ("Custom", "custom")]
SIZE_TIPS = {
    "1.5": "1.5 times the width and height (2.25 times the pixels). FSR calls this ratio Quality.",
    "2": "Twice the width and height, four times the pixels. FSR calls this ratio Performance.",
    "3": "Three times the width and height: FSR's largest step (Ultra Performance). Best for small sources.",
    "1440": "The short side becomes 1440 pixels (2560 × 1440 for a 16:9 video).",
    "2160": "The short side becomes 2160 pixels: 4K, 3840 × 2160 for a 16:9 video.",
    "custom": "Any factor from 1× to 3× with the slider.",
}
TARGET_SHORT_SIDES = {"1440": 1440, "2160": 2160}
ENCODER_CHOICES = [
    ("HEVC · GPU (AMF)", "amf",
     "Encoded by the graphics card: fast and leaves the processor free. Plays on most devices from the last "
     "ten years."),
    ("AV1 · GPU (AMF)", "av1_amf",
     "Encoded by the graphics card, files about 15–20% smaller than HEVC at the same quality. Needs a recent "
     "player or device."),
    ("H.264 · GPU (AMF)", "h264_amf",
     "Plays everywhere, even on old devices and browsers; files about 15% larger than HEVC. At most 4096 pixels "
     "wide."),
    ("H.264 · CPU (x264)", "x264",
     "Encoded by the processor: plays everywhere and suits sharp text, but keeps the processor busy."),
    ("HEVC · CPU (x265)", "x265",
     "Small files from the processor, but the slowest choice: at 4K it can hold the upscale back."),
]
FILTER_HELP = {"nearest": "every source pixel as a block, exactly what the file holds",
               "linear": "smooth, what most players do", "cubic": "a little sharper, the usual photo editor filter",
               "lanczos": "the sharpest classic filter (mpv, madVR)"}
UPSCALED = "upscaled"  # a side showing its settings, the other choices being a filter or OFF
DEFAULT_SIDES = {"left": "cubic", "right": UPSCALED}
CUSTOM_PRESET = "Custom settings"
NOTHING_UPSCALED = "Ready · nothing is upscaled · choose a preset in the Left or Right list to upscale"
OPEN_TYPES = [("Videos and images", " ".join(f"*{suffix}" for suffix in VIDEO_SUFFIXES + IMAGE_SUFFIXES)),
              ("Videos", " ".join(f"*{suffix}" for suffix in VIDEO_SUFFIXES)),
              ("Images", " ".join(f"*{suffix}" for suffix in IMAGE_SUFFIXES)), ("All files", "*.*")]
VIDEO_SAVE_TYPES = [("MP4 video", "*.mp4"), ("Matroska video (keeps every audio track as is)", "*.mkv"),
                    ("QuickTime video", "*.mov")]
IMAGE_SAVE_TYPES = [("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("WebP image", "*.webp")]
VIDEO_CODEC_NAMES = {"h264": "H.264", "hevc": "HEVC", "av1": "AV1", "vp9": "VP9", "vp8": "VP8", "wmv1": "WMV 7",
                     "wmv2": "WMV 8", "wmv3": "WMV 9", "vc1": "VC-1", "mpeg2video": "MPEG-2", "mpeg1video": "MPEG-1",
                     "mpeg4": "MPEG-4", "msmpeg4v3": "DivX 3", "msmpeg4v2": "MS MPEG-4", "flv1": "Flash video",
                     "mjpeg": "Motion JPEG", "prores": "ProRes", "dvvideo": "DV", "msvideo1": "Video 1",
                     "rv40": "RealVideo 4", "theora": "Theora", "png": "PNG", "webp": "WebP", "bmp": "BMP",
                     "tiff": "TIFF", "gif": "GIF"}
AUDIO_CODEC_NAMES = {"aac": "AAC", "mp3": "MP3", "ac3": "AC-3", "eac3": "E-AC-3", "wmav1": "WMA", "wmav2": "WMA",
                     "wmapro": "WMA Pro", "wmavoice": "WMA Voice", "opus": "Opus", "vorbis": "Vorbis", "flac": "FLAC",
                     "alac": "ALAC", "mp2": "MP2", "dts": "DTS", "truehd": "TrueHD", "cook": "RealAudio"}


def codec_name(codec: str, names: dict[str, str]) -> str:
    if codec.startswith("pcm_"):
        return "PCM"
    return names.get(codec, codec.upper())


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def free_name(path: Path) -> Path:
    """`path`, or the first of `name (2).ext`, `name (3).ext`… that is not an existing file."""
    candidate, number = path, 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem} ({number}){path.suffix}")
        number += 1
    return candidate


def describe_source(path: Path, info: MediaInfo) -> str:
    """Two short lines about a probed file for the sidebar."""
    size = f"{info.width} × {info.height}"
    if is_image(path):
        return f"{codec_name(info.codec, VIDEO_CODEC_NAMES)} image · {size}"
    rate = f"{float(info.frame_rate):.3f}".rstrip("0").rstrip(".")
    first = f"{codec_name(info.codec, VIDEO_CODEC_NAMES)} · {size} · {rate} fps"
    parts = []
    if info.pixel_aspect != 1:
        parts.append(f"plays as {round(info.display_width)} × {info.height}")
    if info.duration:
        parts.append(format_duration(info.duration))
    if info.frame_count:
        parts.append(f"{info.frame_count:,} frames")
    if info.audio:
        names = sorted({codec_name(track.codec, AUDIO_CODEC_NAMES) for track in info.audio})
        tracks = f"{len(info.audio)} audio tracks" if len(info.audio) > 1 else "audio"
        parts.append(f"{' + '.join(names)} {tracks}")
    else:
        parts.append("no audio")
    return first + "\n" + " · ".join(parts)


def describe_audio_plan(info: MediaInfo, output: Path) -> str:
    """What happens to the audio, in one line."""
    if not info.audio:
        return "No audio in this file."
    copy_any = output.suffix.lower() in COPY_ANY_AUDIO_SUFFIXES
    copied = [track for track in info.audio if copy_any or track.codec in MP4_AUDIO_COPY]
    converted = [track for track in info.audio if track not in copied]
    parts = []
    if copied:
        parts.append(f"{', '.join(sorted({codec_name(t.codec, AUDIO_CODEC_NAMES) for t in copied}))} copied as is")
    if converted:
        names = ", ".join(sorted({codec_name(t.codec, AUDIO_CODEC_NAMES) for t in converted}))
        parts.append(f"{names} converted to AAC ({output.suffix or 'this format'} cannot hold it)")
    return "Audio: " + "; ".join(parts) + "."


def to_bgra(rgba: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)


class App:
    """The main window. `load`, `start`, `cancel` and the playback methods are also what the UI check drives.
    `preferences` replaces the user's own (the UI check gives it a scratch file)."""

    def __init__(self, root: tk.Tk, preselected: Path | None = None, preferences: PreferenceStore | None = None):
        self.root = root
        self.preferences = preferences or PreferenceStore()
        self.theme = Theme(root)
        self.queue: queue.Queue = queue.Queue()
        self.load_token = 0
        self.input_path: Path | None = None
        self.info: MediaInfo | None = None
        self.keyframes: KeyframeIndex | None = None
        self.renderer: Renderer | None = None
        self.player: Player | None = None
        self.source_thumbnails: dict[int, np.ndarray] = {}
        self.output_custom = False
        self.output_confirmed: Path | None = None  # picked in the Save dialog, which asked about replacing it
        self._setting_output = False
        self.last_saved: Path | None = None
        self.closing = False
        self.image_running = False
        self.image_cancel = threading.Event()
        self.worker: threading.Thread | None = None
        self.batch = False
        self.saving = False
        self.render_position: int | None = None
        self.render_speed = 0.0
        self.gpu_ready: bool | None = None
        self.fsr4_missing: str | None = None  # why the card cannot run FSR 4, when the probe found so
        self.log_visible = False
        self.sidebar_visible = True
        self.fullscreen = False
        self._settings_job: str | None = None
        self._hide_job: str | None = None
        self._comparing_shown = False
        self._last_drag_seek = 0.0
        self.processor = system.processor_name()
        self.video_memory: system.VideoMemory | None = None
        self.gpu_text = ""
        self.screen = Fullscreen(root)
        self._repaint_job: str | None = None
        keep_contents_on_resize(root)
        self.presets = PresetStore()
        self.side_choices = dict(DEFAULT_SIDES)
        self._shown_layers: dict[str, RenderSettings] = {}
        self.image_source: np.ndarray | None = None
        self.image_previewer: ImagePreviewer | None = None
        self._leave_job: str | None = None
        self._pointer_pressed = False
        self._last_pointer: tuple[int, int] | None = None
        self._cursor_hidden = False
        root.title(APP_NAME)
        root.iconphoto(True, *self.theme.icon_photos())
        self.icons = {(name, color): tk.PhotoImage(master=root, data=png_base64(icon(name, self.theme.px(20),
                                                                                       color)))
                      for name in ("sidebar", "add", "delete")
                      for color in (self.theme.palette.text_secondary, self.theme.palette.text_faint)}
        self._build()
        self._update_preset_lists()
        self._place_window()
        log.set_sink(lambda line: self.queue.put(("log", line)))
        version = build_info.describe()
        self.version_label.configure(text=f"v{version}")
        log.emit(f"[ui] {APP_NAME} {version}")
        self.drop = self._enable_drop()
        self._bind_keys()
        root.protocol("WM_DELETE_WINDOW", self._close_requested)
        root.bind("<Configure>", self._window_configured, add="+")
        root.report_callback_exception = self._callback_failed
        root.after(POLL_MS, self._poll)
        self._show_machine()
        threading.Thread(target=self._probe_gpu, daemon=True).start()
        threading.Thread(target=self._clean_temp, daemon=True).start()
        self._refresh()
        if self.presets.load_error is not None:
            self._show_banner(f"Your presets could not be read ({self.presets.load_error}); saving a preset keeps "
                              f"that file as {self.presets.path.with_suffix('.unreadable').name}.")
        if self.preferences.load_error is not None:
            self._show_banner(f"Your preferences could not be read ({self.preferences.load_error}); the defaults "
                              "are used, and changing one keeps that file as "
                              f"{self.preferences.path.with_suffix('.unreadable').name}.")
        if preselected is not None:
            self.load(preselected)

    @property
    def busy(self) -> bool:
        """A whole-file upscale, a save or an image job is running: the settings are locked."""
        return self.batch or self.saving or self.image_running

    # ---- layout ---------------------------------------------------------------------------------

    def _build(self) -> None:
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.body = ttk.Frame(root)
        self.body.grid(row=0, column=0, sticky="nsew")
        self.body.columnconfigure(1, weight=1)
        self.body.rowconfigure(0, weight=1)
        self.sidebar = self._build_sidebar(self.body)
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.main = self._build_main(self.body)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.footer = self._build_footer()
        self.footer.grid(row=1, column=0, sticky="ew")

    def _tip(self, widget: tk.Misc, text) -> None:
        Tooltip(widget, self.theme, text)

    def _icon_images(self, name: str) -> tuple:
        """A ttk image option: the icon, fainter when its button is disabled."""
        palette = self.theme.palette
        return self.icons[(name, palette.text_secondary)], "disabled", self.icons[(name, palette.text_faint)]

    def _sidebar_button(self, parent: ttk.Frame) -> ttk.Button:
        """The button that folds the settings away; there are two, one in the sidebar and one in the toolbar."""
        button = ttk.Button(parent, image=self._icon_images("sidebar"), style="Icon.TButton", takefocus=False,
                            command=self.toggle_sidebar)
        self._tip(button, "Hide or show the settings (Ctrl+B).")
        return button

    def _build_machine_status(self, parent: ttk.Frame) -> ttk.Frame:
        """The processor and the graphics card with the memory this program uses on each, at the sidebar's foot."""
        px = self.theme.px
        status = ttk.Frame(parent, padding=(px(20), px(8), px(6), px(8)))
        status.columnconfigure(1, weight=1)
        self.cpu_label = ttk.Label(status, text="", style="Small.TLabel")
        self.cpu_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, px(2)))
        self._tip(self.cpu_label, "The processor, and the main memory (RAM) this program uses now / installed.")
        self.gpu_dot = ttk.Label(status, text="●", style="Faint.TLabel")
        self.gpu_dot.grid(row=1, column=0, sticky="nw", padx=(0, px(6)))
        self.gpu_label = ttk.Label(status, text="Checking the GPU…", style="Small.TLabel",
                                   wraplength=px(SIDEBAR_WIDTH - 56), justify="left")
        self.gpu_label.grid(row=1, column=1, sticky="w")
        self._tip(self.gpu_label, "The graphics card, the memory on it (VRAM) this program uses now (FSR and its "
                                  "frames) / its size, and the version of FSR in AMD's DLL.\n\n"
                                  f"{APP_NAME} is an independent tool, not made or endorsed by AMD; AMD, Radeon "
                                  "and FidelityFX are trademarks of Advanced Micro Devices, Inc.")
        self.version_label = ttk.Label(status, text="", style="Faint.TLabel", wraplength=px(SIDEBAR_WIDTH - 40),
                                       justify="left")
        self.version_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(px(4), 0))
        self._tip(self.version_label, "This copy's version and the commit it was made from: for the executables, "
                                      "with the time they were built; \"from the sources\" when Python runs it. "
                                      f"\"{build_info.CHANGED_MARK.strip()}\": files differed from that commit.")
        return status

    def _main_padding(self) -> tuple[int, int, int, int]:
        """Around the preview: next to the sidebar, or the window's own margin once it is folded."""
        px = self.theme.px
        return px(4) if self.sidebar_visible else px(20), px(12), px(20), px(6)

    def _section(self, parent: ttk.Frame, row: int, title: str) -> int:
        px = self.theme.px
        ttk.Label(parent, text=title.upper(), style="Section.TLabel").grid(
            row=row, column=0, sticky="w", pady=(px(10) if row else px(2), px(4)))
        return row + 1

    def _build_sidebar(self, parent: ttk.Frame) -> ttk.Frame:
        """Settings top to bottom in the order they are decided: source, size, preset, upscaler and what it can
        change, output.

        The settings sit in a canvas that scrolls (wheel or scrollbar) when the window is too short for them;
        the machine's status stays below it.
        """
        palette, px = self.theme.palette, self.theme.px
        holder = ttk.Frame(parent, width=px(SIDEBAR_WIDTH))
        holder.grid_propagate(False)
        holder.rowconfigure(0, weight=1)
        holder.columnconfigure(0, weight=1)
        canvas = tk.Canvas(holder, background=palette.background, highlightthickness=0, borderwidth=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        # The scrollbar's column stays reserved, so the settings keep one width whether it shows or not (a canvas
        # window does not always follow its canvas back to a narrower width, e.g. after full screen).
        holder.columnconfigure(1, minsize=scrollbar.winfo_reqwidth() + px(2))
        tk.Frame(holder, background=palette.border, height=px(1)).grid(row=1, column=0, columnspan=2, sticky="ew",
                                                                       padx=(px(20), px(6)))
        self._build_machine_status(holder).grid(row=2, column=0, columnspan=2, sticky="ew")
        sidebar = ttk.Frame(canvas, padding=(px(20), px(12), px(6), px(8)))
        sidebar.columnconfigure(0, weight=1)
        item = canvas.create_window(0, 0, window=sidebar, anchor="nw")
        self.sidebar_canvas, self.sidebar_scrollbar, self.sidebar_content = canvas, scrollbar, sidebar

        def fit(_event=None) -> None:
            canvas.itemconfigure(item, width=canvas.winfo_width())
            height = sidebar.winfo_reqheight()
            canvas.configure(scrollregion=(0, 0, canvas.winfo_width(), height))
            scrolling = height > canvas.winfo_height() + 1
            if scrolling:
                scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, px(2)))
            else:
                scrollbar.grid_remove()
                canvas.yview_moveto(0)

        sidebar.bind("<Configure>", fit)
        canvas.bind("<Configure>", fit)
        wrap = px(SIDEBAR_WIDTH - 38)

        title_row = ttk.Frame(sidebar)
        title_row.grid(row=0, column=0, sticky="w", pady=(0, px(4)))
        self.fold_button = self._sidebar_button(title_row)
        self.fold_button.grid(row=0, column=0, padx=(0, px(8)))
        ttk.Label(title_row, text="SOURCE", style="Section.TLabel").grid(row=0, column=1)
        row = 1
        self.source_name = ttk.Label(sidebar, text="No file yet", style="Strong.TLabel", wraplength=wrap)
        self.source_name.grid(row=row, column=0, sticky="w")
        self.source_details = ttk.Label(sidebar, text="Drop a file anywhere on this window, or choose one.",
                                        style="Small.TLabel", wraplength=wrap, justify="left")
        self.source_details.grid(row=row + 1, column=0, sticky="w", pady=(px(2), px(6)))
        open_row = ttk.Frame(sidebar)
        open_row.grid(row=row + 2, column=0, sticky="ew")
        open_row.columnconfigure(1, weight=1)
        self.choose_button = ttk.Button(open_row, text="Choose file…", style="Secondary.TButton", command=self.browse)
        self.choose_button.grid(row=0, column=0)
        self._tip(self.choose_button, "Open a video or an image (Ctrl+O). You can also drop a file anywhere on "
                                      "the window.")
        self._build_upscale_on_open(open_row)

        row = self._section(sidebar, row + 3, "Size")
        chips = ttk.Frame(sidebar)
        chips.grid(row=row, column=0, sticky="w")
        self.size_var = tk.StringVar(value="2")
        self.size_chips: dict[str, ttk.Radiobutton] = {}
        for index, (text, value) in enumerate(SIZE_CHOICES):
            chip = ttk.Radiobutton(chips, text=text, value=value, variable=self.size_var,
                                   style="Segment.TRadiobutton", command=self._settings_changed, takefocus=False)
            chip.grid(row=0, column=index, padx=(0, px(5)))
            self.size_chips[value] = chip
            self._tip(chip, SIZE_TIPS[value])
        self.custom_row = ttk.Frame(sidebar)
        self.custom_row.grid(row=row + 1, column=0, sticky="ew", pady=(px(8), 0))
        self.custom_row.columnconfigure(0, weight=1)
        self.custom_scale = tk.DoubleVar(value=2.0)
        custom_slider = ttk.Scale(self.custom_row, from_=1.0, to=3.0, variable=self.custom_scale,
                                  command=lambda _: self._settings_changed())
        custom_slider.grid(row=0, column=0, sticky="ew")
        self._tip(custom_slider, "The factor applied to the width and the height, from 1× (same size, FSR only "
                                 "cleans the picture) to 3×.")
        self.custom_label = ttk.Label(self.custom_row, text="2.00×", style="Small.TLabel", width=6, anchor="e")
        self.custom_label.grid(row=0, column=1, padx=(px(8), 0))
        self.custom_row.grid_remove()
        self.size_label = ttk.Label(sidebar, text="", style="Strong.TLabel", wraplength=wrap, justify="left")
        self.size_label.grid(row=row + 2, column=0, sticky="w", pady=(px(7), 0))
        self.size_note = ttk.Label(sidebar, text="", style="Faint.TLabel", wraplength=wrap, justify="left")
        self.size_note.grid(row=row + 3, column=0, sticky="w")

        row += 4
        self.save_row = ttk.Frame(sidebar)
        self.save_row.grid(row=row, column=0, sticky="w", pady=(px(10), 0))
        ttk.Label(self.save_row, text="Settings of", style="Small.TLabel").grid(row=0, column=0, padx=(0, px(8)))
        self.save_side_var = tk.StringVar(value="right")
        self.save_chips = []
        for index, side in enumerate(DEFAULT_SIDES):
            chip = ttk.Radiobutton(self.save_row, text=f"{side.capitalize()} side", value=side,
                                   variable=self.save_side_var, style="Segment.TRadiobutton", takefocus=False,
                                   command=self._settings_changed)
            chip.grid(row=0, column=index + 1, padx=(0, px(5)))
            self._tip(chip, "Both sides are upscaled: the one the settings below change, and the one Upscale saves.")
            self.save_chips.append(chip)
        self.save_row.grid_remove()
        row += 1
        self.side_blocks: dict[str, SideBlock] = {}
        for side in DEFAULT_SIDES:
            block = SideBlock(sidebar, self.theme, wrap, self._tip, self._icon_images("add"),
                              self._icon_images("delete"), on_changed=self._settings_changed,
                              on_preset=lambda side=side: self._preset_chosen(side),
                              on_add=lambda side=side: self._begin_save_preset(side),
                              on_delete=lambda side=side: self._delete_preset(side),
                              on_name=lambda side=side: self._finish_save_preset(side),
                              on_cancel_name=lambda _event=None, side=side: self._cancel_save_preset(side))
            block.grid(row=row, column=0, sticky="ew")
            block.grid_remove()
            self.side_blocks[side] = block
            row += 1

        row = self._section(sidebar, row, "Output")
        self.save_note = ttk.Label(sidebar, text="Neither side is upscaled. Choose a preset in the Left or Right "
                                                 "list to save a file.", style="Faint.TLabel", wraplength=wrap,
                                   justify="left")
        self.save_note.grid(row=row + 1, column=0, sticky="w", pady=(0, px(6)))
        self.save_note.grid_remove()
        row += 2
        output_row = ttk.Frame(sidebar)
        output_row.grid(row=row, column=0, sticky="ew")
        output_row.columnconfigure(0, weight=1)
        self.output_var = tk.StringVar()
        self.output_var.trace_add("write", self._output_edited)
        self.output_entry = ttk.Entry(output_row, textvariable=self.output_var)
        self.output_entry.grid(row=0, column=0, sticky="ew")
        for key in ("<Return>", "<Escape>"):
            self.output_entry.bind(key, lambda _: self.root.focus_set())  # done typing: the keys play again
        self._tip(self.output_entry, "Where the upscaled file is saved. .mp4 plays everywhere; .mkv keeps every "
                                     "audio track exactly as it is.")
        self.save_button = ttk.Button(output_row, text="…", width=3, style="Secondary.TButton",
                                      command=self.browse_output)
        self.save_button.grid(row=0, column=1, padx=(px(6), 0))
        self._tip(self.save_button, "Choose where to save, and the file type.")
        self.encoder_choice = ttk.Combobox(sidebar, state="readonly", values=[c[0] for c in ENCODER_CHOICES])
        self.encoder_choice.current(0)
        self.encoder_choice.grid(row=row + 1, column=0, sticky="ew", pady=(px(6), 0))
        self.encoder_choice.bind("<<ComboboxSelected>>", lambda _: self._settings_changed())
        self._tip(self.encoder_choice, "How the upscaled video is compressed. Every choice keeps the same quality; "
                                       "they differ in file size, speed and which devices play the result. The "
                                       "GPU encoders are fast and leave the processor free: HEVC is the default, "
                                       "AV1 makes smaller files for recent devices, H.264 plays on anything. The "
                                       "CPU encoders (x264, x265) are slower and keep the processor busy.")
        self.encoder_note = ttk.Label(sidebar, text="", style="Faint.TLabel", wraplength=wrap, justify="left")
        self.encoder_note.grid(row=row + 2, column=0, sticky="w", pady=(px(4), 0))
        self.audio_note = ttk.Label(sidebar, text="", style="Faint.TLabel", wraplength=wrap, justify="left")
        self.audio_note.grid(row=row + 3, column=0, sticky="w", pady=(px(3), 0))

        row = self._section(sidebar, row + 4, "Processor")
        threads_row = ttk.Frame(sidebar)
        threads_row.grid(row=row, column=0, sticky="ew")
        threads_row.columnconfigure(1, weight=1)
        ttk.Label(threads_row, text="Threads", style="Small.TLabel").grid(row=0, column=0, padx=(0, px(8)))
        self.threads_var = tk.DoubleVar(value=usage.DEFAULT.threads)
        threads_slider = ttk.Scale(threads_row, from_=1, to=usage.CPU_THREADS, variable=self.threads_var,
                                   command=lambda _: self._processor_changed())
        threads_slider.grid(row=0, column=1, sticky="ew")
        self.threads_label = ttk.Label(threads_row, text="", style="Small.TLabel", width=9, anchor="e")
        self.threads_label.grid(row=0, column=2, padx=(px(8), 0))
        threads_tip = (f"How many of the processor's {usage.CPU_THREADS} threads upscaling may use (a core runs two "
                       "threads). Fewer leaves more for other programs. Optical flow uses at most 8, the CPU "
                       "encoders (x264, x265) and the decoder use them all; the GPU encoders barely use any.")
        self._tip(threads_slider, threads_tip)
        priority_row = ttk.Frame(sidebar)
        priority_row.grid(row=row + 1, column=0, sticky="w", pady=(px(8), 0))
        self.priority_var = tk.StringVar(value=usage.DEFAULT.priority)
        for index, priority in enumerate(usage.PRIORITIES.values()):
            chip = ttk.Radiobutton(priority_row, text=priority.label, value=priority.name,
                                   variable=self.priority_var, style="Segment.TRadiobutton", takefocus=False,
                                   command=self._processor_changed)
            chip.grid(row=0, column=index, padx=(0, px(5)))
            self._tip(chip, priority.description)
        self._processor_job: str | None = None
        self.threads_label.configure(text=f"{usage.DEFAULT.threads} of {usage.CPU_THREADS}")
        self.setting_widgets = [self.choose_button, self.output_entry, self.save_button, self.encoder_choice,
                                custom_slider, *self.save_chips]
        self._route_wheel(holder)
        return holder

    def _build_upscale_on_open(self, open_row: ttk.Frame) -> None:
        """The switch just after Choose file, on its line: whether a file is upscaled as soon as it opens, kept
        between runs."""
        self.upscale_on_open_var = tk.BooleanVar(value=self.preferences.preferences.upscale_on_open)
        self.upscale_on_open_switch = ttk.Checkbutton(
            open_row, text="Auto upscale", variable=self.upscale_on_open_var,
            style="Switch.TCheckbutton", command=self._upscale_on_open_changed, takefocus=False)
        self.upscale_on_open_switch.grid(row=0, column=1, sticky="w", padx=(self.theme.px(SWITCH_GAP), 0))
        self._tip(self.upscale_on_open_switch,
                  "Upscale as soon as a file opens.\n"
                  "On: a file opens with each side as you left it, and FSR starts on the sides that show a preset.\n"
                  "Off: a file opens as in a plain player, the source enlarged by a classic filter, and nothing is "
                  "upscaled until you choose a preset in the Left or Right list.\nFrom the next file you open.")

    def _upscale_on_open_changed(self) -> None:
        if not self._save_preferences(upscale_on_open=bool(self.upscale_on_open_var.get())):
            self.upscale_on_open_var.set(self.preferences.preferences.upscale_on_open)

    def _save_preferences(self, **changes) -> bool:
        """Keep the preferences with `changes`; False, said in the banner, when they cannot be written."""
        try:
            self.preferences.change(**changes)
        except OSError as error:
            log.emit(f"[ui] error: cannot save the preferences: {error}")
            self._show_banner(f"The preferences could not be saved: {self._first_line(error)}")
            return False
        return True

    def _route_wheel(self, widget: tk.Misc) -> None:
        """The wheel scrolls the sidebar wherever the pointer is on it (instead of changing a dropdown's value)."""
        widget.bind("<MouseWheel>", self._sidebar_wheel)
        for child in widget.winfo_children():
            self._route_wheel(child)

    def _sidebar_wheel(self, event: tk.Event) -> str:
        if self.sidebar_scrollbar.winfo_ismapped():
            self.sidebar_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        return "break"

    def _build_main(self, parent: ttk.Frame) -> ttk.Frame:
        px = self.theme.px
        main = ttk.Frame(parent, padding=self._main_padding())
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)
        self.banner = ttk.Frame(main, style="Banner.TFrame", padding=(px(14), px(8), px(6), px(8)))
        self.banner.columnconfigure(0, weight=1)
        self.banner_label = ttk.Label(self.banner, text="", style="Banner.TLabel", wraplength=px(700), justify="left")
        self.banner_label.grid(row=0, column=0, sticky="w")
        ttk.Button(self.banner, text="✕", style="Banner.TButton", command=self._hide_banner,
                   takefocus=False).grid(row=0, column=1, padx=(px(8), 0))
        self.banner.grid(row=0, column=0, sticky="ew", pady=(0, px(10)))
        self.banner.grid_remove()

        self.toolbar = ttk.Frame(main)
        self.toolbar.grid(row=1, column=0, sticky="ew", pady=(0, px(8)))
        self.toolbar.columnconfigure(2, weight=1)
        # The sidebar's button, in the same place on the screen, while the sidebar is folded.
        self.unfold_button = self._sidebar_button(self.toolbar)
        self.unfold_button.grid(row=0, column=0, padx=(0, px(10)))
        self.unfold_button.grid_remove()
        self.side_boxes: dict[str, ttk.Combobox] = {}
        for side, column, sticky in (("left", 1, "w"), ("right", 3, "e")):
            group = ttk.Frame(self.toolbar)
            group.grid(row=0, column=column, sticky=sticky)
            ttk.Label(group, text=side.capitalize(), style="Small.TLabel").grid(row=0, column=0, padx=(0, px(6)))
            # Wide enough for "FSR 4 · fast motion · not on this card": the list is as wide as the box.
            box = ttk.Combobox(group, state="readonly", width=28, height=14)
            box.grid(row=0, column=1)
            box.bind("<<ComboboxSelected>>", lambda _, side=side: self._side_chosen(side))
            box.bind("<MouseWheel>", lambda _: "break")
            self._tip(box, f"What the {side} of the divider shows: a preset, upscaled, or the source enlarged by a "
                           "classic filter. The sidebar shows the preset's settings, to change them or save your "
                           "own. Two sides upscaled compare two settings, or FSR 4 with FSR 3.1.\nOriginal pixels: "
                           + FILTER_HELP["nearest"]
                           + ".\nBilinear: " + FILTER_HELP["linear"] + ".\nBicubic: " + FILTER_HELP["cubic"]
                           + ".\nLanczos: " + FILTER_HELP["lanczos"] + ".\nOff hides this side: the other one "
                           "fills the picture. The label on the picture opens this list too. Drag the divider to "
                           "move it.")
            self.side_boxes[side] = box
        zoom_row = ttk.Frame(self.toolbar)
        zoom_row.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(px(4), 0))
        zoom_row.columnconfigure(2, weight=1, minsize=px(ZOOM_MIN_WIDTH))
        zoom_row.columnconfigure(4, weight=ZOOM_SPARE_WEIGHT)
        ttk.Label(zoom_row, text="Zoom", style="Small.TLabel").grid(row=0, column=0, padx=(0, px(6)))
        self.zoom_fit_var = tk.StringVar(value="fit")
        self.zoom_fit_button = ttk.Radiobutton(zoom_row, text="Fit", value="fit", variable=self.zoom_fit_var,
                                               style="Segment.TRadiobutton", takefocus=False,
                                               command=lambda: self.view.set_zoom("fit"))
        self.zoom_fit_button.grid(row=0, column=1, padx=(0, px(10)))
        self._tip(self.zoom_fit_button, "The whole picture, as large as the view allows. A double-click on the "
                                        "picture switches between Fit and 100%.")
        self.zoom_slider = MarkedSlider(zoom_row, self.theme, ZOOM_WIDTH, on_pick=self._zoom_picked,
                                        on_wheel=lambda up: self.view.set_zoom(self.view.zoom_step(1 if up else -1)))
        self.zoom_slider.grid(row=0, column=2, sticky="ew")
        self._tip(self.zoom_slider, "Zoom: click or drag anywhere up to 1600%, or click a mark. Original shows the "
                                    "source at its own size, one screen pixel per source pixel, the upscaled side "
                                    "shrunk to match (what upscaling changes at the size you watched the original); "
                                    "100% is one screen pixel per upscaled pixel, and from there up every upscaled "
                                    "pixel is drawn as a block, exactly as the file holds it. The wheel here or on "
                                    "the picture zooms in quarter steps, stopping at Fit and the marks.")
        self.zoom_label = ttk.Label(zoom_row, text="", style="Small.TLabel", width=5, anchor="w")
        self.zoom_label.grid(row=0, column=3, padx=(px(6), 0))
        self.view = VideoView(main, self.theme, on_browse=self.browse, on_zoom_changed=self._zoom_changed,
                              on_side_click=self._open_side_box)
        self.view.grid(row=2, column=0, sticky="nsew")
        self.view.bind("<Configure>", lambda _: self._update_zoom_controls(), add="+")
        self.transport = Transport(main, self.theme, on_toggle=self.toggle_play, on_seek=self.seek_to,
                                   on_step=self.step, on_volume=self._volume_changed, on_speed=self._speed_changed,
                                   on_fullscreen=self.toggle_fullscreen)
        self.transport.grid(row=3, column=0, sticky="ew", pady=(px(4), 0))
        self.transport.grid_remove()
        self._tip(self.transport.play_button, "Play or pause (Space). ← and → jump 5 seconds.")
        self._tip(self.transport.frame_back_button, "One frame back (,). Hold to keep stepping.")
        self._tip(self.transport.frame_forward_button, "One frame forward (.). Hold to keep stepping.")
        self._tip(self.transport.speed_bar, "Playing speed: drag it, or click a mark ([, ] and the wheel go from "
                                            "mark to mark). 0 pauses. The sound plays from 0.25× to 2×, at its own "
                                            "pitch; otherwise it is silent. Faster than 2× and backwards, "
                                            "playback slows down when decoding cannot keep up; FSR makes about 20 "
                                            "frames a second at 4K with its best motion, so fast speeds wait more.")
        self._tip(self.transport.reverse_button, "Play backwards at the speed of the slider (R).")
        self._tip(self.transport.speed_label, "Click to play forwards at normal speed again (Backspace).")
        self._tip(self.transport.volume_button, "Mute (M). The wheel here, ↑ and ↓ change the volume.")
        self._tip(self.transport.volume_bar, "The volume, on a decibel scale: equal distances sound like equal "
                                             "steps (1% is −40 dB, 10% −20 dB). Click or drag it, or click a mark; "
                                             "the wheel (2 dB) and ↑ ↓ (4 dB) change it too.")
        self._tip(self.transport.fullscreen_button, "Full screen (F11 or F). Esc leaves it.")
        self._tip(self.transport.bar, "Click or drag to jump. The lighter bar is what is upscaled; the small tick is "
                                      "where the upscaler works.")
        self._tip(self.transport.strip, "Bright pictures are upscaled, dim ones are not yet. Click to jump there.")
        self.log_panel = self._build_log(main)
        self.log_panel.grid(row=4, column=0, sticky="ew", pady=(px(10), 0))
        self.log_panel.grid_remove()
        return main

    def _build_log(self, parent: ttk.Frame) -> tk.Frame:
        palette, px = self.theme.palette, self.theme.px
        panel = tk.Frame(parent, background=palette.field, highlightthickness=1,
                         highlightbackground=palette.border, highlightcolor=palette.border)
        panel.columnconfigure(0, weight=1)
        self.log_text = tk.Text(panel, height=8, wrap="word", background=palette.field,
                                foreground=palette.text_secondary, insertbackground=palette.text,
                                selectbackground=palette.accent_soft, selectforeground=palette.text,
                                relief="flat", borderwidth=0, highlightthickness=0, font=self.theme.fonts["mono"],
                                padx=px(12), pady=px(8), state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(panel, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, px(2)), pady=px(2))
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.tag_configure("error", foreground=palette.error_text)
        return panel

    def _build_footer(self) -> ttk.Frame:
        px = self.theme.px
        footer = ttk.Frame(self.root, padding=(px(20), px(4), px(20), px(14)))
        footer.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(footer, style="Thin.Horizontal.TProgressbar", maximum=1000)
        self.progress.grid(row=0, column=0, columnspan=6, sticky="ew", pady=(0, px(10)))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(footer, textvariable=self.status_var, style="Status.TLabel").grid(row=1, column=0, sticky="w")
        self.log_button = ttk.Button(footer, text="Show log", style="Ghost.TButton", command=self.toggle_log,
                                     takefocus=False)
        self.log_button.grid(row=1, column=1, padx=(px(8), 0))
        self.open_button = ttk.Button(footer, text="Open", style="Secondary.TButton", command=self.open_result)
        self.open_button.grid(row=1, column=2, padx=(px(8), 0))
        self._tip(self.open_button, "Open the saved file in your usual player.")
        self.folder_button = ttk.Button(footer, text="Show in folder", style="Secondary.TButton",
                                        command=self.show_result_in_folder)
        self.folder_button.grid(row=1, column=3, padx=(px(8), 0))
        self.cancel_button = ttk.Button(footer, text="Stop", style="Secondary.TButton", command=self.cancel)
        self.cancel_button.grid(row=1, column=4, padx=(px(8), 0))
        self._tip(self.cancel_button, "Stop upscaling the rest (Esc). What is done is kept while the file stays "
                                      "open: Upscale carries on from there.")
        self.start_button = ttk.Button(footer, text="Upscale", style="Accent.TButton", command=self.start)
        self.start_button.grid(row=1, column=5, padx=(px(8), 0))
        self._tip(self.start_button, "Upscale the whole file and save it (Ctrl+Enter). Parts already upscaled "
                                     "while watching are reused.")
        return footer

    def _place_window(self) -> None:
        px = self.theme.px
        screen_width, screen_height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        width = min(px(1280), int(screen_width * 0.92))
        height = min(px(860), int(screen_height * 0.88))
        self.root.minsize(min(px(960), width), min(px(640), height))
        self.root.geometry(f"{width}x{height}+{(screen_width - width) // 2}+{max(0, (screen_height - height) // 3)}")

    def _enable_drop(self):
        if os.name != "nt":
            return None
        from .ui_dnd import FileDrop
        try:
            return FileDrop(self.root, lambda paths: self.queue.put(("drop", paths)))
        except OSError as error:
            log.emit(f"[ui] drag and drop unavailable: {error}")
            return None

    def _bind_keys(self) -> None:
        root = self.root
        root.bind("<Control-o>", lambda _: self.browse())
        root.bind("<Control-Return>", lambda _: self.start())
        root.bind("<Control-l>", lambda _: self.toggle_log())
        root.bind("<Control-b>", lambda _: self.toggle_sidebar())
        root.bind("<Escape>", lambda _: self.toggle_fullscreen() if self.fullscreen else self.cancel())
        root.bind("<F11>", lambda _: self.toggle_fullscreen())
        for sequence, action in (("<space>", self.toggle_play), ("<f>", self.toggle_fullscreen),
                                 ("<Left>", lambda: self.seek_by(-SEEK_STEP_SECONDS)),
                                 ("<Right>", lambda: self.seek_by(SEEK_STEP_SECONDS)),
                                 ("<comma>", lambda: self.step(-1)), ("<period>", lambda: self.step(1)),
                                 ("<Home>", lambda: self.seek_to(0, True)),
                                 ("<End>", lambda: self.seek_to(self._total() - 1, True)),
                                 ("<m>", self.transport.toggle_mute),
                                 ("<Up>", lambda: self.transport.change_volume(VOLUME_KEY_DECIBELS)),
                                 ("<Down>", lambda: self.transport.change_volume(-VOLUME_KEY_DECIBELS)),
                                 ("<bracketleft>", lambda: self.transport.change_speed(False)),
                                 ("<bracketright>", lambda: self.transport.change_speed(True)),
                                 ("<r>", self.transport.toggle_reverse),
                                 ("<BackSpace>", lambda: self.transport.set_speed(1.0, notify=True))):
            root.bind(sequence, self._shortcut(action))
        root.bind_all("<Motion>", self._pointer_moved, add="+")
        root.bind_all("<ButtonPress>", lambda _: self._set_pointer_pressed(True), add="+")
        root.bind_all("<ButtonRelease>", lambda _: self._set_pointer_pressed(False), add="+")
        root.bind_all("<ButtonPress-1>", self._focus_window, add="+")
        root.bind_class("TCombobox", "<<ComboboxSelected>>", self._dropdown_picked, add="+")

    def _focus_window(self, event: tk.Event) -> None:
        """A click anywhere but in a text field gives the keyboard back to the window, so Space and the arrows
        play and seek (a clicked button, chip or checkbox would keep it, and Space would press it again)."""
        if not isinstance(event.widget, (str, tk.Entry, ttk.Entry, tk.Text)):
            self.root.focus_set()

    def _dropdown_picked(self, event: tk.Event) -> None:
        """After a pick from any dropdown, the keyboard goes back to the window too."""
        event.widget.selection_clear()
        self.root.focus_set()

    def _shortcut(self, action):
        """Run `action` unless the key goes to a text field or a list."""
        def handler(event: tk.Event):
            if isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Listbox, tk.Text)):
                return None
            action()
            return "break"
        return handler

    # ---- settings -------------------------------------------------------------------------------

    def _scale_for(self, choice: str) -> float | None:
        """The per-dimension factor a size chip stands for with the current file (None without a file)."""
        if choice == "custom":
            return round(float(self.custom_scale.get()), 2)
        if choice in TARGET_SHORT_SIDES:
            if self.info is None:
                return None
            return TARGET_SHORT_SIDES[choice] / min(self.info.display_width, self.info.height)
        return float(choice)

    def _options(self) -> JobOptions:
        how = self._saved_how()
        output = self.output_var.get().strip().strip('"')
        return JobOptions(
            input=self.input_path, output=Path(output) if output else None, scale=self._scale_for(self.size_var.get()),
            flow=how.flow, flow_preset=how.flow_preset, sharpness=how.sharpness,
            codec=ENCODER_CHOICES[self.encoder_choice.current()][1],
            upscaler=how.upscaler)

    def _side_how(self, side: str) -> Preset | None:
        """How `side` is upscaled; None when it shows a filter or nothing."""
        return self.side_blocks[side].how() if self.side_choices[side] == UPSCALED else None

    def _saved_side(self) -> str | None:
        """The side the sidebar sets and Upscale saves: the one chosen when both are upscaled, the only one, or None
        without any."""
        upscaled = [side for side in DEFAULT_SIDES if self.side_choices[side] == UPSCALED]
        if len(upscaled) == 2:
            return self.save_side_var.get()
        return upscaled[0] if upscaled else None

    def _saved_how(self) -> Preset:
        """How the file is saved: what a whole-file upscale started with while it runs (the sides stay free to change
        then), otherwise the saved side; with no side upscaled, the output size still needs an answer."""
        if (self.batch or self.saving) and self.renderer is not None:
            settings = self.renderer.settings
            return Preset("", settings.upscaler, settings.flow, settings.flow_preset, settings.sharpness)
        side = self._saved_side()
        return self._side_how(side) if side is not None else BEST

    def _plan(self) -> JobPlan | None:
        """The saved side's plan, None without a file; raises ValueError when it cannot be upscaled as set."""
        if self.info is None or self.input_path is None:
            return None
        options = self._options()
        problem = self._upscaler_problem(options.upscaler)
        if problem is not None:
            raise ValueError(problem)
        return plan_from_info(options, self.info)

    def _upscaler_problem(self, upscaler: str | None) -> str | None:
        """Why this graphics card cannot run `upscaler` (FSR 4 needs RDNA 4), or None."""
        return self.fsr4_missing if upscaler is None else None

    def _preset_label(self, preset: Preset) -> str:
        """A preset's entry in the lists, marked when this card cannot run its upscaler."""
        return f"{preset.name} · {NOT_HERE}" if self._upscaler_problem(preset.upscaler) else preset.name

    def _render_settings(self) -> RenderSettings | None:
        try:
            plan = self._plan()
        except ValueError:
            return None
        if plan is None:
            return None
        options = self._options()
        return RenderSettings(plan.output_size, options.flow, options.flow_preset, options.sharpness, options.codec,
                              options.upscaler)

    def _settings_changed(self) -> None:
        """A setting moved: update the labels now, restart the upscale once the setting stops moving."""
        self._refresh()
        if self._settings_job is not None:
            self.root.after_cancel(self._settings_job)
        self._settings_job = self.root.after(SETTINGS_DELAY_MS, self._apply_settings)

    def _apply_settings(self) -> None:
        """Make what the sides say real; a whole-file upscale carries on as it started."""
        self._settings_job = None
        settings = None if self.busy else self._render_settings()
        if settings is not None and (self.player is not None or self.image_source is not None):
            if self.renderer is not None:
                try:
                    self.renderer.change_settings(settings)
                except OSError as error:
                    log.emit(f"[ui] error: {error}")
                    self._show_banner(f"The new settings could not be applied: {self._first_line(error)}")
                    return
                self.transport.set_thumbnails(upscaled=self.renderer.thumbnails())
            elif self.player is not None:
                self._start_renderer(settings)  # the settings at load could not be upscaled (a size the encoder refuses)
            self.view.set_media((self.info.width, self.info.height), settings.output_size)
        self._update_layers()
        self._update_progress()

    def _processor_changed(self) -> None:
        """Threads or priority moved: show it now, apply it once the slider stops."""
        threads = int(round(float(self.threads_var.get())))
        self.threads_label.configure(text=f"{threads} of {usage.CPU_THREADS}")
        if self._processor_job is not None:
            self.root.after_cancel(self._processor_job)
        self._processor_job = self.root.after(SETTINGS_DELAY_MS, self._apply_processor)

    def _apply_processor(self) -> None:
        self._processor_job = None
        applied = usage.apply(int(round(float(self.threads_var.get()))), self.priority_var.get())
        log.emit(f"[ui] processor: {applied.threads} threads ({applied.opencv_threads} for optical flow), "
                 f"{usage.PRIORITIES[applied.priority].label.lower()} priority")
        if self.renderer is not None:
            self.renderer.restart_processes()

    def _output_edited(self, *_args) -> None:
        if not self._setting_output:
            self.output_custom = True
            self._refresh()

    def _set_output(self, path: Path) -> None:
        self._setting_output = True
        self.output_var.set(str(path))
        self._setting_output = False
        self.output_entry.xview_moveto(1.0)

    def _clear_output(self) -> None:
        self._setting_output = True
        self.output_var.set("")
        self._setting_output = False
        self.output_custom = False

    def _set_state(self, widget: ttk.Widget, enabled: bool) -> None:
        widget.state(["!disabled"] if enabled else ["disabled"])

    def _refresh(self) -> None:
        """Bring every derived label and widget state in line with the file, the settings and the work."""
        idle = not self.busy
        image = self.input_path is not None and is_image(self.input_path)
        for widget in self.setting_widgets:
            self._set_state(widget, idle)
        self.choose_button.configure(text="Choose another file…" if self.input_path is not None else "Choose file…")
        if self.info is not None:
            for value, chip in self.size_chips.items():
                scale = self._scale_for(value)
                self._set_state(chip, idle and (value == "custom" or 1.0 - 1e-6 <= scale <= 3.0 + 1e-6))
            if self.size_chips[self.size_var.get()].instate(["disabled"]) and idle:
                self.size_var.set("2")
        else:
            for value, chip in self.size_chips.items():
                self._set_state(chip, idle and value not in TARGET_SHORT_SIDES)
        if self.size_var.get() == "custom":
            self.custom_row.grid()
        else:
            self.custom_row.grid_remove()
        self.custom_label.configure(text=f"{self.custom_scale.get():.2f}×")
        self._set_state(self.encoder_choice, idle and not image)
        self.encoder_note.configure(text="" if image else ENCODER_CHOICES[self.encoder_choice.current()][2])
        saved = self._saved_side()
        for side, block in self.side_blocks.items():
            match = self.presets.matching(self._how_settings(side))
            preset = self._preset_label(match) if match is not None else CUSTOM_PRESET
            block.refresh(side == saved, image, idle, preset, match is not None and not match.builtin)
        self._update_side_boxes()
        show_widget(self.save_row, all(self._side_how(side) is not None for side in DEFAULT_SIDES))
        show_widget(self.save_note, saved is None)
        plan, problem = None, None
        if self.input_path is not None and not self.output_custom:
            self._set_output(self._default_output())
        try:
            plan = self._plan()
        except ValueError as error:
            problem = str(error)
        if plan is not None:
            (width, height), (out_width, out_height) = plan.render_size, plan.output_size
            self.size_label.configure(text=f"{width} × {height}   →   {out_width} × {out_height}",
                                      style="Strong.TLabel")
            across, down = plan.scales
            self.size_note.configure(text=f"{plan.describe_scale('×')} per side · {across * down:.1f}× the pixels")
            self.audio_note.configure(text="" if image else describe_audio_plan(self.info, plan.output))
        else:
            self.size_label.configure(style="Faint.TLabel",
                                      text="Cannot upscale with these settings." if problem
                                      else "Reading the file…" if self.input_path is not None
                                      else "The output size appears once a file is loaded.")
            self.size_note.configure(text="")
            self.audio_note.configure(text="")
        if problem:
            self.size_note.configure(text=problem)
        ready = (idle and plan is not None and problem is None and self.gpu_ready is not False
                 and self._saved_side() is not None)
        self._set_state(self.start_button, ready)
        self._set_state(self.cancel_button, self.batch and not self.saving)
        if self.batch:
            self.cancel_button.grid()
        else:
            self.cancel_button.grid_remove()
        done = idle and self.last_saved is not None and self.last_saved.exists()
        for button in (self.open_button, self.folder_button):
            if done:
                button.grid()
            else:
                button.grid_remove()
        self._comparing_shown = self.view.comparing
        self._update_zoom_controls()

    def _default_output(self) -> Path:
        """`<name>.fsr4.<ext>` next to the file, numbered when that exists: a default never replaces a file."""
        current = self.output_var.get().strip()
        path = default_output(self.input_path)
        if current and not is_image(self.input_path) and Path(current).suffix.lower() in (".mkv", ".mov"):
            path = path.with_suffix(Path(current).suffix)
        return free_name(path)

    def _zoom_changed(self, _zoom) -> None:
        self._update_zoom_controls()

    def _zoom_fraction(self, scale: float) -> float:
        """Where `scale` lies along the zoom slider: its range (`VideoView.zoom_range`) in octaves."""
        low, high = self.view.zoom_range()
        return min(1.0, max(0.0, math.log2(scale / low) / math.log2(high / low)))

    def _update_zoom_controls(self) -> None:
        """The Fit button (lit at Fit), the zoom slider's marks (Original moves with the file, and the whole scale
        with Fit) and knob, and the zoom in percent; nothing to zoom, they are disabled."""
        comparing = self.view.comparing
        self._set_state(self.zoom_fit_button, comparing)
        self.zoom_slider.set_enabled(comparing)
        self.zoom_fit_var.set("fit" if comparing and self.view.zoom == "fit" else "")
        if not comparing:
            self.zoom_label.configure(text="")
            return
        self.zoom_slider.set_marks([Mark(self._zoom_fraction(self.view.zoom_scale(level)), label, level, rank,
                                         emphasis=level == 1) for level, label, rank in ZOOM_TICKS])
        scale = self.view.zoom_scale(self.view.zoom)
        self.zoom_slider.show(self._zoom_fraction(scale), self.view.zoom)
        self.zoom_label.configure(text=f"{scale * 100:.0f}%")

    def _zoom_picked(self, fraction: float, mark: Mark | None) -> None:
        """The zoom slider clicked or dragged: a mark's zoom, else the scale there to the percent."""
        if mark is not None:
            self.view.set_zoom(mark.value)
            return
        low, high = self.view.zoom_range()
        self.view.set_zoom(self.view.zoom_named(round(low * (high / low) ** fraction, 2)))

    # ---- what each side shows -------------------------------------------------------------------

    def _still(self) -> bool:
        """Whether the sides may offer what only suits still images: an image, or no file yet."""
        return self.input_path is None or is_image(self.input_path)

    def _how_settings(self, side: str) -> RenderSettings:
        """How `side` upscales, which presets hold (the size is a placeholder)."""
        return self.side_blocks[side].how().applied_to(RenderSettings((0, 0)))

    def _offered_presets(self) -> list[Preset]:
        """The presets this file can use: a video leaves out the models for still images."""
        still = self._still()
        return [preset for preset in self.presets.all if still or preset.upscaler not in STILL_ONLY]

    def _side_entries(self) -> list[tuple[str, str]]:
        """(label, choice) of the side lists: the presets, the source filters, off."""
        return ([(self._preset_label(preset), f"preset:{preset.name}") for preset in self._offered_presets()]
                + [(SOURCE_LABELS[name], name) for name in REFERENCE_FILTERS]
                + [(OFF_LABEL, OFF)])

    def _update_preset_lists(self) -> None:
        """The lists after presets or the file changed; a model this file cannot use leaves its side."""
        labels = [label for label, _ in self._side_entries()]
        names = [self._preset_label(preset) for preset in self._offered_presets()]
        for side, block in self.side_blocks.items():
            block.load(block.how(), self._still())
            block.set_presets(names)
            self.side_boxes[side].configure(values=labels)
        self._update_side_boxes()

    def _update_side_boxes(self) -> None:
        """What the side lists say: the filter, the preset these settings are, or else the settings."""
        for side, box in self.side_boxes.items():
            choice = self.side_choices[side]
            if choice == OFF:
                box.set(OFF_LABEL)
            elif choice in SOURCE_LABELS:
                box.set(SOURCE_LABELS[choice])
            else:
                box.set(self._side_label(self._how_settings(side)))

    def _show_on_side(self, side: str, choice: str) -> None:
        """Show an entry of the side lists on `side`: a preset loads into its settings."""
        if choice.startswith("preset:"):
            preset = self.presets.find(choice.removeprefix("preset:"))
            if preset is None:
                raise ValueError(f"no preset is called {choice.removeprefix('preset:')}")
            self.side_blocks[side].load(preset, self._still())
            choice = UPSCALED
        self.side_choices[side] = choice

    def _show_source_only(self) -> None:
        """Open the next file without upscaling: sides showing a preset are turned off, and when no side would
        show anything the first of them shows the source bicubic. Their settings stay, for when a preset is
        chosen again."""
        upscaled = [side for side in DEFAULT_SIDES if self.side_choices[side] == UPSCALED]
        if not upscaled:
            return
        for side in upscaled:
            self._show_on_side(side, OFF)
        if all(choice == OFF for choice in self.side_choices.values()):
            self._show_on_side(upscaled[0], DEFAULT_SIDES["left"])  # the source, through the default filter

    def _side_chosen(self, side: str) -> None:
        box = self.side_boxes[side]
        self._show_on_side(side, self._side_entries()[box.current()][1])
        other = "right" if side == "left" else "left"
        if self.side_choices[side] == OFF and self.side_choices[other] == OFF:
            self._show_on_side(other, DEFAULT_SIDES[other])  # never both: the other side comes back
        box.selection_clear()
        self.root.focus_set()  # back to the window, so Space and the arrows play and seek again
        self._refresh()
        if self._settings_job is not None:
            self.root.after_cancel(self._settings_job)
        self._apply_settings()

    def _open_side_box(self, side: str) -> None:
        """A label on the picture was clicked: open that side's dropdown."""
        self.root.tk.call("ttk::combobox::Post", self.side_boxes[side])

    def _preset_chosen(self, side: str) -> None:
        """Load a preset into the settings of `side`."""
        block = self.side_blocks[side]
        index = block.preset_choice.current()  # the entries may carry a mark after the name
        preset = self._offered_presets()[index] if index >= 0 else None
        block.preset_choice.selection_clear()
        self.root.focus_set()
        if preset is not None:
            block.load(preset, self._still())
            self._settings_changed()

    def _begin_save_preset(self, side: str) -> None:
        suggestion = self._how_settings(side).describe()
        if self.presets.check_name(suggestion) is not None:
            suggestion = "My preset"
        self.side_blocks[side].ask_name(suggestion)

    def _finish_save_preset(self, side: str) -> None:
        block = self.side_blocks[side]
        problem = self.presets.check_name(block.name_var.get())
        if problem is not None:
            block.name_note.configure(text=problem)
            return
        try:
            preset = self.presets.save(block.name_var.get(), self._how_settings(side))
        except OSError as error:
            log.emit(f"[ui] error: cannot save the preset: {error}")
            self._show_banner(f"The preset could not be saved: {self._first_line(error)}")
            return
        log.emit(f"[ui] saved the preset {preset.name} in {self.presets.path}")
        block.hide_name()
        self._update_preset_lists()
        self._update_layers()  # the label on the picture names it
        self._refresh()

    def _cancel_save_preset(self, side: str) -> str:
        self.side_blocks[side].hide_name()
        return "break"

    def _delete_preset(self, side: str) -> None:
        """Delete the preset of the user's that `side`'s settings are; the side keeps the settings."""
        preset = self.presets.matching(self._how_settings(side))
        if preset is None or preset.builtin:
            return
        try:
            self.presets.delete(preset.name)
        except OSError as error:
            log.emit(f"[ui] error: cannot delete the preset: {error}")
            self._show_banner(f"The preset could not be deleted: {self._first_line(error)}")
            return
        self._update_preset_lists()
        self._update_layers()
        self._refresh()

    def _side_label(self, settings: RenderSettings) -> str:
        """What the label on the picture says: the preset these settings are, or the settings; marked when this
        card cannot run them."""
        match = self.presets.matching(settings)
        if match is not None:
            return self._preset_label(match)
        label = settings.describe()
        return f"{label} · {NOT_HERE}" if self._upscaler_problem(settings.upscaler) else label

    def _update_layers(self) -> None:
        """Turn what each side shows into the view's layers, and have those versions upscaled and played. Two
        sides that come to the same settings share one version."""
        renderer = self.renderer
        current = renderer.settings if renderer is not None else self._render_settings()
        image = self.image_source is not None
        sides, labels, layers = {}, {}, {}
        for side, choice in self.side_choices.items():
            how = self._side_how(side)
            if how is None:
                sides[side] = choice
                continue
            if self._upscaler_problem(how.upscaler) is not None:
                sides[side] = UNAVAILABLE  # never handed to the renderer
                labels[side] = self._side_label(self._how_settings(side))
                continue
            if current is None:
                sides[side] = UNAVAILABLE
                continue
            settings = how.applied_to(current)
            labels[side] = self._side_label(settings)
            if image:
                settings = replace(settings, flow="zero", flow_preset="fast")  # no motion in a single picture
            key = f"upscaled:{settings.upscaler}:{settings.flow}:{settings.flow_preset}:{settings.sharpness}"
            sides[side] = key
            layers[key] = settings
        self._shown_layers = layers
        self.view.set_sides(sides, labels)
        if renderer is not None:
            renderer.show(list(layers.values()))
            # During a whole-file upscale only the current version is made: the others never hold playback up.
            self.player.attach(renderer.timeline, current.output_size,
                               {key: (renderer.timeline_of(settings), not self.batch or settings == current)
                                for key, settings in layers.items()})
        elif self.player is not None:
            self.player.attach(None, current.output_size if current is not None else (self.info.width,
                                                                                       self.info.height))
        if self.image_previewer is not None:
            self.image_previewer.request(list(layers.values()))
        if image:
            self._show_image()

    def _show_image(self) -> None:
        """The open image and every upscaled version shown that is ready."""
        previewer = self.image_previewer
        layers = {key: previewer.result(settings) if previewer is not None else None
                  for key, settings in self._shown_layers.items()}
        working = previewer is not None and previewer.working
        self.view.show_still(self.image_source, layers, "upscaling…" if working else "not upscaled",
                             spinner=working)

    # ---- files ----------------------------------------------------------------------------------

    def browse(self) -> None:
        if self.busy:
            return
        initial = str(self.input_path.parent) if self.input_path else None
        chosen = filedialog.askopenfilename(parent=self.root, title="Choose a video or an image",
                                            filetypes=OPEN_TYPES, initialdir=initial)
        if chosen:
            self.load(Path(chosen))

    def browse_output(self) -> None:
        if self.input_path is None:
            return
        current = Path(self.output_var.get().strip() or self._default_output())
        types = IMAGE_SAVE_TYPES if is_image(self.input_path) else VIDEO_SAVE_TYPES
        chosen = filedialog.asksaveasfilename(parent=self.root, title="Save the upscaled file as",
                                              initialdir=str(current.parent), initialfile=current.name,
                                              defaultextension=current.suffix, filetypes=types)
        if chosen:
            self.output_custom = True
            self.output_confirmed = Path(chosen)
            self._set_output(Path(chosen))
            self._refresh()

    def load(self, path: Path) -> bool:
        """Probe `path` (on a worker thread), then set up its preview and playback. False, with the reason in the
        banner, when it cannot be opened now."""
        if self.busy:
            self._show_banner("The upscaled video is being saved; open another file once it is saved."
                              if self.saving else "An upscale is running. Stop it before opening another file.")
            return False
        path = Path(path)
        if path.is_dir():
            self._show_banner(f"{path.name} is a folder: open a video or an image.")
            return False
        if not path.is_file():
            self._show_banner(f"Not found: {path}")
            return False
        self._close_media()
        self.load_token += 1
        self.input_path, self.info, self.last_saved = path, None, None
        if not self.upscale_on_open_var.get():
            self._show_source_only()
        self._update_preset_lists()
        self.output_custom = False
        self._hide_banner()
        self.source_name.configure(text=path.name)
        self.source_details.configure(text="Reading…")
        self.view.show_message(f"Reading {path.name}…", spinner=True)
        self.progress.configure(value=0)
        self.status_var.set("Reading the file…")
        self._refresh()
        threading.Thread(target=self._load_worker, args=(self.load_token, path), daemon=True).start()
        return True

    def _load_worker(self, token: int, path: Path) -> None:
        try:
            info = probe(path)
            image = to_bgra(read_frame(path, info)) if is_image(path) else None
            self.queue.put(("loaded", token, path, info, image))
        except Exception as error:  # noqa: BLE001 - any failure is shown in the window
            self.queue.put(("load_failed", token, path, error))

    def _close_media(self) -> None:
        """Stop playback and upscaling of the current file and delete its pieces."""
        if self.keyframes is not None:
            self.keyframes.close()  # first: a player or renderer waiting for the index stops at once
        if self.player is not None:
            self.player.close()
            self.player = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.image_previewer is not None:
            self.image_previewer.close()
            self.image_previewer = None
        self.image_source = None
        self._shown_layers = {}
        self.keyframes = None
        self.source_thumbnails = {}
        self.render_position, self.render_speed = None, 0.0
        self.transport.grid_remove()
        self.transport.place_forget()  # in full screen it is placed over the picture instead
        self.root.title(APP_NAME)

    def _start_video(self, path: Path, info: MediaInfo) -> None:
        self.keyframes = KeyframeIndex(path, info)
        settings = self._render_settings()
        output_size = settings.output_size if settings is not None else (info.width, info.height)
        self.view.set_media((info.width, info.height), output_size)
        self.player = Player(self.view, path, info, self.keyframes,
                             lambda kind, *values: self.queue.put(("player", kind, values)))
        self.player.set_volume(self.transport.volume, self.transport.muted)  # the slider's, kept between files
        if settings is not None:
            self._start_renderer(settings)
        self._update_layers()
        self.transport.set_media(self._total(), float(info.frame_rate))
        if self.fullscreen:
            self._flash_controls()  # over the picture, hiding at rest
        else:
            self.transport.grid()
        threading.Thread(target=self._source_thumbnail_worker, args=(self.load_token, path, info),
                         daemon=True).start()

    def _start_renderer(self, settings: RenderSettings) -> None:
        """Start upscaling the open video with `settings` (valid ones), unless FSR 4 cannot start here."""
        if self.gpu_ready is False:
            return
        sdk_dir = self._sdk_dir()
        if sdk_dir is not None:
            self._use_output_folder()
            self.renderer = Renderer(self.input_path, self.info, settings, sdk_dir,
                                     lambda kind, *values: self.queue.put(("render", kind, values)), self.keyframes)

    def _use_output_folder(self) -> None:
        """Put the open file's temporary files (pieces, Real-ESRGAN's pictures) in its output's folder, and delete
        the pieces a crash left there."""
        folder = temporary.use_beside(Path(self.output_var.get().strip() or self._default_output()))
        try:
            for directory in clean_stale_directories([folder]):
                log.emit(f"[ui] removed pieces left by an earlier session: {directory}")
        except OSError as error:
            log.emit(f"[ui] could not clean old pieces in {folder}: {error}")

    def _sdk_dir(self) -> Path | None:
        """The folder of the FSR DLLs, or None (said in the banner) when they are missing."""
        try:
            return find_sdk_dir(None)
        except FileNotFoundError as error:
            log.emit(f"[ui] error: {error}")
            self._show_banner(f"FSR 4 cannot start: {self._first_line(error)}")
            return None

    def _why_not_upscaling(self) -> str:
        """Why the open video has no renderer, for Play and Upscale."""
        if self.gpu_ready is False:
            return "FSR 4 cannot start on this machine (see the GPU status)"
        try:
            self._plan()
        except ValueError as error:
            return str(error)
        return "the upscaler is not running (see the log)"

    def _source_thumbnail_worker(self, token: int, path: Path, info: MediaInfo) -> None:
        usage.lower_current_thread()
        total = info.frame_count or 1
        try:
            for index in range(SOURCE_THUMBNAILS):
                if token != self.load_token:
                    return
                frame = int(index * total / SOURCE_THUMBNAILS)
                thumbnail = read_thumbnail(path, info, frame, THUMBNAIL_HEIGHT)
                if thumbnail is not None:
                    self.queue.put(("source_thumbnail", token, frame, thumbnail))
        except Exception as error:  # noqa: BLE001 - thumbnails are a nicety: log, keep going without them
            log.emit(f"[ui] filmstrip pictures unavailable: {error}")

    def _total(self) -> int:
        if self.renderer is not None:
            return self.renderer.timeline.total
        if self.player is not None:
            return self.player.total
        return (self.info.frame_count or 1) if self.info is not None else 1

    # ---- playback -------------------------------------------------------------------------------

    def toggle_play(self) -> None:
        if self.player is None:
            return
        if self.transport.speed == 0:
            self.transport.set_pace(1.0)  # the slider stopped it: Play means normal speed
            self.player.play()
        else:
            self.player.toggle()

    def seek_to(self, frame: int, final: bool) -> None:
        """Jump to `frame`. While dragging (not final) the player gets a seek every DRAG_SEEK_SECONDS at most
        and the renderer stays where it is; the release moves both."""
        if self.player is None:
            return
        now = time.perf_counter()
        if not final and now - self._last_drag_seek < DRAG_SEEK_SECONDS:
            return
        self._last_drag_seek = now
        self.player.seek(frame)
        if final and self.renderer is not None:
            self.renderer.set_target(frame)

    def seek_by(self, seconds: float) -> None:
        if self.player is not None and self.info is not None:
            self.seek_to(self.player.position + int(round(seconds * float(self.info.frame_rate))), True)

    def step(self, frames: int) -> None:
        if self.player is not None:
            if self.player.playing:
                self.player.pause()
            self.player.step(frames)

    def _speed_changed(self, speed: float) -> None:
        if self.player is not None:
            self.player.set_speed(speed)

    def _render_target(self, frame: int) -> int:
        """Where the renderer works: at the playhead; playing backwards, on the chunk just behind the stretch
        ready behind the playhead, until REVERSE_AHEAD_SECONDS are ready there."""
        player = self.player
        if player is None or not player.playing or player.speed > 0 or self.renderer is None:
            return frame
        rate = float(self.info.frame_rate)
        current = self.renderer.settings
        upscaled = [settings for settings in self._shown_layers.values() if not self.batch or settings == current]
        timelines = [self.renderer.timeline_of(settings) for settings in upscaled] or [self.renderer.timeline]
        edge = max(timeline.ready_back_to(frame) for timeline in timelines)
        if frame - edge >= REVERSE_AHEAD_SECONDS * rate:
            return frame
        return max(0, edge - int(round(REVERSE_CHUNK_SECONDS * rate)))

    def _volume_changed(self, volume: float, muted: bool) -> None:
        if self.player is not None:
            self.player.set_volume(volume, muted)

    def toggle_sidebar(self) -> None:
        if self.fullscreen:
            return
        self.sidebar_visible = not self.sidebar_visible
        with self._behind_curtain():
            if self.sidebar_visible:
                self.sidebar.grid()
                self.unfold_button.grid_remove()
            else:
                self.sidebar.grid_remove()
                self.unfold_button.grid()
            self.main.configure(padding=self._main_padding())

    def toggle_fullscreen(self) -> None:
        """Full screen on the monitor the window is on, or back to the window exactly as it was."""
        self.fullscreen = not self.fullscreen
        px = self.theme.px
        if not self.fullscreen:
            self.screen.leave()
            self.root.update_idletasks()  # the rows come back laid out once, at the window's own size
        for widget in (self.footer, self.toolbar):
            if self.fullscreen:
                widget.grid_remove()
            else:
                widget.grid()
        if self.fullscreen:
            self.sidebar.grid_remove()
            self.banner.grid_remove()
            self.log_panel.grid_remove()
            self.main.configure(padding=0)
        else:
            if self.sidebar_visible:
                self.sidebar.grid()
            if self.log_visible:
                self.log_panel.grid()
            if self.banner_label.cget("text"):
                self.banner.grid()  # a message that came during full screen waited for the window
            self.main.configure(padding=self._main_padding())
        self.view.set_fullscreen(self.fullscreen)
        self.transport.set_fullscreen(self.fullscreen)
        if self.info is not None and not is_image(self.input_path):
            if self.fullscreen:
                self.transport.grid_remove()
                self._place_controls()
            else:
                self.transport.place_forget()
                self.transport.grid()
        for name in ("_hide_job", "_leave_job"):
            job = getattr(self, name)
            if job is not None:
                self.root.after_cancel(job)
                setattr(self, name, None)
        self._cursor_hidden = False
        self.view.configure(cursor="")
        if self.fullscreen:
            self.root.update_idletasks()
            try:
                self.screen.enter()
            except OSError as error:
                log.emit(f"[ui] error: {error}")
                self.toggle_fullscreen()  # back to the window with all its controls
                self._show_banner(f"Full screen is unavailable: {self._first_line(error)}")
                return
        self._repaint_everything()
        if self.fullscreen:
            self._rest_later()

    @contextmanager
    def _behind_curtain(self) -> Iterator[None]:
        """Change the layout out of sight: the window's old picture stays in front until the new layout is drawn,
        then dissolves into it."""
        curtain = Curtain(self.root)
        try:
            yield
            self.root.update_idletasks()
        finally:
            # Idle callbacks run once no event is left, and the paints that follow are idle callbacks too: the
            # second one comes after them.
            self.root.after_idle(self.root.after_idle, curtain.fade)

    def _repaint_everything(self) -> None:
        """After a big layout change: lay out now, then have every widget repaint where it now is, now and once
        more when the layout has settled, so none is left unpainted (black) or showing the old layout."""
        self.root.update_idletasks()
        self._repaint_widgets()
        if self._repaint_job is not None:
            self.root.after_cancel(self._repaint_job)
        self._repaint_job = self.root.after(REPAINT_SETTLE_MS, self._repaint_widgets)

    def _repaint_widgets(self) -> None:
        """Windows repaints every child window and Tk redraws every widget (an Expose each), whichever missed
        the paint after being shown again."""
        self._repaint_job = None
        repaint(self.root)
        pending = [self.root]
        while pending:
            for child in pending.pop().winfo_children():
                if child.winfo_ismapped() and not isinstance(child, tk.Toplevel):
                    child.event_generate("<Expose>", x=0, y=0, width=child.winfo_width(),
                                         height=child.winfo_height())
                    pending.append(child)

    def _window_configured(self, event: tk.Event) -> None:
        """The window moved or changed size (another monitor, restored from the taskbar): lay it out at once, or
        each level of nested widgets would wait for its own idle pass and what follows the right edge would trail
        it by several steps of a drag, then repaint it all once it stops."""
        if event.widget is not self.root:
            return
        self.root.update_idletasks()
        if self._repaint_job is not None:
            self.root.after_cancel(self._repaint_job)
        self._repaint_job = self.root.after(REPAINT_SETTLE_MS, self._repaint_widgets)

    def _set_pointer_pressed(self, pressed: bool) -> None:
        self._pointer_pressed = pressed

    def _place_controls(self) -> None:
        """Full screen: the playback controls over the bottom of the video."""
        if self.info is not None and not is_image(self.input_path) and not self.transport.winfo_ismapped():
            self.transport.place(in_=self.view, relx=0, rely=1, anchor="sw", relwidth=1)
            self.transport.lift()

    def _pointer_moved(self, event: tk.Event) -> None:
        """Full screen: a moving pointer shows itself again, and near the bottom of the screen the controls;
        leaving the bottom hides them soon after, and resting hides both (`_rest`)."""
        if not self.fullscreen or (event.x_root, event.y_root) == self._last_pointer:
            return  # Windows repeats the position when a window appears or vanishes under a still pointer
        self._last_pointer = (event.x_root, event.y_root)
        if self._cursor_hidden:
            self._cursor_hidden = False
            if str(self.view.cget("cursor")) == "none":
                self.view.configure(cursor="")
        bottom = self.root.winfo_rooty() + self.root.winfo_height()
        if event.y_root >= bottom - self.transport.winfo_reqheight() - self.theme.px(CONTROLS_ZONE_MARGIN):
            if self._leave_job is not None:
                self.root.after_cancel(self._leave_job)
                self._leave_job = None
            self._place_controls()
        elif self.transport.winfo_ismapped() and self._leave_job is None:
            self._leave_job = self.root.after(HIDE_LEAVE_MS, self._left_bottom)
        self._rest_later()

    def _rest_later(self) -> None:
        if self._hide_job is not None:
            self.root.after_cancel(self._hide_job)
        self._hide_job = self.root.after(HIDE_CONTROLS_MS, self._rest)

    def _rest(self) -> None:
        """The pointer has not moved for HIDE_CONTROLS_MS: hide the controls and the pointer (not mid-drag)."""
        self._hide_job = None
        if not self.fullscreen:
            return
        if self._pointer_pressed or self.transport.dragging:
            self._rest_later()
            return
        self.transport.place_forget()
        self._cursor_hidden = True
        self.view.configure(cursor="none")

    def _left_bottom(self) -> None:
        self._leave_job = None
        if self.fullscreen and not self._pointer_pressed and not self.transport.dragging:
            self.transport.place_forget()

    def _flash_controls(self) -> None:
        """Full screen: show the controls (after a pause, say) until the pointer rests."""
        if self.fullscreen:
            self._place_controls()
            self._rest_later()

    # ---- upscaling and saving -------------------------------------------------------------------

    def start(self) -> None:
        if self.busy or self.info is None:
            return
        if self._saved_side() is None:
            self._show_banner("Neither side is upscaled: choose a preset in the Left or Right list.")
            return
        try:
            plan = self._plan()
        except ValueError as error:
            self._show_banner(str(error))
            return
        if plan.output.exists() and plan.output != self.output_confirmed and not messagebox.askyesno(
                APP_NAME, f"{plan.output.name} already exists. Replace it?", icon="warning", parent=self.root):
            return
        self._hide_banner()
        if plan.image_mode:
            self._start_image(self._options())
            return
        if self.renderer is None:
            self._show_banner(f"Cannot upscale: {self._why_not_upscaling()}.")
            return
        if self._settings_job is not None:
            self.root.after_cancel(self._settings_job)
            self._apply_settings()
        self.batch = True
        self.last_saved = None
        self.renderer.set_batch(True)
        self._update_layers()
        if self.player is not None and not self.player.playing:
            self.player.follow_live = True
        self.renderer.want_live = True
        self.status_var.set(f"Upscaling · {plan.output_size[0]} × {plan.output_size[1]}")
        self._refresh()
        if self.renderer.timeline.complete():
            self._save()

    def cancel(self) -> None:
        if self.image_running and not self.image_cancel.is_set():
            self.image_cancel.set()
            self.status_var.set("Cancelling…")
            return
        if self.batch and not self.saving:
            done = self.renderer.timeline.ready_frames() / max(1, self._total())
            self.batch = False
            self.renderer.set_batch(False)
            self._apply_settings()  # the sides may have changed meanwhile
            self.renderer.want_live = False
            self.root.title(APP_NAME)
            self.status_var.set(f"Stopped at {done:.0%} upscaled · Upscale carries on from there")
            self._refresh()

    def _save(self) -> None:
        output = self._plan().output
        self.saving = True
        self.renderer.want_live = False
        self.status_var.set(f"Saving {output.name}…")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self._refresh()
        renderer = self.renderer
        self.worker = threading.Thread(target=self._save_worker, args=(renderer, output), daemon=True)
        self.worker.start()

    def _save_worker(self, renderer: Renderer, output: Path) -> None:
        try:
            frames = renderer.export(output)
            self.queue.put(("saved", output, frames))
        except Exception as error:  # noqa: BLE001 - every failure must reach the window
            self.queue.put(("save_failed", error))

    def _start_image(self, options: JobOptions) -> None:
        self.image_cancel.clear()
        self.image_running = True
        self.last_saved = None
        self._clear_log()
        self.progress.configure(mode="indeterminate", value=0)
        self.progress.start(12)
        self.status_var.set("Upscaling the image…")
        self.worker = threading.Thread(target=self._image_work, args=(options,), daemon=True)
        self.worker.start()
        self._refresh()

    def _image_work(self, options: JobOptions) -> None:
        try:
            result = run_job(options, cancel=self.image_cancel)
            self.queue.put(("image_done", result))
        except Exception as error:  # noqa: BLE001 - every failure must reach the window
            self.queue.put(("image_failed", error))

    def open_result(self) -> None:
        if self.last_saved is not None and self.last_saved.exists():
            os.startfile(self.last_saved)

    def show_result_in_folder(self) -> None:
        if self.last_saved is not None and self.last_saved.exists():
            subprocess.Popen(["explorer", "/select,", str(self.last_saved)])

    def toggle_log(self) -> None:
        self.log_visible = not self.log_visible
        with self._behind_curtain():
            if self.log_visible and not self.fullscreen:
                self.log_panel.grid()
            else:
                self.log_panel.grid_remove()
            self.log_button.configure(text="Hide log" if self.log_visible else "Show log")

    def _close_requested(self) -> None:
        """The window's close button: a whole-file upscale in progress would be lost, so ask first."""
        if self.batch and not self.saving and not messagebox.askyesno(
                APP_NAME, "The video is still being upscaled. Close the window and lose that work?",
                icon="warning", parent=self.root):
            return
        self.close()

    def close(self) -> None:
        """Close the window once a save in progress finishes; an image being upscaled is cancelled first."""
        if self.closing:
            return  # already waiting for it
        self.closing = True
        if self.image_running:
            self.image_cancel.set()
        if self.saving:
            self.status_var.set("Closing once the video is saved…")
        self._close_when_done()

    def _close_when_done(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.root.after(POLL_MS, self._close_when_done)  # the window keeps painting meanwhile
            return
        try:
            self._close_media()
            if self.drop is not None:
                self.drop.close()
        finally:
            if self.video_memory is not None:
                self.video_memory.close()
            log.set_sink(None)
            self.root.destroy()

    # ---- messages from the workers --------------------------------------------------------------

    def _probe_gpu(self) -> None:
        """Start FSR once on the card the upscaler will use: FSR 4, or else the newest upscaler the DLL offers there,
        so the window can say that FSR 4 cannot run and still offer the others."""
        fsr4_missing = None
        try:
            from .ffx import FFX_UPSCALE_ENABLE_AUTO_EXPOSURE, NoFsr4Error, Upscaler
            from .gpu import Gpu
            gpu = Gpu(verbose=False)
            try:
                sdk_dir = find_sdk_dir(None)
                try:
                    upscaler = Upscaler(gpu, (256, 144), (512, 288), FFX_UPSCALE_ENABLE_AUTO_EXPOSURE, sdk_dir)
                except NoFsr4Error as error:
                    if not error.versions:
                        raise
                    fsr4_missing = str(error)
                    log.emit(f"[ui] {fsr4_missing}")
                    upscaler = Upscaler(gpu, (256, 144), (512, 288), FFX_UPSCALE_ENABLE_AUTO_EXPOSURE, sdk_dir,
                                        provider=error.versions[0])
                name, version = gpu.device_name, upscaler.version_name
                upscaler.destroy()
            finally:
                gpu.destroy()
        except Exception as error:  # noqa: BLE001 - shown in the header and the banner
            self.queue.put(("gpu", False, str(error), None, None))
            return
        try:
            video_memory = system.VideoMemory(name)
        except Exception as error:  # noqa: BLE001 - the header shows the card without its memory; logged
            log.emit(f"[ui] error: the graphics card's memory use is unavailable: {error}")
            video_memory = None
        fsr = f"FSR {version}" if fsr4_missing is None else "no FSR 4"
        self.queue.put(("gpu", True, f"{name.replace('AMD ', '')}|{fsr}", video_memory, fsr4_missing))

    def _clean_temp(self) -> None:
        try:
            for directory in clean_stale_directories(temporary.folders_to_clean()):
                log.emit(f"[ui] removed pieces left by an earlier session: {directory}")
        except OSError as error:
            log.emit(f"[ui] could not clean old pieces: {error}")

    def _poll(self) -> None:
        messages = []
        try:
            while True:
                messages.append(self.queue.get_nowait())
        except queue.Empty:
            pass
        latest: dict[tuple, int] = {}
        for index, message in enumerate(messages):
            if message[0] in ("render", "player") and message[1] in ("status", "position", "live"):
                latest[message[:2]] = index
        try:
            for index, message in enumerate(messages):
                key = message[:2] if message[0] in ("render", "player") else message[:1]
                if key in latest and latest[key] != index:
                    continue
                try:
                    self._handle_message(message)
                except Exception as error:  # noqa: BLE001 - one bad message must not stop the others
                    log.emit(f"[ui] error handling {message[0]} {message[1] if len(message) > 1 else ''}: {error}")
                    self._show_banner(f"Something went wrong: {self._first_line(error)}")
            if self.view.comparing != self._comparing_shown:
                self._refresh()
        finally:
            self.root.after(POLL_MS, self._poll)

    def _handle_message(self, message: tuple) -> None:
        if message[0] == "render":
            self._on_render(message[1], *message[2])
        elif message[0] == "player":
            self._on_player(message[1], *message[2])
        else:
            getattr(self, f"_on_{message[0]}")(*message[1:])

    def _on_log(self, line: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line.rstrip("\n") + "\n", ("error",) if "error" in line.lower() else ())
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_gpu(self, ready: bool, text: str, video_memory: system.VideoMemory | None,
                fsr4_missing: str | None) -> None:
        self.gpu_ready = ready
        palette = self.theme.palette
        self.gpu_dot.configure(foreground=palette.error if not ready else palette.warning if fsr4_missing
                               else palette.success)
        self.gpu_text = text if ready else "GPU unavailable"
        self.video_memory = video_memory
        self._show_machine(again=False)
        if not ready:
            self._show_banner(f"FSR 4 cannot start on this machine: {text}")
        if fsr4_missing is not None:
            self.fsr4_missing = fsr4_missing
            for block in self.side_blocks.values():
                block.fsr4_missing = True
            self._show_banner(f"{fsr4_missing}. The FSR 4 presets cannot run here; FSR 3.1, FSR 2.3, the HQ scaler "
                              "and Real-ESRGAN can.")
            self._update_preset_lists()
            self._update_layers()
        self._refresh()

    def _show_machine(self, again: bool = True) -> None:
        """The processor with this program's RAM, the graphics card with its VRAM (every MACHINE_REFRESH_MS)."""
        used, total = system.main_memory()
        self.cpu_label.configure(text=f"{self.processor} · RAM {used / system.GIGABYTE:.1f} / "
                                      f"{total / system.GIGABYTE:.0f} GB")
        name, _, version = self.gpu_text.partition("|")
        parts = [name]
        if self.video_memory is not None:
            try:
                used, total = self.video_memory.read()
                parts.append(f"VRAM {used / system.GIGABYTE:.1f} / {total / system.GIGABYTE:.0f} GB")
            except OSError as error:
                log.emit(f"[ui] error: the graphics card's memory use is unavailable: {error}")
                self.video_memory = None
        self.gpu_label.configure(text=" · ".join(parts + ([version] if version else [])) or "Checking the GPU…")
        if again:
            self.root.after(MACHINE_REFRESH_MS, self._show_machine)

    def _on_drop(self, paths: list[str]) -> None:
        if paths and self.load(Path(paths[0])) and len(paths) > 1:
            self._show_banner(f"Opened {Path(paths[0]).name}; drop one file at a time.")

    def _on_loaded(self, token: int, path: Path, info: MediaInfo, image: np.ndarray | None) -> None:
        if token != self.load_token:
            return
        self.info = info
        self.source_details.configure(text=describe_source(path, info))
        self._refresh()
        if image is not None:
            settings = self._render_settings()
            self.view.set_media((info.width, info.height), settings.output_size if settings else (info.width, info.height))
            self.image_source = image
            sdk_dir = self._sdk_dir() if self.gpu_ready is not False else None
            if sdk_dir is not None:
                self._use_output_folder()
                self.image_previewer = ImagePreviewer(
                    path, sdk_dir, lambda kind, *values: self.queue.put(("preview", token, kind, values)))
            self._update_layers()
            self.status_var.set("Ready · press Upscale or Ctrl+Enter to save" if self._saved_side() is not None
                                else NOTHING_UPSCALED)
        else:
            self._start_video(path, info)
            self.status_var.set("Ready · Space plays it upscaled · Upscale saves the whole video"
                                if self._saved_side() is not None else NOTHING_UPSCALED)
        self._refresh()

    def _on_load_failed(self, token: int, path: Path, error: Exception) -> None:
        if token != self.load_token:
            return
        self.input_path, self.info = None, None
        self._update_preset_lists()
        self._clear_output()
        self.source_name.configure(text="No file yet")
        self.source_details.configure(text="Drop a file anywhere on this window, or choose one.")
        self.view.show_empty()
        self.status_var.set("Ready")
        self._show_banner(f"Cannot open {path.name}: {self._first_line(error)}")
        log.emit(f"[ui] error reading {path}: {error}")
        self._refresh()

    def _on_source_thumbnail(self, token: int, frame: int, thumbnail: np.ndarray) -> None:
        if token == self.load_token:
            self.source_thumbnails[frame] = thumbnail
            self.transport.set_thumbnails(source=dict(self.source_thumbnails))

    def _on_render(self, kind: str, *values) -> None:
        renderer = self.renderer
        if renderer is None:
            return
        if kind == "piece":
            self.player.refresh()
            if self.player.playing and self.player.speed < 0 and not self.transport.dragging:
                renderer.set_target(self._render_target(self.player.position))
            self.transport.set_thumbnails(upscaled=renderer.thumbnails())
            self._update_progress()
        elif kind == "status":
            position, speed = values
            self.render_position = position
            if position is not None:
                self.render_speed = speed
                self.player.render_speed = speed
            self.player.render_active = position is not None
            self._update_progress()
        elif kind == "total":
            self.transport.set_total(values[0])
            self._update_progress()
        elif kind == "complete":
            if self.batch and not self.saving:
                self._save()
        elif kind == "live":
            self.player.show_live(*values)
            self.renderer.want_live = self.batch and self.player.follow_live
        elif kind == "error":
            error = values[0]
            if self.batch:
                self.batch = False
                renderer.set_batch(False)
                self._update_layers()
                self.root.title(APP_NAME)
            if self.player.playing:
                self.player.pause()
            self._show_banner(f"Upscaling stopped: {self._first_line(error)}")
            if not self.log_visible:
                self.toggle_log()
            self._refresh()

    def _update_progress(self) -> None:
        renderer = self.renderer
        if renderer is None:
            return
        total = renderer.timeline.total
        ready = renderer.timeline.ready_frames()
        ranges = self._playable_ranges()
        self.transport.set_progress(ranges, self.render_position)
        if self.saving:
            return
        self.progress.configure(value=1000 * ready / max(1, total))
        speed = f" · {self.render_speed:.0f} fps" if self.render_position is not None else ""
        if self.batch:
            left = (total - ready) / self.render_speed if self.render_speed else 0
            remaining = f" · {format_duration(left)} left" if self.render_position is not None and left else ""
            self.status_var.set(f"Upscaling · {ready / max(1, total):.0%}{speed}{remaining}")
            self.root.title(f"{int(100 * ready / max(1, total))}% · {APP_NAME}")
        elif self._saved_side() is None:
            self.status_var.set(NOTHING_UPSCALED)
        elif self.render_position is not None:
            position = self.player.position
            end = next((end for start, end in ranges if start <= position < end), position)
            ahead = (end - position) / float(self.info.frame_rate)
            ready = f"{ahead:.1f} s" if ahead < 60 else format_time(ahead)
            self.status_var.set(f"Upscaling ahead of the playhead · {ready} ready{speed}")

    def _playable_ranges(self) -> list[tuple[int, int]]:
        """The frames that play without waiting: ready in every upscaled version shown (in the current version
        alone during a whole-file upscale, or when the view shows none)."""
        renderer = self.renderer
        shown = list(self._shown_layers.values())
        if self.batch or not shown:
            return renderer.timeline.ready_ranges()
        ranges = renderer.timeline_of(shown[0]).ready_ranges()
        for settings in shown[1:]:
            ranges = intersect_ranges(ranges, renderer.timeline_of(settings).ready_ranges())
        return ranges

    def _on_player(self, kind: str, *values) -> None:
        if self.player is None:
            return
        if kind == "position":
            self.transport.set_position(values[0])
            if self.renderer is not None and not self.transport.dragging:
                self.renderer.set_target(self._render_target(values[0]))
        elif kind == "state":
            playing, buffering = values
            self.transport.set_playing(playing, buffering)
            if not playing:
                self._flash_controls()
        elif kind == "notice":
            self._show_banner(values[0])
        elif kind == "unplayable":
            self._show_banner(f"Nothing to play yet: {self._why_not_upscaling()}.")
        elif kind == "end":
            # The video ends earlier than its estimated length: the renderer takes the real one (and says so).
            if self.renderer is not None:
                self.renderer.set_total(values[0])
            else:
                self.transport.set_total(values[0])
        elif kind == "error":
            self._show_banner(f"Playback stopped: {self._first_line(values[0])}")

    def _on_saved(self, output: Path, frames: int) -> None:
        self._end_save()
        self.last_saved = output
        width, height = self.renderer.settings.output_size
        seconds = frames / float(self.info.frame_rate)
        self.status_var.set(f"Saved {output.name} · {format_duration(seconds)} of video at {width} × {height}")
        self._refresh()

    def _on_save_failed(self, error: Exception) -> None:
        self._end_save()
        self.status_var.set("Saving failed")
        log.emit(f"[ui] error: {error}")
        self._show_banner(f"Saving failed: {self._first_line(error)}")
        if not self.log_visible:
            self.toggle_log()
        self._refresh()

    def _end_save(self) -> None:
        self.saving = False
        self.batch = False
        if self.renderer is not None:
            self.renderer.set_batch(False)
            self._apply_settings()  # the sides may have changed meanwhile
        self.progress.stop()
        self.progress.configure(mode="determinate", value=1000)
        self.root.title(APP_NAME)

    def _on_preview(self, token: int, kind: str, values: tuple) -> None:
        """News from the image previewer: a picture is ready, one failed, or it could not start."""
        if token != self.load_token or self.image_source is None:
            return
        if kind == "error":
            self._show_banner(f"This version could not be upscaled: {self._first_line(values[1])}")
        elif kind == "failed":
            self._show_banner(f"The image cannot be upscaled here: {self._first_line(values[0])}")
            self.image_previewer.close()  # its worker has stopped
            self.image_previewer = None
        self._show_image()

    def _on_image_done(self, result: JobResult) -> None:
        self._end_image()
        if result.cancelled:
            self.progress.configure(value=0)
            self.status_var.set("Cancelled · nothing was saved")
        else:
            self.last_saved = result.output
            width, height = self.view.output_size
            self.status_var.set(f"Saved {result.output.name} · {width} × {height} in {result.seconds:.1f} s")
        self._refresh()

    def _on_image_failed(self, error: Exception) -> None:
        self._end_image()
        self.status_var.set("Failed")
        log.emit(f"[ui] error: {error}")
        self._show_banner(f"Upscaling failed: {self._first_line(error)}")
        if not self.log_visible:
            self.toggle_log()
        self._refresh()

    def _end_image(self) -> None:
        self.image_running = False
        self.progress.stop()
        self.progress.configure(mode="determinate", value=1000)

    # ---- small helpers --------------------------------------------------------------------------

    def _callback_failed(self, kind, error, trace) -> None:
        """An exception in a Tk callback: log it with its traceback and show it (stderr is not visible)."""
        import traceback
        log.emit("[ui] error: " + "".join(traceback.format_exception(kind, error, trace)).rstrip())
        self._show_banner(f"Something went wrong: {self._first_line(error)}")

    @staticmethod
    def _first_line(error: Exception) -> str:
        text = str(error).strip() or type(error).__name__
        line = text.splitlines()[0]
        return line if len(line) <= 220 else line[:217] + "…"

    def _show_banner(self, text: str) -> None:
        self.banner_label.configure(text=text)
        if not self.fullscreen:
            self.banner.grid()

    def _hide_banner(self) -> None:
        self.banner_label.configure(text="")
        self.banner.grid_remove()

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")


def run_ui(preselected: Path | None = None) -> int:
    """Open the window; returns when it is closed."""
    if os.name == "nt":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError) as error:
            log.emit(f"[ui] DPI awareness unavailable: {error}")
        enable_dark_dialogs()
    usage.apply(usage.DEFAULT.threads, usage.DEFAULT.priority)
    root = tk.Tk()
    root.withdraw()
    App(root, preselected)
    style_title_bar(root)
    root.deiconify()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(run_ui(Path(sys.argv[1]) if len(sys.argv) > 1 else None))
