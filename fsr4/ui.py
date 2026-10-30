"""The window: pick or drop a file, set the upscale, run it, and compare the result with a plain upscale.

Layout: a header (title and GPU), a settings sidebar, the compare view, an optional log, and a footer with the
progress and the actions. The job runs on a worker thread; progress, live previews, log lines and the result
reach the window through a queue polled every 50 ms. Files can be dropped anywhere on the window.
"""
from __future__ import annotations

import ctypes
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from . import log
from .job import JobOptions, JobPlan, JobResult, default_output, plan_from_info, run_job
from .media import (COPY_ANY_AUDIO_SUFFIXES, IMAGE_SUFFIXES, MP4_AUDIO_COPY, VIDEO_SUFFIXES, MediaInfo, is_image,
                    probe, read_frame)
from .ui_compare import ZOOM_LEVELS, CompareView
from .ui_theme import Theme, enable_dark_dialogs, style_title_bar

APP_NAME = "FSR4 Upscaler"
POLL_MS = 50
SIDEBAR_WIDTH = 340
PREVIEW_POSITION = 0.1
PREVIEW_MAX_SECONDS = 10.0
SIZE_CHOICES = [("1.5×", "1.5"), ("2×", "2"), ("3×", "3"), ("1440p", "1440"), ("4K", "2160"), ("Custom", "custom")]
TARGET_SHORT_SIDES = {"1440": 1440, "2160": 2160}
MOTION_CHOICES = [("Optical flow · fast", "dis", "fast"), ("Optical flow · ultrafast", "dis", "ultrafast"),
                  ("Optical flow · medium, best", "dis", "medium"), ("Off", "zero", "fast")]
ENCODER_CHOICES = [("HEVC · GPU (AMF)", "amf"), ("H.264 · GPU (AMF)", "h264_amf"), ("H.264 · CPU (x264)", "x264"),
                   ("HEVC · CPU (x265)", "x265")]
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


def describe_source(path: Path, info: MediaInfo) -> str:
    """Two short lines about a probed file for the sidebar."""
    size = f"{info.width} × {info.height}"
    if is_image(path):
        return f"{codec_name(info.codec, VIDEO_CODEC_NAMES)} image · {size}"
    rate = f"{float(info.frame_rate):.3f}".rstrip("0").rstrip(".")
    first = f"{codec_name(info.codec, VIDEO_CODEC_NAMES)} · {size} · {rate} fps"
    parts = []
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


class App:
    """The main window. `load`, `start` and `cancel` are also what the automated UI check drives."""

    def __init__(self, root: tk.Tk, preselected: Path | None = None):
        self.root = root
        self.theme = Theme(root)
        self.queue: queue.Queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.load_token = 0
        self.input_path: Path | None = None
        self.info: MediaInfo | None = None
        self.output_custom = False
        self._setting_output = False
        self.last_result: JobResult | None = None
        self.running = False
        self.gpu_ready: bool | None = None
        self.log_visible = False
        root.title(APP_NAME)
        root.iconphoto(True, *self.theme.icon_photos())
        self._build()
        self._place_window()
        log.set_sink(lambda line: self.queue.put(("log", line)))
        self.drop = self._enable_drop()
        self._bind_keys()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(POLL_MS, self._poll)
        threading.Thread(target=self._probe_gpu, daemon=True).start()
        self._refresh()
        if preselected is not None:
            self.load(preselected)

    # ---- layout ---------------------------------------------------------------------------------

    def _build(self) -> None:
        px = self.theme.px
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        self._build_header().grid(row=0, column=0, sticky="ew")
        body = ttk.Frame(root)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        self._build_sidebar(body).grid(row=0, column=0, sticky="ns")
        self._build_main(body).grid(row=0, column=1, sticky="nsew")
        self._build_footer().grid(row=2, column=0, sticky="ew")

    def _build_header(self) -> ttk.Frame:
        px = self.theme.px
        header = ttk.Frame(self.root, padding=(px(20), px(12), px(20), px(8)))
        header.columnconfigure(2, weight=1)
        ttk.Label(header, image=self.theme.icon(24)).grid(row=0, column=0, padx=(0, px(10)))
        ttk.Label(header, text=APP_NAME, style="Title.TLabel").grid(row=0, column=1, sticky="w")
        status = ttk.Frame(header)
        status.grid(row=0, column=3, sticky="e")
        self.gpu_dot = ttk.Label(status, text="●", style="Faint.TLabel")
        self.gpu_dot.grid(row=0, column=0, padx=(0, px(6)))
        self.gpu_label = ttk.Label(status, text="Checking the GPU…", style="Small.TLabel")
        self.gpu_label.grid(row=0, column=1)
        return header

    def _section(self, parent: ttk.Frame, row: int, title: str) -> int:
        px = self.theme.px
        ttk.Label(parent, text=title.upper(), style="Section.TLabel").grid(
            row=row, column=0, sticky="w", pady=(px(13) if row else px(2), px(5)))
        return row + 1

    def _build_sidebar(self, parent: ttk.Frame) -> ttk.Frame:
        """Settings top to bottom in the order they are decided: source, size, motion, sharpening, output."""
        px = self.theme.px
        sidebar = ttk.Frame(parent, width=px(SIDEBAR_WIDTH), padding=(px(20), px(2), px(18), px(8)))
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        wrap = px(SIDEBAR_WIDTH - 38)

        row = self._section(sidebar, 0, "Source")
        self.source_name = ttk.Label(sidebar, text="No file yet", style="Strong.TLabel", wraplength=wrap)
        self.source_name.grid(row=row, column=0, sticky="w")
        self.source_details = ttk.Label(sidebar, text="Drop a file anywhere on this window, or choose one.",
                                        style="Small.TLabel", wraplength=wrap, justify="left")
        self.source_details.grid(row=row + 1, column=0, sticky="w", pady=(px(2), px(7)))
        self.choose_button = ttk.Button(sidebar, text="Choose file…", style="Secondary.TButton", command=self.browse)
        self.choose_button.grid(row=row + 2, column=0, sticky="w")

        row = self._section(sidebar, row + 3, "Size")
        chips = ttk.Frame(sidebar)
        chips.grid(row=row, column=0, sticky="w")
        self.size_var = tk.StringVar(value="2")
        self.size_chips: dict[str, ttk.Radiobutton] = {}
        for index, (text, value) in enumerate(SIZE_CHOICES):
            chip = ttk.Radiobutton(chips, text=text, value=value, variable=self.size_var,
                                   style="Segment.TRadiobutton", command=self._refresh, takefocus=False)
            chip.grid(row=0, column=index, padx=(0, px(5)))
            self.size_chips[value] = chip
        self.custom_row = ttk.Frame(sidebar)
        self.custom_row.grid(row=row + 1, column=0, sticky="ew", pady=(px(8), 0))
        self.custom_row.columnconfigure(0, weight=1)
        self.custom_scale = tk.DoubleVar(value=2.0)
        ttk.Scale(self.custom_row, from_=1.0, to=3.0, variable=self.custom_scale,
                  command=lambda _: self._refresh()).grid(row=0, column=0, sticky="ew")
        self.custom_label = ttk.Label(self.custom_row, text="2.00×", style="Small.TLabel", width=6, anchor="e")
        self.custom_label.grid(row=0, column=1, padx=(px(8), 0))
        self.custom_row.grid_remove()
        self.size_label = ttk.Label(sidebar, text="", style="Strong.TLabel", wraplength=wrap, justify="left")
        self.size_label.grid(row=row + 2, column=0, sticky="w", pady=(px(8), 0))
        self.size_note = ttk.Label(sidebar, text="", style="Faint.TLabel", wraplength=wrap, justify="left")
        self.size_note.grid(row=row + 3, column=0, sticky="w")

        row = self._section(sidebar, row + 4, "Motion vectors")
        self.motion_choice = ttk.Combobox(sidebar, state="readonly", values=[c[0] for c in MOTION_CHOICES])
        self.motion_choice.current(0)
        self.motion_choice.grid(row=row, column=0, sticky="ew")
        self.motion_choice.bind("<<ComboboxSelected>>", lambda _: self._refresh())
        self.motion_note = ttk.Label(sidebar, text="", style="Faint.TLabel", wraplength=wrap, justify="left")
        self.motion_note.grid(row=row + 1, column=0, sticky="w", pady=(px(4), 0))

        row = self._section(sidebar, row + 2, "Sharpening")
        sharpen_row = ttk.Frame(sidebar)
        sharpen_row.grid(row=row, column=0, sticky="ew")
        sharpen_row.columnconfigure(1, weight=1)
        self.sharpen_var = tk.BooleanVar(value=False)
        self.sharpen_switch = ttk.Checkbutton(sharpen_row, text="", variable=self.sharpen_var,
                                              style="Switch.TCheckbutton", command=self._refresh, takefocus=False)
        self.sharpen_switch.grid(row=0, column=0, padx=(0, px(4)))
        self.sharpness_var = tk.DoubleVar(value=0.4)
        self.sharpness_scale = ttk.Scale(sharpen_row, from_=0.0, to=1.0, variable=self.sharpness_var,
                                         command=lambda _: self._refresh())
        self.sharpness_scale.grid(row=0, column=1, sticky="ew")
        self.sharpness_label = ttk.Label(sharpen_row, text="off", style="Small.TLabel", width=5, anchor="e")
        self.sharpness_label.grid(row=0, column=2, padx=(px(8), 0))

        row = self._section(sidebar, row + 1, "Output")
        output_row = ttk.Frame(sidebar)
        output_row.grid(row=row, column=0, sticky="ew")
        output_row.columnconfigure(0, weight=1)
        self.output_var = tk.StringVar()
        self.output_var.trace_add("write", self._output_edited)
        self.output_entry = ttk.Entry(output_row, textvariable=self.output_var)
        self.output_entry.grid(row=0, column=0, sticky="ew")
        self.save_button = ttk.Button(output_row, text="…", width=3, style="Secondary.TButton",
                                      command=self.browse_output)
        self.save_button.grid(row=0, column=1, padx=(px(6), 0))
        self.encoder_choice = ttk.Combobox(sidebar, state="readonly", values=[c[0] for c in ENCODER_CHOICES])
        self.encoder_choice.current(0)
        self.encoder_choice.grid(row=row + 1, column=0, sticky="ew", pady=(px(6), 0))
        self.encoder_choice.bind("<<ComboboxSelected>>", lambda _: self._refresh())
        self.audio_note = ttk.Label(sidebar, text="", style="Faint.TLabel", wraplength=wrap, justify="left")
        self.audio_note.grid(row=row + 2, column=0, sticky="w", pady=(px(4), 0))
        self.setting_widgets = [self.choose_button, self.output_entry, self.save_button, self.motion_choice,
                                self.sharpen_switch, self.sharpness_scale, self.encoder_choice]
        return sidebar

    def _build_main(self, parent: ttk.Frame) -> ttk.Frame:
        px = self.theme.px
        main = ttk.Frame(parent, padding=(px(4), px(4), px(20), px(8)))
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

        toolbar = ttk.Frame(main)
        toolbar.grid(row=1, column=0, sticky="ew", pady=(0, px(8)))
        toolbar.columnconfigure(1, weight=1)
        self.view_title = ttk.Label(toolbar, text="Preview", style="Strong.TLabel")
        self.view_title.grid(row=0, column=0, sticky="w")
        self.view_hint = ttk.Label(toolbar, text="", style="Faint.TLabel")
        self.view_hint.grid(row=0, column=1, sticky="w", padx=(px(12), 0))
        zoom_chips = ttk.Frame(toolbar)
        zoom_chips.grid(row=0, column=2, sticky="e")
        self.zoom_var = tk.IntVar(value=1)
        self.zoom_chips = []
        for index, level in enumerate(ZOOM_LEVELS):
            chip = ttk.Radiobutton(zoom_chips, text=f"{level * 100}%", value=level, variable=self.zoom_var,
                                   style="Segment.TRadiobutton", takefocus=False,
                                   command=lambda: self.view.set_zoom(self.zoom_var.get()))
            chip.grid(row=0, column=index, padx=(px(5) if index else 0, 0))
            self.zoom_chips.append(chip)

        self.view = CompareView(main, self.theme, on_browse=self.browse, on_zoom_changed=self.zoom_var.set)
        self.view.grid(row=2, column=0, sticky="nsew")
        self.log_panel = self._build_log(main)
        self.log_panel.grid(row=3, column=0, sticky="ew", pady=(px(10), 0))
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
        footer = ttk.Frame(self.root, padding=(px(20), px(4), px(20), px(16)))
        footer.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(footer, style="Thin.Horizontal.TProgressbar", maximum=1000)
        self.progress.grid(row=0, column=0, columnspan=6, sticky="ew", pady=(0, px(12)))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(footer, textvariable=self.status_var, style="Status.TLabel").grid(row=1, column=0, sticky="w")
        self.log_button = ttk.Button(footer, text="Show log", style="Ghost.TButton", command=self.toggle_log,
                                     takefocus=False)
        self.log_button.grid(row=1, column=1, padx=(px(8), 0))
        self.open_button = ttk.Button(footer, text="Open", style="Secondary.TButton", command=self.open_result)
        self.open_button.grid(row=1, column=2, padx=(px(8), 0))
        self.folder_button = ttk.Button(footer, text="Show in folder", style="Secondary.TButton",
                                        command=self.show_result_in_folder)
        self.folder_button.grid(row=1, column=3, padx=(px(8), 0))
        self.cancel_button = ttk.Button(footer, text="Cancel", style="Secondary.TButton", command=self.cancel)
        self.cancel_button.grid(row=1, column=4, padx=(px(8), 0))
        self.start_button = ttk.Button(footer, text="Upscale", style="Accent.TButton", command=self.start)
        self.start_button.grid(row=1, column=5, padx=(px(8), 0))
        return footer

    def _place_window(self) -> None:
        px = self.theme.px
        screen_width, screen_height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        width = min(px(1240), int(screen_width * 0.92))
        height = min(px(820), int(screen_height * 0.88))
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
        self.root.bind("<Control-o>", lambda _: self.browse())
        self.root.bind("<Control-Return>", lambda _: self.start())
        self.root.bind("<Escape>", lambda _: self.cancel())
        self.root.bind("<Control-l>", lambda _: self.toggle_log())

    # ---- settings -------------------------------------------------------------------------------

    def _scale_for(self, choice: str) -> float | None:
        """The per-dimension factor a size chip stands for with the current file (None without a file)."""
        if choice == "custom":
            return round(float(self.custom_scale.get()), 2)
        if choice in TARGET_SHORT_SIDES:
            if self.info is None:
                return None
            return TARGET_SHORT_SIDES[choice] / min(self.info.width, self.info.height)
        return float(choice)

    def _options(self) -> JobOptions:
        _, flow, preset = MOTION_CHOICES[self.motion_choice.current()]
        output = self.output_var.get().strip().strip('"')
        return JobOptions(
            input=self.input_path, output=Path(output) if output else None, scale=self._scale_for(self.size_var.get()),
            flow=flow, flow_preset=preset,
            sharpness=round(float(self.sharpness_var.get()), 2) if self.sharpen_var.get() else None,
            codec=ENCODER_CHOICES[self.encoder_choice.current()][1])

    def _plan(self) -> JobPlan | None:
        if self.info is None or self.input_path is None:
            return None
        return plan_from_info(self._options(), self.info)

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
        """Bring every derived label and widget state in line with the file, the settings and the job."""
        idle = not self.running
        image = self.input_path is not None and is_image(self.input_path)
        for widget in self.setting_widgets:
            self._set_state(widget, idle)
        if self.info is not None:
            for value, chip in self.size_chips.items():
                scale = self._scale_for(value)
                self._set_state(chip, idle and (value == "custom" or 1.0 - 1e-6 <= scale <= 3.0 + 1e-6))
            if self.size_chips[self.size_var.get()].instate(["disabled"]) and idle:
                self.size_var.set("2")
        else:
            for value, chip in self.size_chips.items():
                self._set_state(chip, idle and value not in TARGET_SHORT_SIDES)
        custom = self.size_var.get() == "custom"
        if custom:
            self.custom_row.grid()
        else:
            self.custom_row.grid_remove()
        self.custom_label.configure(text=f"{self.custom_scale.get():.2f}×")
        self._set_state(self.motion_choice, idle and not image)
        self._set_state(self.encoder_choice, idle and not image)
        self._set_state(self.sharpness_scale, idle and self.sharpen_var.get())
        self.sharpness_label.configure(text=f"{self.sharpness_var.get():.2f}" if self.sharpen_var.get() else "off")
        self.motion_note.configure(text="Not used for images." if image
                                   else "Lets FSR reuse detail from earlier frames.")
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
            self.size_note.configure(text=f"{plan.scale:.2f}× per side · {plan.scale ** 2:.1f}× the pixels")
            self.audio_note.configure(text="" if image else describe_audio_plan(self.info, plan.output))
            if plan.output.resolve() == self.input_path.resolve():
                problem = "The output would overwrite the input; choose another name."
        else:
            self.size_label.configure(style="Faint.TLabel",
                                      text="Reading the file…" if self.input_path is not None
                                      else "The output size appears once a file is loaded.")
            self.size_note.configure(text="")
            self.audio_note.configure(text="")
        if problem:
            self.size_note.configure(text=problem)
        ready = idle and plan is not None and problem is None and self.gpu_ready is not False
        self._set_state(self.start_button, ready)
        self._set_state(self.cancel_button, self.running and not self.cancel_event.is_set())
        if self.running:
            self.cancel_button.grid()
        else:
            self.cancel_button.grid_remove()
        done = idle and self.last_result is not None and self.last_result.output.exists()
        for button in (self.open_button, self.folder_button):
            if done:
                button.grid()
            else:
                button.grid_remove()
        comparing = self.view.comparing
        for chip in self.zoom_chips:
            self._set_state(chip, comparing)
        self.view_title.configure(text="Compare" if comparing else "Preview")
        self.view_hint.configure(text="Drag to compare · scroll to zoom · right-drag to pan" if comparing else "")

    def _default_output(self) -> Path:
        current = self.output_var.get().strip()
        path = default_output(self.input_path)
        if current and not is_image(self.input_path) and Path(current).suffix.lower() in (".mkv", ".mov"):
            path = path.with_suffix(Path(current).suffix)
        return path

    # ---- actions --------------------------------------------------------------------------------

    def browse(self) -> None:
        if self.running:
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
            self._set_output(Path(chosen))
            self._refresh()

    def load(self, path: Path) -> None:
        """Probe `path` and show its preview frame (on a worker thread)."""
        if self.running:
            self._show_banner("A job is running. Cancel it before opening another file.")
            return
        path = Path(path)
        if not path.is_file():
            self._show_banner(f"Not found: {path}")
            return
        self.load_token += 1
        self.input_path, self.info, self.last_result = path, None, None
        self.output_custom = False
        self._hide_banner()
        self.source_name.configure(text=path.name)
        self.source_details.configure(text="Reading…")
        self.view.show_message(f"Reading {path.name}…")
        self.progress.configure(value=0)
        self.status_var.set("Reading the file…")
        self._refresh()
        threading.Thread(target=self._load_worker, args=(self.load_token, path), daemon=True).start()

    def _load_worker(self, token: int, path: Path) -> None:
        try:
            info = probe(path)
            seconds = None
            if not is_image(path) and info.duration:
                seconds = min(info.duration * PREVIEW_POSITION, PREVIEW_MAX_SECONDS)
            frame = read_frame(path, info, seconds)
            self.queue.put(("loaded", token, path, info, frame, seconds))
        except Exception as error:  # noqa: BLE001 - any failure is shown in the window
            self.queue.put(("load_failed", token, path, error))

    def start(self) -> None:
        if self.running or self.info is None:
            return
        try:
            options = self._options()
            plan = plan_from_info(options, self.info)
        except ValueError as error:
            self._show_banner(str(error))
            return
        self.cancel_event.clear()
        self.running = True
        self.last_result = None
        self._hide_banner()
        self._clear_log()
        self.progress.configure(mode="indeterminate", value=0)
        self.progress.start(12)
        self.status_var.set(f"Starting · {plan.output_size[0]} × {plan.output_size[1]}")
        self.worker = threading.Thread(target=self._work, args=(options,), daemon=True)
        self.worker.start()
        self._refresh()

    def _work(self, options: JobOptions) -> None:
        try:
            result = run_job(options, cancel=self.cancel_event,
                             progress=lambda done, total, fps: self.queue.put(("progress", done, total, fps)),
                             preview=lambda source, upscaled, done: self.queue.put(("preview", source, upscaled)))
            self.queue.put(("done", result))
        except Exception as error:  # noqa: BLE001 - every failure must reach the window
            self.queue.put(("failed", error))

    def cancel(self) -> None:
        if self.running and not self.cancel_event.is_set():
            self.cancel_event.set()
            self.status_var.set("Cancelling after the current frame…")
            self._refresh()

    def open_result(self) -> None:
        if self.last_result is not None and self.last_result.output.exists():
            os.startfile(self.last_result.output)

    def show_result_in_folder(self) -> None:
        if self.last_result is not None and self.last_result.output.exists():
            subprocess.Popen(["explorer", "/select,", str(self.last_result.output)])

    def toggle_log(self) -> None:
        self.log_visible = not self.log_visible
        if self.log_visible:
            self.log_panel.grid()
        else:
            self.log_panel.grid_remove()
        self.log_button.configure(text="Hide log" if self.log_visible else "Show log")

    def close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.cancel_event.set()
            self.worker.join(timeout=15)
        if self.drop is not None:
            self.drop.close()
        log.set_sink(None)
        self.root.destroy()

    # ---- messages from the workers --------------------------------------------------------------

    def _probe_gpu(self) -> None:
        try:
            from .ffx import FFX_UPSCALE_ENABLE_AUTO_EXPOSURE, Upscaler, find_sdk_dir
            from .gpu import Gpu
            gpu = Gpu(verbose=False)
            try:
                upscaler = Upscaler(gpu, (256, 144), (512, 288), FFX_UPSCALE_ENABLE_AUTO_EXPOSURE, find_sdk_dir(None))
                text = f"{gpu.device_name.replace('AMD ', '')} · FSR {upscaler.version_name}"
                upscaler.destroy()
            finally:
                gpu.destroy()
            self.queue.put(("gpu", True, text))
        except Exception as error:  # noqa: BLE001 - shown in the header and the banner
            self.queue.put(("gpu", False, str(error)))

    def _poll(self) -> None:
        messages = []
        try:
            while True:
                messages.append(self.queue.get_nowait())
        except queue.Empty:
            pass
        last_preview = max((index for index, message in enumerate(messages) if message[0] == "preview"), default=-1)
        last_progress = max((index for index, message in enumerate(messages) if message[0] == "progress"), default=-1)
        for index, message in enumerate(messages):
            kind = message[0]
            if (kind == "preview" and index != last_preview) or (kind == "progress" and index != last_progress):
                continue
            getattr(self, f"_on_{kind}")(*message[1:])
        self.root.after(POLL_MS, self._poll)

    def _on_log(self, line: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line.rstrip("\n") + "\n", ("error",) if "error" in line.lower() else ())
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_gpu(self, ready: bool, text: str) -> None:
        self.gpu_ready = ready
        palette = self.theme.palette
        self.gpu_dot.configure(foreground=palette.success if ready else palette.error)
        self.gpu_label.configure(text=text if ready else "GPU unavailable")
        if not ready:
            self._show_banner(f"FSR 4 cannot start on this machine: {text}")
        self._refresh()

    def _on_drop(self, paths: list[str]) -> None:
        if paths:
            self.load(Path(paths[0]))
        if len(paths) > 1:
            self._show_banner(f"Opened {Path(paths[0]).name}; drop one file at a time.")

    def _on_loaded(self, token: int, path: Path, info: MediaInfo, frame, seconds: float | None) -> None:
        if token != self.load_token:
            return
        self.info = info
        self.source_details.configure(text=describe_source(path, info))
        where = f" · frame at {format_duration(seconds)}" if seconds else ""
        self.view.show_source(frame, f"Source · {info.width} × {info.height}{where}")
        self.status_var.set("Ready · press Upscale or Ctrl+Enter")
        self._refresh()

    def _on_load_failed(self, token: int, path: Path, error: Exception) -> None:
        if token != self.load_token:
            return
        self.input_path, self.info = None, None
        self._clear_output()
        self.source_name.configure(text="No file yet")
        self.source_details.configure(text="Drop a file anywhere on this window, or choose one.")
        self.view.show_empty()
        self.status_var.set("Ready")
        self._show_banner(f"Cannot open {path.name}: {self._first_line(error)}")
        log.emit(f"[ui] error reading {path}: {error}")
        self._refresh()

    def _on_progress(self, done: int, total: int | None, fps: float) -> None:
        if str(self.progress.cget("mode")) != "determinate":
            self.progress.stop()
            self.progress.configure(mode="determinate")
        if total:
            fraction = min(1.0, done / total)
            self.progress.configure(value=1000 * fraction)
            left = (total - done) / fps if fps else 0
            self.status_var.set(f"Upscaling · frame {done:,} / {total:,} · {fps:.1f} fps · {format_duration(left)} left")
            self.root.title(f"{int(fraction * 100)}% · {APP_NAME}")
        else:
            self.status_var.set(f"Upscaling · frame {done:,} · {fps:.1f} fps")

    def _on_preview(self, source, upscaled) -> None:
        self.view.show_result(source, upscaled)
        self._refresh()

    def _on_done(self, result: JobResult) -> None:
        self._finish()
        self.last_result = result
        self.progress.configure(value=1000)
        rate = result.frames / max(result.seconds, 1e-6)
        if result.cancelled:
            self.status_var.set(f"Cancelled after {result.frames:,} frames · partial file saved as {result.output.name}")
        else:
            frames = "1 image" if result.frames == 1 else f"{result.frames:,} frames"
            self.status_var.set(f"Saved {result.output.name} · {frames} in {format_duration(result.seconds)}"
                                f" ({rate:.1f} fps)")
        self._refresh()

    def _on_failed(self, error: Exception) -> None:
        self._finish()
        self.progress.configure(value=0)
        self.status_var.set("Failed")
        log.emit(f"[ui] error: {error}")
        self._show_banner(f"Upscaling failed: {self._first_line(error)}")
        if not self.log_visible:
            self.toggle_log()
        self._refresh()

    def _finish(self) -> None:
        self.running = False
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.root.title(APP_NAME)

    # ---- small helpers --------------------------------------------------------------------------

    @staticmethod
    def _first_line(error: Exception) -> str:
        text = str(error).strip() or type(error).__name__
        line = text.splitlines()[0]
        return line if len(line) <= 220 else line[:217] + "…"

    def _show_banner(self, text: str) -> None:
        self.banner_label.configure(text=text)
        self.banner.grid()

    def _hide_banner(self) -> None:
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
    root = tk.Tk()
    root.withdraw()
    App(root, preselected)
    style_title_bar(root)
    root.deiconify()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(run_ui(Path(sys.argv[1]) if len(sys.argv) > 1 else None))
