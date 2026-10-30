"""The window: pick a file, choose the scale and the motion vectors, watch the progress, compare before and after.

Plain tkinter (ships with Python, nothing to install). The job runs on a worker thread; log lines, progress
and the result reach the window through a queue polled every 100 ms. Previews are PNGs made by ffmpeg.
"""
from __future__ import annotations

import ctypes
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import __version__, log
from .job import JobOptions, JobPlan, default_output, plan, run_job
from .media import tool_path

PREVIEW_WIDTH = 440
PREVIEW_HEIGHT = 248
SCALE_PRESETS = [("1.5x  (FSR Quality)", 1.5), ("1.7x  (FSR Balanced)", 1.7), ("2x  (FSR Performance)", 2.0),
                 ("3x  (FSR Ultra performance)", 3.0), ("Custom", None)]
FLOW_CHOICES = [("Optical flow, fast", "dis", "fast"), ("Optical flow, ultrafast", "dis", "ultrafast"),
                ("Optical flow, medium (best, slower)", "dis", "medium"), ("None", "zero", "fast")]
ENCODER_CHOICES = [("HEVC on the GPU (AMF)", "amf"), ("H.264 on the GPU (AMF)", "h264_amf"),
                   ("H.264 software (x264)", "x264"), ("HEVC software (x265)", "x265")]
FILE_TYPES = [("Images and videos", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.mp4 *.mkv *.mov *.avi *.webm *.m4v"),
              ("All files", "*.*")]
CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def make_preview(source: Path, destination: Path, video_filter: str, seek_seconds: float | None = None) -> None:
    """Render one frame of `source` through `video_filter` into a PNG."""
    command = [tool_path("ffmpeg"), "-v", "error", "-y"]
    if seek_seconds is not None:
        command += ["-ss", f"{seek_seconds:.3f}"]
    command += ["-i", str(source), "-frames:v", "1", "-vf", video_filter, str(destination)]
    completed = subprocess.run(command, capture_output=True, creationflags=CREATE_NO_WINDOW)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg preview failed: {completed.stderr.decode('utf-8', 'replace').strip()}")


class App:
    """The main window."""

    def __init__(self, root: tk.Tk, preselected: Path | None):
        self.root = root
        root.title(f"FSR4 upscaler {__version__}")
        root.minsize(960, 700)
        self.queue: queue.Queue = queue.Queue()
        self.cancel = threading.Event()
        self.worker: threading.Thread | None = None
        self.temp = Path(tempfile.mkdtemp(prefix="fsr4-ui-"))
        self.job_plan: JobPlan | None = None
        self.result_path: Path | None = None
        self.photos: dict[str, tk.PhotoImage] = {}
        self._build()
        log.set_sink(lambda line: self.queue.put(("log", line)))
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.after(100, self._poll)
        threading.Thread(target=self._probe_gpu, daemon=True).start()
        if preselected is not None:
            self.input_var.set(str(preselected))
            self._input_changed()

    # ---- layout ---------------------------------------------------------------------------------

    def _build(self) -> None:
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)
        padding = {"padx": 8, "pady": 4}

        files = ttk.Frame(root)
        files.grid(row=0, column=0, sticky="ew", padx=8, pady=(10, 0))
        files.columnconfigure(1, weight=1)
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        ttk.Label(files, text="Input").grid(row=0, column=0, sticky="w", **padding)
        input_entry = ttk.Entry(files, textvariable=self.input_var)
        input_entry.grid(row=0, column=1, sticky="ew", **padding)
        input_entry.bind("<FocusOut>", lambda _: self._input_changed())
        input_entry.bind("<Return>", lambda _: self._input_changed())
        ttk.Button(files, text="Browse…", command=self._browse_input).grid(row=0, column=2, **padding)
        ttk.Label(files, text="Output").grid(row=1, column=0, sticky="w", **padding)
        ttk.Entry(files, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", **padding)
        ttk.Button(files, text="Browse…", command=self._browse_output).grid(row=1, column=2, **padding)

        settings = ttk.LabelFrame(root, text="Settings")
        settings.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        for column in (1, 3):
            settings.columnconfigure(column, weight=1)
        ttk.Label(settings, text="Scale").grid(row=0, column=0, sticky="w", **padding)
        self.scale_choice = ttk.Combobox(settings, state="readonly", values=[label for label, _ in SCALE_PRESETS], width=28)
        self.scale_choice.current(2)
        self.scale_choice.grid(row=0, column=1, sticky="w", **padding)
        self.scale_choice.bind("<<ComboboxSelected>>", lambda _: self._scale_changed())
        self.custom_scale = tk.DoubleVar(value=2.0)
        self.custom_spin = ttk.Spinbox(settings, from_=1.0, to=3.0, increment=0.1, textvariable=self.custom_scale,
                                       width=6, command=self._scale_changed, state="disabled")
        self.custom_spin.grid(row=0, column=2, sticky="w", **padding)
        self.custom_spin.bind("<KeyRelease>", lambda _: self._scale_changed())
        self.size_label = ttk.Label(settings, text="")
        self.size_label.grid(row=0, column=3, sticky="w", **padding)

        ttk.Label(settings, text="Motion vectors").grid(row=1, column=0, sticky="w", **padding)
        self.flow_choice = ttk.Combobox(settings, state="readonly", values=[label for label, _, _ in FLOW_CHOICES], width=28)
        self.flow_choice.current(0)
        self.flow_choice.grid(row=1, column=1, sticky="w", **padding)
        ttk.Label(settings, text="Video encoder").grid(row=1, column=2, sticky="w", **padding)
        self.encoder_choice = ttk.Combobox(settings, state="readonly", values=[label for label, _ in ENCODER_CHOICES], width=24)
        self.encoder_choice.current(0)
        self.encoder_choice.grid(row=1, column=3, sticky="w", **padding)

        self.sharpen_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings, text="Sharpen", variable=self.sharpen_var,
                        command=self._sharpen_changed).grid(row=2, column=0, sticky="w", **padding)
        self.sharpness_var = tk.DoubleVar(value=0.5)
        self.sharpness_scale = ttk.Scale(settings, from_=0.0, to=1.0, variable=self.sharpness_var, length=180,
                                         command=lambda _: self._sharpen_changed(), state="disabled")
        self.sharpness_scale.grid(row=2, column=1, sticky="w", **padding)
        self.sharpness_label = ttk.Label(settings, text="off")
        self.sharpness_label.grid(row=2, column=2, sticky="w", **padding)

        previews = ttk.LabelFrame(root, text="Preview")
        previews.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        previews.columnconfigure(0, weight=1)
        previews.columnconfigure(1, weight=1)
        self.left_title = ttk.Label(previews, text="Input")
        self.left_title.grid(row=0, column=0, sticky="w", padx=8)
        self.right_title = ttk.Label(previews, text="Result (100% crop)")
        self.right_title.grid(row=0, column=1, sticky="w", padx=8)
        self.left_canvas = tk.Canvas(previews, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT, bg="#202020",
                                     highlightthickness=0)
        self.left_canvas.grid(row=1, column=0, padx=8, pady=(0, 8))
        self.right_canvas = tk.Canvas(previews, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT, bg="#202020",
                                      highlightthickness=0)
        self.right_canvas.grid(row=1, column=1, padx=8, pady=(0, 8))
        self._canvas_text(self.left_canvas, "Choose an image or a video")
        self._canvas_text(self.right_canvas, "Run to see the result")

        log_frame = ttk.LabelFrame(root, text="Log")
        log_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=6)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=7, wrap="word", state="disabled", font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        bottom = ttk.Frame(root)
        bottom.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 10))
        bottom.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(bottom, mode="determinate", maximum=1000)
        self.progress.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 4))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(bottom, textvariable=self.status_var).grid(row=1, column=0, sticky="w")
        self.open_button = ttk.Button(bottom, text="Open folder", command=self._open_folder, state="disabled")
        self.open_button.grid(row=1, column=1, padx=4)
        self.cancel_button = ttk.Button(bottom, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_button.grid(row=1, column=2, padx=4)
        self.run_button = ttk.Button(bottom, text="Upscale", command=self._run)
        self.run_button.grid(row=1, column=3, padx=(4, 0))
        self.gpu_var = tk.StringVar(value="Checking the GPU…")
        ttk.Label(root, textvariable=self.gpu_var, foreground="#666666").grid(row=5, column=0, sticky="w", padx=12, pady=(0, 6))

    @staticmethod
    def _canvas_text(canvas: tk.Canvas, text: str) -> None:
        canvas.delete("all")
        canvas.create_text(PREVIEW_WIDTH // 2, PREVIEW_HEIGHT // 2, text=text, fill="#aaaaaa", font=("Segoe UI", 10))

    def _show_image(self, canvas: tk.Canvas, key: str, path: Path) -> None:
        photo = tk.PhotoImage(file=str(path))
        self.photos[key] = photo
        canvas.delete("all")
        canvas.create_image(PREVIEW_WIDTH // 2, PREVIEW_HEIGHT // 2, image=photo, anchor="center")

    # ---- settings -------------------------------------------------------------------------------

    def _scale_value(self) -> float:
        label, preset = SCALE_PRESETS[self.scale_choice.current()]
        if preset is not None:
            return preset
        try:
            return min(3.0, max(1.0, float(self.custom_scale.get())))
        except (tk.TclError, ValueError):
            return 2.0

    def _scale_changed(self) -> None:
        custom = SCALE_PRESETS[self.scale_choice.current()][1] is None
        self.custom_spin.configure(state="normal" if custom else "disabled")
        self._update_size_label()

    def _sharpen_changed(self) -> None:
        enabled = self.sharpen_var.get()
        self.sharpness_scale.configure(state="normal" if enabled else "disabled")
        self.sharpness_label.configure(text=f"{self.sharpness_var.get():.2f}" if enabled else "off")

    def _update_size_label(self) -> None:
        if self.job_plan is None:
            self.size_label.configure(text="")
            return
        try:
            current = plan(self._options())
        except ValueError as error:
            self.size_label.configure(text=str(error))
            return
        width, height = current.render_size
        out_width, out_height = current.output_size
        self.size_label.configure(text=f"{width}x{height}  →  {out_width}x{out_height}")

    def _options(self) -> JobOptions:
        _, flow, preset = FLOW_CHOICES[self.flow_choice.current()]
        output = self.output_var.get().strip()
        return JobOptions(
            input=Path(self.input_var.get().strip()), output=Path(output) if output else None,
            scale=self._scale_value(), flow=flow, flow_preset=preset,
            sharpness=float(self.sharpness_var.get()) if self.sharpen_var.get() else None,
            codec=ENCODER_CHOICES[self.encoder_choice.current()][1])

    # ---- files ----------------------------------------------------------------------------------

    def _browse_input(self) -> None:
        chosen = filedialog.askopenfilename(title="Image or video to upscale", filetypes=FILE_TYPES)
        if chosen:
            self.input_var.set(chosen)
            self._input_changed()

    def _browse_output(self) -> None:
        current = Path(self.output_var.get() or self.input_var.get() or ".")
        chosen = filedialog.asksaveasfilename(title="Save the upscaled file as", initialdir=str(current.parent),
                                              initialfile=current.name, defaultextension=current.suffix)
        if chosen:
            self.output_var.set(chosen)

    def _input_changed(self) -> None:
        text = self.input_var.get().strip().strip('"')
        self.input_var.set(text)
        path = Path(text)
        self.job_plan = None
        self.result_path = None
        self.open_button.configure(state="disabled")
        if not text:
            return
        if not path.is_file():
            self.status_var.set(f"Not found: {path}")
            return
        self.output_var.set(str(default_output(path)))
        try:
            self.job_plan = plan(self._options())
        except (ValueError, RuntimeError) as error:
            self.status_var.set(str(error))
            self._canvas_text(self.left_canvas, "Cannot read this file")
            return
        info = self.job_plan.info
        kind = "image" if self.job_plan.image_mode else (
            f"{info.codec} video, {float(info.frame_rate):.3g} fps, {info.frame_count or '?'} frames")
        self.status_var.set(f"{path.name}: {info.width}x{info.height}, {kind}")
        self._update_size_label()
        self.left_title.configure(text="Input")
        self.right_title.configure(text="Result (100% crop)")
        self._canvas_text(self.right_canvas, "Run to see the result")
        self._canvas_text(self.left_canvas, "Loading preview…")
        threading.Thread(target=self._render_input_preview, args=(path,), daemon=True).start()

    def _render_input_preview(self, path: Path) -> None:
        target = self.temp / "input_preview.png"
        try:
            make_preview(path, target, f"scale={PREVIEW_WIDTH}:{PREVIEW_HEIGHT}:force_original_aspect_ratio=decrease")
            self.queue.put(("preview", "left", target, "input"))
        except RuntimeError as error:
            self.queue.put(("log", f"[ui] {error}"))

    # ---- running --------------------------------------------------------------------------------

    def _run(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        if self.job_plan is None:
            self._input_changed()
            if self.job_plan is None:
                messagebox.showinfo("FSR4 upscaler", "Choose an image or a video first.")
                return
        try:
            options = self._options()
            self.job_plan = plan(options)
        except ValueError as error:
            messagebox.showerror("FSR4 upscaler", str(error))
            return
        self.cancel.clear()
        self.result_path = None
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.open_button.configure(state="disabled")
        self.progress.configure(value=0)
        self._clear_log()
        self.status_var.set("Starting…")
        self.worker = threading.Thread(target=self._work, args=(options,), daemon=True)
        self.worker.start()

    def _work(self, options: JobOptions) -> None:
        try:
            result = run_job(options, progress=lambda done, total, fps: self.queue.put(("progress", done, total, fps)),
                             cancel=self.cancel)
            self.queue.put(("done", result))
        except Exception as error:  # noqa: BLE001 - everything must reach the window
            self.queue.put(("error", error))

    def _cancel(self) -> None:
        self.cancel.set()
        self.status_var.set("Cancelling after the current frame…")
        self.cancel_button.configure(state="disabled")

    def _probe_gpu(self) -> None:
        try:
            from .ffx import FFX_UPSCALE_ENABLE_AUTO_EXPOSURE, Upscaler, find_sdk_dir
            from .gpu import Gpu
            gpu = Gpu(verbose=False)
            try:
                upscaler = Upscaler(gpu, (256, 144), (512, 288), FFX_UPSCALE_ENABLE_AUTO_EXPOSURE, find_sdk_dir(None))
                text = f"{gpu.device_name}  ·  FSR Upscaling {upscaler.version_name}  ·  Vulkan {gpu.api_version_text}"
                upscaler.destroy()
            finally:
                gpu.destroy()
        except Exception as error:  # noqa: BLE001 - shown in the status line
            text = f"GPU check failed: {error}"
        self.queue.put(("gpu", text))

    # ---- queue ----------------------------------------------------------------------------------

    def _poll(self) -> None:
        try:
            while True:
                message = self.queue.get_nowait()
                self._handle(message)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _handle(self, message: tuple) -> None:
        kind = message[0]
        if kind == "log":
            self._append_log(message[1])
        elif kind == "gpu":
            self.gpu_var.set(message[1])
        elif kind == "preview":
            _, side, path, key = message
            self._show_image(self.left_canvas if side == "left" else self.right_canvas, key, path)
        elif kind == "progress":
            _, done, total, fps = message
            if total:
                self.progress.configure(value=1000 * done / total)
                remaining = (total - done) / fps if fps else 0
                self.status_var.set(f"Frame {done} / {total}  ·  {fps:.1f} fps  ·  {remaining:.0f} s left")
            else:
                self.status_var.set(f"Frame {done}  ·  {fps:.1f} fps")
        elif kind == "titles":
            self.left_title.configure(text=message[1])
            self.right_title.configure(text=message[2])
        elif kind == "done":
            self._finished(message[1])
        elif kind == "error":
            self._append_log(f"[ui] error: {message[1]}")
            self.status_var.set(f"Failed: {message[1]}")
            messagebox.showerror("FSR4 upscaler", str(message[1]))
            self._idle()

    def _idle(self) -> None:
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")

    def _finished(self, result) -> None:
        self._idle()
        self.result_path = result.output
        self.open_button.configure(state="normal")
        self.progress.configure(value=1000)
        verb = "Cancelled after" if result.cancelled else "Done:"
        self.status_var.set(f"{verb} {result.frames} frames in {result.seconds:.1f} s "
                            f"({result.frames / max(result.seconds, 1e-6):.1f} fps)  →  {result.output.name}")
        if result.frames:
            threading.Thread(target=self._render_result_previews, args=(result,), daemon=True).start()

    def _render_result_previews(self, result) -> None:
        job = self.job_plan
        if job is None:
            return
        seek = None if job.image_mode else (result.frames // 2) / float(job.info.frame_rate or 30)
        crop = f"crop=min(iw\\,{PREVIEW_WIDTH}):min(ih\\,{PREVIEW_HEIGHT})"
        try:
            after = self.temp / "result_preview.png"
            make_preview(result.output, after, crop, seek)
            before = self.temp / "before_preview.png"
            scale = job.scale
            make_preview(Path(self.input_var.get()), before, f"scale=iw*{scale:.4f}:ih*{scale:.4f}:flags=neighbor,{crop}", seek)
            self.queue.put(("preview", "left", before, "before"))
            self.queue.put(("preview", "right", after, "after"))
            self.queue.put(("titles", "Input, enlarged without FSR (100% crop)", "FSR 4 (100% crop)"))
        except RuntimeError as error:
            self.queue.put(("log", f"[ui] {error}"))

    def _append_log(self, line: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line.rstrip("\n") + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _open_folder(self) -> None:
        if self.result_path is not None and self.result_path.exists():
            subprocess.Popen(["explorer", "/select,", str(self.result_path)])

    def _close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.cancel.set()
            self.worker.join(timeout=10)
        log.set_sink(None)
        shutil.rmtree(self.temp, ignore_errors=True)
        self.root.destroy()


def run_ui(preselected: Path | None = None) -> int:
    """Open the window; returns when it is closed."""
    if os.name == "nt":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    App(root, preselected)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(run_ui(Path(sys.argv[1]) if len(sys.argv) > 1 else None))
