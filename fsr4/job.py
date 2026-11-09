"""One upscaling job (image or video) with progress, live preview and cancellation, shared by the CLI and the UI."""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np

from . import log
from .ffx import QUALITY_MODE_SCALES, find_sdk_dir
from .flow import create_flow
from .hq_scaler import HQ_SCALER, MIN_SIZE as HQ_MIN_SIZE
from .media import (ENCODER_LIMITS, ENCODER_NAMES, FrameReader, FrameWriter, MediaInfo, describe_audio, is_image,
                    partial_path, probe, read_image, write_image)
from .pipeline import upscale_stream
from .realesrgan import MODELS, NOT_INSTALLED, STILL_ONLY, STILL_ONLY_NOTE, Stopped, find_program
from .upscalers import EXTERNAL_UPSCALERS, create_pipeline

VIDEO_SUFFIX = ".mp4"
MIN_SCALE = 1.0
MAX_SCALE = 3.0
PROGRESS_INTERVAL = 0.1
PREVIEW_INTERVAL = 0.5

ProgressCallback = Callable[[int, int | None, float], None]
PreviewCallback = Callable[[np.ndarray, np.ndarray, int], None]


@dataclass
class JobOptions:
    """Everything that defines a job; `scale`, `width`/`height` override `mode` in that order."""
    input: Path
    output: Path | None = None
    mode: str = "performance"
    scale: float | None = None
    width: int | None = None
    height: int | None = None
    flow: str = "dis"
    flow_preset: str = "fast"
    flow_scale: int = 1
    sharpness: float | None = None
    codec: str = "amf"
    upscaler: str | None = None
    frames: int | None = None
    debug_view: bool = False
    debug_checking: bool = False
    validate: bool = False
    gpu: int | None = None
    sdk: str | None = None
    verbose: bool = True


@dataclass
class JobPlan:
    """The resolved sizes and paths of a job, before any GPU work."""
    info: MediaInfo
    output: Path
    render_size: tuple[int, int]
    output_size: tuple[int, int]
    image_mode: bool

    @property
    def scales(self) -> tuple[float, float]:
        """Output pixels per source pixel across and down (they differ for an anamorphic source)."""
        return self.output_size[0] / self.render_size[0], self.output_size[1] / self.render_size[1]

    def describe_scale(self, times: str = "x") -> str:
        """"2.00x", or "2.37x by 2.00x" for an anamorphic source."""
        across, down = self.scales
        if abs(across - down) < 0.005:
            return f"{across:.2f}{times}"
        return f"{across:.2f}{times} by {down:.2f}{times}"

    @property
    def total_frames(self) -> int | None:
        return 1 if self.image_mode else self.info.frame_count


@dataclass
class JobResult:
    output: Path
    frames: int
    seconds: float
    cancelled: bool
    timings: str


def round_to(value: float, multiple: int) -> int:
    """Round to the nearest multiple of `multiple` (at least `multiple`)."""
    return max(multiple, int(round(value / multiple)) * multiple)


def fit(value: float, source: int, multiple: int) -> int:
    """`value` rounded to a multiple of `multiple`, one step back inside FSR's 1x-3x range if rounding left it
    (an odd source at 3x would otherwise ask for a hair more than 3x)."""
    size = round_to(value, multiple)
    if size > source * MAX_SCALE:
        size -= multiple
    elif size < source * MIN_SCALE:
        size += multiple
    return size


def default_output(path: Path) -> Path:
    suffix = path.suffix if is_image(path) else VIDEO_SUFFIX
    return path.with_name(f"{path.stem}.fsr4{suffix}")


def output_size(width: int, height: int, options: JobOptions, pixel_aspect: Fraction = Fraction(1)) -> tuple[int, int]:
    """The width and height asked for, the other side following the source's shape, or the scale. The shape is
    the one players show: an anamorphic source (`pixel_aspect` not 1, a DVD) comes out with square pixels, FSR
    widening it by the pixel aspect too (measured as good as one scale), and a scale that would take that width
    outside FSR's 1x-3x range is moved inside it. Video encoders need even sizes, and AV1 on the GPU widths that
    are multiples of 8: it pads other widths and tags the padded one, so the file reads back sheared."""
    across = 8 if options.codec == "av1_amf" and not is_image(options.input) else 2
    shown_width = width * float(pixel_aspect)
    if options.width and options.height:
        return round_to(options.width, across), round_to(options.height, 2)
    if options.width:
        return round_to(options.width, across), fit(options.width * height / shown_width, height, 2)
    if options.height:
        return fit(options.height * shown_width / height, width, across), round_to(options.height, 2)
    scale = options.scale if options.scale else QUALITY_MODE_SCALES[options.mode]
    if pixel_aspect != 1 and MIN_SCALE <= scale <= MAX_SCALE:
        scale = min(max(scale, MIN_SCALE / float(pixel_aspect)), MAX_SCALE / float(pixel_aspect))
    return fit(shown_width * scale, width, across), fit(height * scale, height, 2)


def check_output(output: Path, source: Path) -> None:
    """Raise ValueError when writing `output` would overwrite `source`, under any spelling of its path."""
    if output.exists() and os.path.samefile(output, source):
        raise ValueError(f"the output is the input file {source.name}; choose another output name")
    partial = partial_path(output)
    if partial.exists() and os.path.samefile(partial, source):
        raise ValueError(f"{output.name} is written as {partial.name} until it is complete, which is the input "
                         f"file; choose another output name")


def check_hq_scaler(options: JobOptions, info: MediaInfo) -> None:
    """Raise ValueError for what AMD's HQ scaler cannot do: small inputs, and the options that belong to FSR."""
    if min(info.width, info.height) < HQ_MIN_SIZE:
        raise ValueError(f"AMD's HQ scaler takes at least {HQ_MIN_SIZE} × {HQ_MIN_SIZE}; this is {info.width} × "
                         f"{info.height}")
    if options.debug_view or options.debug_checking or options.validate or options.gpu is not None:
        raise ValueError("--debug-view, --debug-checking, --validate and --gpu do not apply to AMD's HQ scaler")


def check_realesrgan(options: JobOptions) -> None:
    """Raise ValueError when Real-ESRGAN is not installed, and for the options that do not apply to it."""
    if (options.sharpness is not None or options.debug_view or options.debug_checking or options.validate
            or options.gpu is not None):
        raise ValueError("--sharpness, --debug-view, --debug-checking, --validate and --gpu do not apply to Real-ESRGAN")
    if options.upscaler in STILL_ONLY and not is_image(options.input):
        raise ValueError(STILL_ONLY_NOTE)
    if find_program(options.upscaler) is None:
        raise ValueError(NOT_INSTALLED)


def plan_from_info(options: JobOptions, info: MediaInfo) -> JobPlan:
    """Resolve the output path and sizes for an already probed input; raises ValueError for bad scales, a
    sharpening outside 0 to 1 and an output that would overwrite the input."""
    if options.sharpness is not None and not 0.0 <= options.sharpness <= 1.0:
        raise ValueError(f"sharpening {options.sharpness:g} is not between 0 and 1")
    if options.upscaler == HQ_SCALER:
        check_hq_scaler(options, info)
    elif options.upscaler in MODELS:
        check_realesrgan(options)
    output = options.output or default_output(options.input)
    check_output(output, options.input)
    width, height = info.width, info.height
    out_width, out_height = output_size(width, height, options, info.pixel_aspect)
    scale_x, scale_y = out_width / width, out_height / height
    if not (MIN_SCALE - 1e-6 <= scale_x <= MAX_SCALE + 1e-6 and MIN_SCALE - 1e-6 <= scale_y <= MAX_SCALE + 1e-6):
        raise ValueError(f"FSR supports scales from 1x to 3x; {width}x{height} -> {out_width}x{out_height} "
                         f"is {scale_x:.2f}x")
    image_mode = is_image(options.input)
    limits = ENCODER_LIMITS.get(options.codec)
    if not image_mode and limits is not None:
        (low_width, low_height), (high_width, high_height) = limits
        if not (low_width <= out_width <= high_width and low_height <= out_height <= high_height):
            raise ValueError(f"{ENCODER_NAMES[options.codec]} encodes from {low_width} × {low_height} to "
                             f"{high_width} × {high_height}; {out_width} × {out_height} needs another encoder "
                             f"or another size")
    return JobPlan(info=info, output=output, render_size=(width, height), output_size=(out_width, out_height),
                   image_mode=image_mode)


def plan(options: JobOptions) -> JobPlan:
    """Probe the input and resolve the output path and sizes; raises ValueError for bad inputs."""
    if not options.input.is_file():
        raise ValueError(f"input not found: {options.input}")
    return plan_from_info(options, probe(options.input))


def run_job(options: JobOptions, progress: ProgressCallback | None = None,
            cancel: threading.Event | None = None, preview: PreviewCallback | None = None) -> JobResult:
    """Run the whole job on the calling thread.

    `progress(frames_done, total, fps)` is called about ten times a second. `preview(source, upscaled,
    frames_done)` receives copies of the current input and output frames about twice a second and after the
    last frame (with the HQ scaler, a video's frames as they go to the encoder, which upscales them). Set `cancel`
    to stop after the current frame: a video keeps what was written so far, an image is
    not written (`JobResult.frames` is 0).
    """
    from .gpu import Gpu
    job = plan(options)
    uses_fsr = options.upscaler not in EXTERNAL_UPSCALERS
    sdk_dir = find_sdk_dir(options.sdk) if uses_fsr else None
    width, height = job.render_size
    out_width, out_height = job.output_size
    if options.verbose:
        kind = "image" if job.image_mode else f"{job.info.codec} video, {float(job.info.frame_rate):.3f} fps"
        log.emit(f"[fsr4] {options.input.name}: {width}x{height} -> {out_width}x{out_height} ({job.describe_scale()}), {kind}")
        if not job.image_mode:
            for line in describe_audio(job.info.audio, job.output):
                log.emit(f"[fsr4] {line}")

    gpu = Gpu(gpu_index=options.gpu, validate=options.validate, verbose=options.verbose) if uses_fsr else None
    pipeline = None
    try:
        flow = create_flow(options.flow if uses_fsr and not job.image_mode else "zero", width, height,
                           options.flow_preset, options.flow_scale)
        pipeline = create_pipeline(gpu, job.render_size, job.output_size, sdk_dir, flow, options.sharpness,
                                   options.upscaler, frame_time_ms=job.info.frame_time_ms,
                                   debug_view=options.debug_view, debug_checking=options.debug_checking,
                                   verbose=options.verbose)
        started = time.perf_counter()
        cancelled = False
        if job.image_mode:
            source = read_image(options.input)
            if cancel is not None:
                pipeline.stop = cancel  # only the slow ones read it
            try:
                upscaled = pipeline.picture(source)
            except Stopped:
                upscaled = None
            if upscaled is None or (cancel is not None and cancel.is_set()):
                cancelled, frames_done = True, 0
            else:
                transparent = bool((source[..., 3] < 255).any())
                write_image(job.output, upscaled, alpha_source=options.input if transparent else None)
                frames_done = 1
                if preview:
                    preview(source, upscaled.copy(), 1)
                if progress:
                    progress(1, 1, 1 / max(time.perf_counter() - started, 1e-6))
        else:
            frames_done, cancelled = _run_video(options, job, pipeline, progress, preview, cancel, started)
        elapsed = time.perf_counter() - started
        if options.verbose:
            if cancelled and not frames_done:
                log.emit(f"[fsr4] cancelled before writing {job.output}")
            else:
                verb = "cancelled after" if cancelled else "wrote"
                log.emit(f"[fsr4] {verb} {job.output} ({frames_done} frames in {elapsed:.1f} s, "
                         f"{frames_done / max(elapsed, 1e-6):.1f} fps)")
            log.emit(f"[fsr4] {pipeline.timings.summary()}")
        return JobResult(output=job.output, frames=frames_done, seconds=elapsed, cancelled=cancelled,
                         timings=pipeline.timings.summary())
    finally:
        try:
            if pipeline is not None:
                pipeline.destroy()
        finally:
            if gpu is not None:
                gpu.destroy()


def _run_video(options: JobOptions, job: JobPlan, pipeline, progress: ProgressCallback | None,
               preview: PreviewCallback | None, cancel: threading.Event | None, started: float) -> tuple[int, bool]:
    info = job.info
    reader = FrameReader(options.input, info, max_frames=options.frames)
    audio_seconds = options.frames / float(info.frame_rate) if options.frames and info.frame_rate else None
    writer = FrameWriter(job.output, job.output_size[0], job.output_size[1], info.frame_rate, options.codec,
                         audio_source=options.input, audio=info.audio, audio_seconds=audio_seconds,
                         scale=pipeline.encoder_scale)
    total = info.frame_count
    if options.frames is not None:
        total = min(options.frames, total) if total else options.frames
    frames_done = 0
    cancelled = False
    exhausted = False
    last_report = last_preview = started
    frame = upscaled = None
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            stream = upscale_stream(iter(reader), pipeline, executor)
            for frame, upscaled in stream:
                writer.write(upscaled)
                frames_done += 1
                now = time.perf_counter()
                if preview and now - last_preview >= PREVIEW_INTERVAL:
                    preview(frame.copy(), upscaled.copy(), frames_done)
                    last_preview = now
                if progress and now - last_report >= PROGRESS_INTERVAL:
                    progress(frames_done, total, frames_done / (now - started))
                    last_report = now
                if cancel is not None and cancel.is_set():
                    cancelled = True
                    break
            stream.close()
            exhausted = not cancelled
    except BaseException:
        reader.stop()
        writer.abort()
        raise
    if not exhausted:
        reader.stop()
    writer.close()
    # The pair from the last iteration is still intact: the reader stopped and the pipeline ran no further.
    if frames_done:
        if preview:
            preview(frame.copy(), upscaled.copy(), frames_done)
        if progress:
            progress(frames_done, frames_done if exhausted else total,
                     frames_done / max(time.perf_counter() - started, 1e-6))
    return frames_done, cancelled
