"""One upscaling job (image or video) with progress reporting and cancellation, shared by the CLI and the UI."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import log
from .ffx import QUALITY_MODE_SCALES, find_sdk_dir
from .flow import create_flow
from .media import FrameReader, FrameWriter, MediaInfo, is_image, probe, read_image, write_image
from .pipeline import Pipeline

VIDEO_SUFFIX = ".mp4"
MIN_SCALE = 1.0
MAX_SCALE = 3.0

ProgressCallback = Callable[[int, int | None, float], None]


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
    def scale(self) -> float:
        return self.output_size[0] / self.render_size[0]

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


def even(value: float) -> int:
    """Round to the nearest even integer (video encoders need even dimensions)."""
    return max(2, int(round(value / 2)) * 2)


def default_output(path: Path) -> Path:
    suffix = path.suffix if is_image(path) else VIDEO_SUFFIX
    return path.with_name(f"{path.stem}.fsr4{suffix}")


def output_size(width: int, height: int, options: JobOptions) -> tuple[int, int]:
    if options.width or options.height:
        if options.width and options.height:
            return even(options.width), even(options.height)
        if options.width:
            return even(options.width), even(options.width * height / width)
        return even(options.height * width / height), even(options.height)
    scale = options.scale if options.scale else QUALITY_MODE_SCALES[options.mode]
    return even(width * scale), even(height * scale)


def plan(options: JobOptions) -> JobPlan:
    """Probe the input and resolve the output path and sizes; raises ValueError for bad inputs."""
    if not options.input.is_file():
        raise ValueError(f"input not found: {options.input}")
    info = probe(options.input)
    width, height = info.width, info.height
    out_width, out_height = output_size(width, height, options)
    scale_x, scale_y = out_width / width, out_height / height
    if not (MIN_SCALE <= scale_x <= MAX_SCALE + 1e-6 and MIN_SCALE <= scale_y <= MAX_SCALE + 1e-6):
        raise ValueError(f"FSR supports scales from 1x to 3x; {width}x{height} -> {out_width}x{out_height} "
                         f"is {scale_x:.2f}x")
    return JobPlan(info=info, output=options.output or default_output(options.input),
                   render_size=(width, height), output_size=(out_width, out_height),
                   image_mode=is_image(options.input))


def run_job(options: JobOptions, progress: ProgressCallback | None = None,
            cancel: threading.Event | None = None) -> JobResult:
    """Run the whole job on the calling thread. `progress(frames_done, total, fps)` is called about ten
    times a second; set `cancel` to stop after the current frame (the output written so far is kept)."""
    from .gpu import Gpu
    job = plan(options)
    sdk_dir = find_sdk_dir(options.sdk)
    width, height = job.render_size
    out_width, out_height = job.output_size
    if options.verbose:
        kind = "image" if job.image_mode else f"{job.info.codec} video, {float(job.info.frame_rate):.3f} fps"
        log.emit(f"[fsr4] {options.input.name}: {width}x{height} -> {out_width}x{out_height} ({job.scale:.2f}x), {kind}")

    gpu = Gpu(gpu_index=options.gpu, validate=options.validate, verbose=options.verbose)
    pipeline = None
    try:
        flow = create_flow(options.flow if not job.image_mode else "zero", width, height,
                           options.flow_preset, options.flow_scale)
        pipeline = Pipeline(gpu, job.render_size, job.output_size, sdk_dir, flow, sharpness=options.sharpness,
                            frame_time_ms=job.info.frame_time_ms, debug_view=options.debug_view,
                            debug_checking=options.debug_checking, verbose=options.verbose)
        started = time.perf_counter()
        cancelled = False
        if job.image_mode:
            write_image(job.output, pipeline.process(read_image(options.input)))
            frames_done = 1
            if progress:
                progress(1, 1, 1 / max(time.perf_counter() - started, 1e-6))
        else:
            frames_done, cancelled = _run_video(options, job, pipeline, flow, progress, cancel, started)
        elapsed = time.perf_counter() - started
        if options.verbose:
            verb = "cancelled after" if cancelled else "wrote"
            log.emit(f"[fsr4] {verb} {job.output} ({frames_done} frames in {elapsed:.1f} s, "
                     f"{frames_done / max(elapsed, 1e-6):.1f} fps)")
            log.emit(f"[fsr4] {pipeline.timings.summary()}")
        return JobResult(output=job.output, frames=frames_done, seconds=elapsed, cancelled=cancelled,
                         timings=pipeline.timings.summary())
    finally:
        if pipeline is not None:
            pipeline.destroy()
        gpu.destroy()


def _run_video(options: JobOptions, job: JobPlan, pipeline: Pipeline, flow, progress: ProgressCallback | None,
               cancel: threading.Event | None, started: float) -> tuple[int, bool]:
    reader = FrameReader(options.input, job.info, max_frames=options.frames)
    writer = FrameWriter(job.output, job.output_size[0], job.output_size[1], job.info.frame_rate, options.codec,
                         audio_source=options.input if job.info.has_audio else None)
    total = job.info.frame_count
    if options.frames is not None:
        total = min(options.frames, total) if total else options.frames
    frames_done = 0
    cancelled = False
    last_report = started

    # Optical flow for the next frame runs on a worker thread while this one is on the GPU and in the
    # encoder pipe (OpenCV and the Vulkan waits release the GIL). The reader double-buffers, so `frame`
    # stays valid while `following` is read.
    def compute_flow(rgba):
        flow_started = time.perf_counter()
        motion = flow.compute(rgba)
        pipeline.timings.add("flow (worker thread)", time.perf_counter() - flow_started)
        return motion

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            frames = iter(reader)
            frame = next(frames, None)
            pending = executor.submit(compute_flow, frame) if frame is not None else None
            while frame is not None:
                motion = pending.result()
                following = next(frames, None)
                pending = executor.submit(compute_flow, following) if following is not None else None
                writer.write(pipeline.process(frame, motion))
                frame = following
                frames_done += 1
                now = time.perf_counter()
                if progress and (now - last_report >= 0.1 or frame is None):
                    progress(frames_done, total, frames_done / (now - started))
                    last_report = now
                if cancel is not None and cancel.is_set():
                    cancelled = True
                    if pending is not None:
                        pending.result()
                    break
    finally:
        if cancelled:
            reader.process.kill()
        writer.close()
    return frames_done, cancelled
