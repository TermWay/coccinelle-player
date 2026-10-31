"""The per-frame pipeline: optical flow on the CPU, upload, FSR 4 dispatch, readback.

FSR expects a jittered, motion-vectored, depth-tested render. A video has none of that, so this module
substitutes what it can: motion vectors come from optical flow (`flow`), depth is a constant plane, and the
camera jitter is zero. The quality lab (`tools/lab_quality.py`) showed that faking the jitter by resampling
the input on FSR's Halton sequence only blurs the result, so it is not done.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import numpy as np

from . import ffx
from . import log
from . import vk_types as T
from .gpu import Gpu

SHADER_READ_STAGE = T.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
CONSTANT_DEPTH = 0.5


class Timings:
    """Accumulated milliseconds per stage over the processed frames."""

    def __init__(self):
        self.frames = 0
        self.stages: dict[str, float] = {}

    def add(self, stage: str, seconds: float) -> None:
        self.stages[stage] = self.stages.get(stage, 0.0) + seconds * 1000

    def per_frame(self, stage: str) -> float:
        return self.stages.get(stage, 0.0) / self.frames if self.frames else 0.0

    def summary(self) -> str:
        """Per-frame averages; stages marked "worker thread" overlap the others and are listed, not summed."""
        if self.frames == 0:
            return "no frames"
        parts = [f"{stage} {total / self.frames:.1f} ms" for stage, total in self.stages.items()]
        total = sum(value for stage, value in self.stages.items() if "worker thread" not in stage) / self.frames
        return f"{self.frames} frames, {total:.1f} ms per frame on the main thread ({1000 / total:.1f} fps): " + ", ".join(parts)


class Pipeline:
    """Upscales RGBA frames of one fixed size through an FSR 4 context."""

    def __init__(self, gpu: Gpu, render_size: tuple[int, int], output_size: tuple[int, int], sdk_dir: Path,
                 flow, sharpness: float | None = None, frame_time_ms: float = 1000 / 30,
                 debug_view: bool = False, debug_checking: bool = False, verbose: bool = False,
                 provider: str | None = None):
        self.gpu = gpu
        self.flow = flow
        self.sharpness = sharpness
        self.frame_time_ms = frame_time_ms
        self.debug_view = debug_view
        self.verbose = verbose
        self.timings = Timings()
        width, height = render_size
        output_width, output_height = output_size
        self.render_size = render_size
        self.output_size = output_size

        sampled = T.VK_IMAGE_USAGE_SAMPLED_BIT | T.VK_IMAGE_USAGE_TRANSFER_DST_BIT
        self.color = gpu.create_image(width, height, T.VK_FORMAT_R8G8B8A8_UNORM, sampled)
        self.depth = gpu.create_image(width, height, T.VK_FORMAT_R32_SFLOAT, sampled)
        self.motion = gpu.create_image(width, height, T.VK_FORMAT_R32G32_SFLOAT, sampled)
        self.output = gpu.create_image(output_width, output_height, T.VK_FORMAT_R8G8B8A8_UNORM,
                                       T.VK_IMAGE_USAGE_STORAGE_BIT | T.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                                       | T.VK_IMAGE_USAGE_SAMPLED_BIT)
        self.color_staging = gpu.create_staging_buffer(self.color.byte_size)
        self.motion_staging = gpu.create_staging_buffer(self.motion.byte_size)
        self.output_staging = gpu.create_staging_buffer(self.output.byte_size, readback=True)
        self.output_pixels = np.empty((output_height, output_width, 4), dtype=np.uint8)

        gpu.begin()
        gpu.cmd_clear(self.depth, CONSTANT_DEPTH, 0.0, 0.0, 0.0, T.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      SHADER_READ_STAGE, T.VK_ACCESS_SHADER_READ_BIT)
        gpu.submit_and_wait()

        flags = (ffx.FFX_UPSCALE_ENABLE_AUTO_EXPOSURE | ffx.FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE
                 | ffx.FFX_UPSCALE_ENABLE_DEPTH_INVERTED)
        if debug_view:
            flags |= ffx.FFX_UPSCALE_ENABLE_DEBUG_VISUALIZATION
        self.provider = provider
        self.upscaler = ffx.Upscaler(gpu, render_size, output_size, flags, sdk_dir, verbose=verbose,
                                     debug_checking=debug_checking, provider=provider)
        self.frame_index = 0
        if verbose:
            log.emit(f"[pipeline] flow {flow.name}, sharpness {sharpness if sharpness is not None else 'off'}")

    def reset(self) -> None:
        """Start a new sequence: the next frame does not blend with history."""
        self.flow.reset()
        self.frame_index = 0

    def process(self, rgba: np.ndarray, motion: np.ndarray | None = None) -> np.ndarray:
        """Upscale one frame. Returns a view of an internal buffer, valid until the next call.

        `motion` is the result of `flow.compute(rgba)`; pass it when it was computed ahead of time on another
        thread, otherwise it is computed here.
        """
        if rgba.shape != (self.render_size[1], self.render_size[0], 4) or rgba.dtype != np.uint8:
            raise ValueError(f"frame must be {self.render_size[1]}x{self.render_size[0]}x4 uint8, got {rgba.shape} {rgba.dtype}")
        reset = self.frame_index == 0
        if motion is None:
            started = time.perf_counter()
            motion = self.flow.compute(rgba)
            self.timings.add("flow", time.perf_counter() - started)

        started = time.perf_counter()
        self.color_staging.write(np.ascontiguousarray(rgba))
        self.motion_staging.write(np.ascontiguousarray(motion))
        self.timings.add("upload", time.perf_counter() - started)

        started = time.perf_counter()
        gpu = self.gpu
        command_buffer = gpu.begin()
        gpu.cmd_upload(self.color_staging, self.color, T.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       SHADER_READ_STAGE, T.VK_ACCESS_SHADER_READ_BIT)
        gpu.cmd_upload(self.motion_staging, self.motion, T.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       SHADER_READ_STAGE, T.VK_ACCESS_SHADER_READ_BIT)
        gpu.cmd_transition(self.output, T.VK_IMAGE_LAYOUT_GENERAL, T.VK_PIPELINE_STAGE_TRANSFER_BIT,
                           T.VK_ACCESS_TRANSFER_READ_BIT, SHADER_READ_STAGE, T.VK_ACCESS_SHADER_WRITE_BIT)
        flags = ffx.FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB
        if self.debug_view:
            flags |= ffx.FFX_UPSCALE_FLAG_DRAW_DEBUG_VIEW
        self.upscaler.dispatch(command_buffer, self.color, self.depth, self.motion, self.output, (0.0, 0.0),
                               self.frame_time_ms, reset, self.sharpness, flags)
        gpu.cmd_readback(self.output, self.output_staging, SHADER_READ_STAGE, T.VK_ACCESS_SHADER_WRITE_BIT)
        gpu.submit_and_wait()
        self.timings.add("gpu", time.perf_counter() - started)

        started = time.perf_counter()
        self.output_staging.read_into(self.output_pixels)
        self.timings.add("readback", time.perf_counter() - started)
        self.timings.frames += 1
        self.frame_index += 1
        return self.output_pixels

    def destroy(self) -> None:
        """Free the FSR context and this pipeline's images and buffers; the device stays for the next one."""
        self.gpu.wait_idle()
        self.upscaler.destroy()
        for image in (self.color, self.depth, self.motion, self.output):
            self.gpu.free_image(image)
        for buffer in (self.color_staging, self.motion_staging, self.output_staging):
            self.gpu.free_buffer(buffer)


def upscale_stream(frames: Iterator[np.ndarray], pipeline: Pipeline,
                   executor: ThreadPoolExecutor) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Upscale consecutive frames, yielding (source, upscaled) pairs valid until the next iteration.

    The optical flow of the next frame runs on `executor`'s single worker while the current frame is on the GPU
    and with the consumer (OpenCV and the Vulkan waits release the GIL). `frames` must keep the previous frame
    valid while it reads the next one (FrameReader double-buffers). Closing the generator early waits for the
    pending flow, so the worker never reads a buffer that is being overwritten.
    """
    flow = pipeline.flow

    def compute_flow(rgba: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        motion = flow.compute(rgba)
        pipeline.timings.add("flow (worker thread)", time.perf_counter() - started)
        return motion

    frame = next(frames, None)
    pending = executor.submit(compute_flow, frame) if frame is not None else None
    try:
        while frame is not None:
            motion = pending.result()
            following = next(frames, None)
            pending = executor.submit(compute_flow, following) if following is not None else None
            upscaled = pipeline.process(frame, motion)
            yield frame, upscaled
            frame = following
    finally:
        if pending is not None and not pending.done():
            pending.result()
