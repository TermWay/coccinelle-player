"""The per-frame pipeline: optical flow on the CPU, upload, FSR 4 dispatch, readback.

FSR expects a jittered, motion-vectored, depth-tested render. A video has none of that, so this module
fakes what it can: motion vectors come from optical flow (`flow`), depth is a constant plane, and the
optional "fake jitter" resamples the input frame on FSR's Halton sequence so that the upscaler sees the
sub-pixel sample positions it was trained on.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import ffx
from . import vk_types as T
from .gpu import Gpu

JITTER_MODES = ("fake", "none")

SHADER_READ_STAGE = T.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
CONSTANT_DEPTH = 0.5


class Timings:
    """Accumulated milliseconds per stage over the processed frames."""

    def __init__(self):
        self.frames = 0
        self.stages: dict[str, float] = {}

    def add(self, stage: str, seconds: float) -> None:
        self.stages[stage] = self.stages.get(stage, 0.0) + seconds * 1000

    def summary(self) -> str:
        if self.frames == 0:
            return "no frames"
        parts = [f"{stage} {total / self.frames:.1f} ms" for stage, total in self.stages.items()]
        total = sum(self.stages.values()) / self.frames
        return f"{self.frames} frames, {total:.1f} ms per frame ({1000 / total:.1f} fps): " + ", ".join(parts)


class Pipeline:
    """Upscales RGBA frames of one fixed size through an FSR 4 context."""

    def __init__(self, gpu: Gpu, render_size: tuple[int, int], output_size: tuple[int, int], sdk_dir: Path,
                 flow, jitter_mode: str = "fake", sharpness: float | None = None, frame_time_ms: float = 1000 / 30,
                 debug_view: bool = False, debug_checking: bool = False, verbose: bool = False,
                 jitter_sign: float = -1.0):
        if jitter_mode not in JITTER_MODES:
            raise ValueError(f"unknown jitter mode {jitter_mode}; choose from {', '.join(JITTER_MODES)}")
        self.gpu = gpu
        self.flow = flow
        self.jitter_mode = jitter_mode
        self.jitter_sign = jitter_sign
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
        self.motion = gpu.create_image(width, height, T.VK_FORMAT_R16G16_SFLOAT, sampled)
        self.output = gpu.create_image(output_width, output_height, T.VK_FORMAT_R8G8B8A8_UNORM,
                                       T.VK_IMAGE_USAGE_STORAGE_BIT | T.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                                       | T.VK_IMAGE_USAGE_SAMPLED_BIT)
        self.color_staging = gpu.create_staging_buffer(self.color.byte_size)
        self.motion_staging = gpu.create_staging_buffer(self.motion.byte_size)
        self.output_staging = gpu.create_staging_buffer(self.output.byte_size)
        self.output_pixels = np.empty((output_height, output_width, 4), dtype=np.uint8)
        self._jittered = np.empty((height, width, 4), dtype=np.uint8)

        gpu.begin()
        gpu.cmd_clear(self.depth, CONSTANT_DEPTH, 0.0, 0.0, 0.0, T.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      SHADER_READ_STAGE, T.VK_ACCESS_SHADER_READ_BIT)
        gpu.submit_and_wait()

        flags = (ffx.FFX_UPSCALE_ENABLE_AUTO_EXPOSURE | ffx.FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE
                 | ffx.FFX_UPSCALE_ENABLE_DEPTH_INVERTED)
        if debug_view:
            flags |= ffx.FFX_UPSCALE_ENABLE_DEBUG_VISUALIZATION
        self.upscaler = ffx.Upscaler(gpu, render_size, output_size, flags, sdk_dir, verbose=verbose,
                                     debug_checking=debug_checking)
        self.phase_count = self.upscaler.jitter_phase_count()
        self.frame_index = 0
        self._warp = None
        if jitter_mode == "fake":
            try:
                import cv2
            except ImportError as error:
                raise RuntimeError("--jitter fake needs the opencv-python package; use --jitter none") from error
            self._warp = cv2
        if verbose:
            print(f"[pipeline] jitter {jitter_mode} ({self.phase_count} phases), flow {flow.name}, "
                  f"sharpness {sharpness if sharpness is not None else 'off'}")

    def reset(self) -> None:
        """Start a new sequence: the next frame does not blend with history."""
        self.flow.reset()
        self.frame_index = 0

    def _jitter(self) -> tuple[tuple[float, float], np.ndarray | None]:
        """The dispatch jitter for this frame and, in fake mode, the resampled input to match it."""
        if self.jitter_mode == "none":
            return (0.0, 0.0), None
        jitter_x, jitter_y = self.upscaler.jitter_offset(self.frame_index, self.phase_count)
        return (self.jitter_sign * jitter_x, self.jitter_sign * jitter_y), (jitter_x, jitter_y)

    def process(self, rgba: np.ndarray) -> np.ndarray:
        """Upscale one frame. Returns a view of an internal buffer, valid until the next call."""
        if rgba.shape != (self.render_size[1], self.render_size[0], 4) or rgba.dtype != np.uint8:
            raise ValueError(f"frame must be {self.render_size[1]}x{self.render_size[0]}x4 uint8, got {rgba.shape} {rgba.dtype}")
        reset = self.frame_index == 0
        started = time.perf_counter()
        motion = self.flow.compute(rgba)
        self.timings.add("flow", time.perf_counter() - started)

        started = time.perf_counter()
        dispatch_jitter, content_shift = self._jitter()
        color_input = rgba
        if content_shift is not None:
            matrix = np.array([[1.0, 0.0, content_shift[0]], [0.0, 1.0, content_shift[1]]], dtype=np.float32)
            self._warp.warpAffine(rgba, matrix, (rgba.shape[1], rgba.shape[0]), dst=self._jittered,
                                  flags=self._warp.INTER_LINEAR, borderMode=self._warp.BORDER_REFLECT_101)
            color_input = self._jittered
        self.color_staging.write(np.ascontiguousarray(color_input))
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
        self.upscaler.dispatch(command_buffer, self.color, self.depth, self.motion, self.output, dispatch_jitter,
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
        self.upscaler.destroy()
