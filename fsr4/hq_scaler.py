"""AMD's HQ scaler (ffmpeg's `sr_amf` filter, Video SR 1.1): the upscaler beside the ones in the FSR DLL.

It runs inside ffmpeg, one frame at a time, with no motion vectors and no history. Reading its 4K frames back
through a pipe costs more than FSR 4's whole frame (62 ms against 24 ms, measured), so a video's frames stay
small: they go to the encoder's ffmpeg as they are, and it upscales them just before encoding (`EncoderScale`,
given to `FrameWriter`). `HqPipeline` stands in for `Pipeline`, so `upscale_stream` stays the one frame loop. A
single picture takes an ffmpeg run of its own (`upscale_picture`).
"""
from __future__ import annotations

import functools
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .flow import ZeroFlow
from .media import _CREATE_NO_WINDOW, ffmpeg_failure, tool_path
from .pipeline import Pipeline, Timings

HQ_SCALER = "amf-hq"
FIRST_FFMPEG = "8.0"
# The smallest width and height it takes (measured: 32x32 works, 24x24 and 32x18 do not).
MIN_SIZE = 32
# The filter counts down from its strongest sharpening (0); this is the driver's default, which keeps the
# source's detail (0.99 of it on one frame, measured). Its own "default" of -1 gives bit for bit the same.
NEUTRAL_SHARPNESS = 0.75
# The scaler and an AMF encoder deadlock at random on one shared device (5 runs of 8 hung). ffmpeg hands the
# encoder a device only when exactly one of its type exists, so an unused second one makes it open its own
# (36 runs of 36 finished).
DEVICE_ARGUMENTS = ["-init_hw_device", "amf=scaler", "-init_hw_device", "amf=spare", "-filter_hw_device", "scaler"]


@functools.cache
def available() -> bool:
    """Whether this ffmpeg has the scaler's filter."""
    return " sr_amf " in subprocess.run([tool_path("ffmpeg"), "-hide_banner", "-filters"], check=True,
                                        capture_output=True, text=True, creationflags=_CREATE_NO_WINDOW).stdout


def filter_sharpness(sharpness: float | None) -> float:
    """The filter's value for the tool's 0 to 1: off and 0 keep the source's detail, 1 is the filter's strongest."""
    return NEUTRAL_SHARPNESS * (1.0 - (sharpness or 0.0))


@dataclass(frozen=True)
class EncoderScale:
    """Tells `FrameWriter` that its frames come at `source_size` and that its ffmpeg upscales them."""
    source_size: tuple[int, int]
    sharpness: float | None = None

    def video_filter(self, width: int, height: int) -> str:
        """The filters that take the frames to `width` by `height`, back in system memory as RGBA."""
        return (f"hwupload,sr_amf=w={width}:h={height}:algorithm=sr1-1:sharpness={filter_sharpness(self.sharpness):.3f},"
                f"hwdownload,format=rgba")


def upscale_picture(rgba: np.ndarray, output_size: tuple[int, int], sharpness: float | None) -> np.ndarray:
    """One RGBA picture at `output_size`, opaque (the scaler writes no alpha)."""
    height, width = rgba.shape[:2]
    output_width, output_height = output_size
    command = [tool_path("ffmpeg"), "-v", "error", "-nostdin", *DEVICE_ARGUMENTS, "-f", "rawvideo", "-pix_fmt", "rgba",
               "-s", f"{width}x{height}", "-i", "-",
               "-vf", EncoderScale((width, height), sharpness).video_filter(output_width, output_height),
               "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgba", "-"]
    completed = subprocess.run(command, input=memoryview(np.ascontiguousarray(rgba)).cast("B"), capture_output=True,
                               creationflags=_CREATE_NO_WINDOW)
    if completed.returncode != 0 or len(completed.stdout) != output_width * output_height * 4:
        raise ffmpeg_failure(f"AMD's HQ scaler failed ({completed.returncode})",
                             completed.stderr.decode("utf-8", "replace"))
    picture = np.frombuffer(completed.stdout, np.uint8).reshape(output_height, output_width, 4).copy()
    picture[..., 3] = 255
    return picture


class HqPipeline:
    """Stands in for `Pipeline` with the HQ scaler: `process` hands a frame on as it is, to be upscaled by the
    encoder's ffmpeg (`encoder_scale`), and `picture` upscales here."""

    def __init__(self, render_size: tuple[int, int], output_size: tuple[int, int], sharpness: float | None = None):
        if not available():
            raise RuntimeError(f"AMD's HQ scaler needs ffmpeg {FIRST_FFMPEG} or later, built with AMF: this ffmpeg "
                               f"has no sr_amf filter")
        self.render_size = render_size
        self.output_size = output_size
        self.sharpness = sharpness
        self.flow = ZeroFlow(*render_size)
        self.timings = Timings()

    @property
    def encoder_scale(self) -> EncoderScale:
        return EncoderScale(self.render_size, self.sharpness)

    def reset(self) -> None:
        """Nothing to do: the scaler keeps no history."""

    def process(self, rgba: np.ndarray, motion: np.ndarray | None = None) -> np.ndarray:
        """The frame itself; see the class."""
        self.timings.frames += 1
        return rgba

    def picture(self, rgba: np.ndarray) -> np.ndarray:
        """Upscale one picture."""
        self.timings.frames += 1
        return upscale_picture(rgba, self.output_size, self.sharpness)

    def destroy(self) -> None:
        """Nothing to free: ffmpeg holds what the scaler uses."""


def create_pipeline(gpu, render_size: tuple[int, int], output_size: tuple[int, int], sdk_dir: Path | None, flow,
                    sharpness: float | None, upscaler: str | None, **options) -> Pipeline | HqPipeline:
    """The pipeline of `upscaler`, a `RenderSettings.upscaler` value. The HQ scaler needs neither `gpu` nor `sdk_dir`
    and takes no motion, so it ignores `flow`; `options` go to `Pipeline`."""
    if upscaler == HQ_SCALER:
        return HqPipeline(render_size, output_size, sharpness)
    return Pipeline(gpu, render_size, output_size, sdk_dir, flow, sharpness=sharpness, provider=upscaler, **options)
