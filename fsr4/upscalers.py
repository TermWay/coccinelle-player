"""Which upscaler runs where: the ones in the FSR DLL, and those another program runs."""
from __future__ import annotations

from pathlib import Path

from .hq_scaler import HQ_SCALER, HqPipeline
from .pipeline import Pipeline
from .realesrgan import REAL_ESRGAN, RealEsrganPipeline

# Run by ffmpeg or a program of their own: no FSR DLL, no device here, no motion vectors.
EXTERNAL_UPSCALERS = frozenset({HQ_SCALER, REAL_ESRGAN})


def create_pipeline(gpu, render_size: tuple[int, int], output_size: tuple[int, int], sdk_dir: Path | None, flow,
                    sharpness: float | None, upscaler: str | None,
                    **options) -> Pipeline | HqPipeline | RealEsrganPipeline:
    """The pipeline of `upscaler`, a `RenderSettings.upscaler` value. `EXTERNAL_UPSCALERS` need neither `gpu` nor
    `sdk_dir` and ignore `flow`; `options` go to `Pipeline`."""
    if upscaler == HQ_SCALER:
        return HqPipeline(render_size, output_size, sharpness)
    if upscaler == REAL_ESRGAN:
        return RealEsrganPipeline(render_size, output_size)
    return Pipeline(gpu, render_size, output_size, sdk_dir, flow, sharpness=sharpness, provider=upscaler, **options)
