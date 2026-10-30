"""Quality lab: downscale a reference, upscale it back with FSR 4 under several settings, measure PSNR.

Image lab (`--image ref.png`): the reference is downscaled by `--scale` (area filter), upscaled back with
and without sharpening, and compared to the reference. Bicubic and Lanczos (ffmpeg) are the baselines.
Alongside PSNR, the table shows the mean per-channel bias and the best PSNR over small integer shifts, which
separates "blurry" from "misaligned" and from "wrong colour transform".

Video lab (`--video gt.mkv`): the ground-truth clip is downscaled the same way and upscaled with every
flow setting; PSNR is averaged over the frames after the first `--warmup` ones.

    python tools/lab_quality.py --image out/lab/ref.png --video out/lab/gt.mkv --out out/lab
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fsr4 import media  # noqa: E402
from fsr4.ffx import find_sdk_dir  # noqa: E402
from fsr4.flow import create_flow  # noqa: E402
from fsr4.gpu import Gpu  # noqa: E402
from fsr4.pipeline import Pipeline  # noqa: E402


def psnr(image: np.ndarray, reference: np.ndarray) -> float:
    difference = image[..., :3].astype(np.float32) - reference[..., :3].astype(np.float32)
    mse = float(np.mean(difference * difference))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def bias(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.mean(image[..., :3].astype(np.float32) - reference[..., :3].astype(np.float32), axis=(0, 1))


def best_shift(image: np.ndarray, reference: np.ndarray, radius: int = 2) -> tuple[float, tuple[int, int]]:
    """PSNR after the best integer (dx, dy) shift of `image`, on the interior to avoid border effects."""
    best = (-1.0, (0, 0))
    height, width = reference.shape[:2]
    margin = radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = image[margin + dy: height - margin + dy, margin + dx: width - margin + dx]
            value = psnr(shifted, reference[margin: height - margin, margin: width - margin])
            if value > best[0]:
                best = (value, (dx, dy))
    return best


def ffmpeg_scale(source: Path, destination: Path, width: int, height: int, flags: str) -> None:
    subprocess.run([media.tool_path("ffmpeg"), "-v", "error", "-y", "-i", str(source),
                    "-vf", f"scale={width}:{height}:flags={flags}", str(destination)], check=True)


def run_pipeline_on_image(gpu: Gpu, sdk_dir: Path, frame: np.ndarray, output_size: tuple[int, int],
                          passes: int, sharpness: float | None) -> np.ndarray:
    height, width = frame.shape[:2]
    flow = create_flow("zero", width, height)
    pipeline = Pipeline(gpu, (width, height), output_size, sdk_dir, flow, sharpness=sharpness)
    try:
        for _ in range(passes):
            result = pipeline.process(frame)
        return result.copy()
    finally:
        pipeline.destroy()


def image_lab(gpu: Gpu, sdk_dir: Path, reference_path: Path, out_dir: Path, scale: float) -> list[str]:
    reference = media.read_image(reference_path)
    height, width = reference.shape[:2]
    render = (int(round(width / scale)) // 2 * 2, int(round(height / scale)) // 2 * 2)
    low_path = out_dir / f"{reference_path.stem}_low.png"
    ffmpeg_scale(reference_path, low_path, render[0], render[1], "area")
    low = media.read_image(low_path)
    lines = [f"## Image: {reference_path.name} ({width}x{height}), render {render[0]}x{render[1]}, scale {scale}x", "",
             "| setting | PSNR dB | best shifted PSNR (dx,dy) | bias R,G,B |", "|---|---|---|---|"]

    def report(name: str, image: np.ndarray) -> None:
        shifted, offset = best_shift(image, reference)
        channel_bias = bias(image, reference)
        lines.append(f"| {name} | {psnr(image, reference):.2f} | {shifted:.2f} {offset} | "
                     f"{channel_bias[0]:+.2f},{channel_bias[1]:+.2f},{channel_bias[2]:+.2f} |")
        print(lines[-1], flush=True)

    for flags in ("bicubic", "lanczos"):
        path = out_dir / f"{reference_path.stem}_{flags}.png"
        ffmpeg_scale(low_path, path, width, height, flags)
        report(flags, media.read_image(path))
    settings = [
        ("fsr4, 1 pass", 1, None),
        ("fsr4, 8 passes of the same frame", 8, None),
        ("fsr4, 1 pass, sharpness 0.3", 1, 0.3),
        ("fsr4, 1 pass, sharpness 0.6", 1, 0.6),
    ]
    for name, passes, sharpness in settings:
        image = run_pipeline_on_image(gpu, sdk_dir, low, (width, height), passes, sharpness)
        media.write_image(out_dir / f"{reference_path.stem}_fsr4_{passes}{f'_sharp{sharpness}' if sharpness else ''}.png", image)
        report(name, image)
    lines.append("")
    return lines


def video_lab(gpu: Gpu, sdk_dir: Path, truth_path: Path, out_dir: Path, scale: float, frames: int,
              warmup: int) -> list[str]:
    truth_info = media.probe(truth_path)
    width, height = truth_info.width, truth_info.height
    render = (int(round(width / scale)) // 2 * 2, int(round(height / scale)) // 2 * 2)
    low_path = out_dir / f"{truth_path.stem}_low.mkv"
    subprocess.run([media.tool_path("ffmpeg"), "-v", "error", "-y", "-i", str(truth_path),
                    "-vf", f"scale={render[0]}:{render[1]}:flags=area", "-frames:v", str(frames),
                    "-c:v", "ffv1", "-pix_fmt", "rgb24", str(low_path)], check=True)
    low_info = media.probe(low_path)
    truth_frames = [frame.copy() for frame in media.FrameReader(truth_path, truth_info, max_frames=frames)]
    low_frames = [frame.copy() for frame in media.FrameReader(low_path, low_info, max_frames=frames)]
    lines = [f"## Video: {truth_path.name} ({width}x{height}), render {render[0]}x{render[1]}, {len(low_frames)} frames, "
             f"PSNR over frames {warmup + 1}..{len(low_frames)}", "",
             "| setting | mean PSNR dB | min PSNR dB | ms per frame |", "|---|---|---|---|"]

    def report(name: str, values: list[float], ms: float) -> None:
        lines.append(f"| {name} | {np.mean(values):.2f} | {np.min(values):.2f} | {ms:.1f} |")
        print(lines[-1], flush=True)

    lanczos = []
    started = time.perf_counter()
    import cv2
    for low, truth in zip(low_frames, truth_frames):
        up = cv2.resize(low, (width, height), interpolation=cv2.INTER_LANCZOS4)
        lanczos.append(psnr(up, truth))
    report("lanczos (opencv)", lanczos[warmup:], (time.perf_counter() - started) * 1000 / len(low_frames))

    settings = [
        ("fsr4 flow zero", "zero", 1.0, "fast", 1),
        ("fsr4 flow dis fast", "dis", 1.0, "fast", 1),
        ("fsr4 flow dis fast negated", "dis", -1.0, "fast", 1),
        ("fsr4 flow dis ultrafast", "dis", 1.0, "ultrafast", 1),
        ("fsr4 flow dis medium", "dis", 1.0, "medium", 1),
        ("fsr4 flow dis fast at 1/2 res", "dis", 1.0, "fast", 2),
        ("fsr4 flow dis medium at 1/2 res", "dis", 1.0, "medium", 2),
    ]
    for name, flow_name, flow_sign, preset, flow_scale in settings:
        flow = create_flow(flow_name, render[0], render[1], preset=preset, scale=flow_scale)
        if flow_sign < 0:
            original = flow.compute
            flow.compute = lambda frame, original=original: np.negative(original(frame))
        pipeline = Pipeline(gpu, render, (width, height), sdk_dir, flow, frame_time_ms=low_info.frame_time_ms)
        values = []
        started = time.perf_counter()
        try:
            for index, (low, truth) in enumerate(zip(low_frames, truth_frames)):
                result = pipeline.process(low)
                values.append(psnr(result, truth))
                if index == len(low_frames) - 1:
                    tag = f"{flow_name}_{preset}_s{flow_scale}{'_neg' if flow_sign < 0 else ''}"
                    media.write_image(out_dir / f"{truth_path.stem}_{tag}_last.png", result)
                    media.write_image(out_dir / f"{truth_path.stem}_low_last.png", low)
        finally:
            pipeline.destroy()
        flow_ms = pipeline.timings.stages.get("flow", 0.0) / max(pipeline.timings.frames, 1)
        report(f"{name} (flow {flow_ms:.1f} ms)", values[warmup:], (time.perf_counter() - started) * 1000 / len(low_frames))
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, action="append", default=[], help="reference image (repeatable)")
    parser.add_argument("--video", type=Path, action="append", default=[], help="ground-truth clip (repeatable)")
    parser.add_argument("--out", type=Path, default=Path("out/lab"))
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sdk")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sdk_dir = find_sdk_dir(args.sdk)
    gpu = Gpu(verbose=False)
    lines = ["# Quality lab", ""]
    try:
        for image in args.image:
            lines += image_lab(gpu, sdk_dir, image, args.out, args.scale)
        for video in args.video:
            lines += video_lab(gpu, sdk_dir, video, args.out, args.scale, args.frames, args.warmup)
    finally:
        gpu.destroy()
    (args.out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"written {args.out / 'results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
