"""Quality lab: downscale a reference, upscale it back with FSR 4 under several settings, measure PSNR.

Image lab (`--image ref.png`): the reference is downscaled by `--scale` (area filter), upscaled back with
and without sharpening, and compared to the reference. Bicubic and Lanczos (ffmpeg) are the baselines; AMD's
HQ scaler (`AMF_SETTINGS`, ffmpeg's `sr_amf`) is measured beside them when this ffmpeg has it, and Real-ESRGAN
(`REALESRGAN_SETTINGS`) when `--realesrgan` names its realesrgan-ncnn-vulkan.exe.
Alongside PSNR, the table shows the mean per-channel bias and the best PSNR over small integer shifts, which
separates "blurry" from "misaligned" and from "wrong colour transform".

Video lab (`--video gt.mkv`): the ground-truth clip is downscaled the same way and upscaled with every
flow setting; PSNR is averaged over the frames after the first `--warmup` ones. `--compress CRF` puts the small
clip through H.264 first, as a real file would be. Temporal PSNR scores each change between two frames against
the same change in the clip: flicker lowers it. Times cover the upscaling, not the PSNR; the AMF rows are a whole
ffmpeg run, so theirs also cover decoding, upload and read back, and the Real-ESRGAN rows a whole run of its exe,
PNG files in and out.

    python tools/lab_quality.py --image out/lab/ref.png --video out/lab/gt.mkv --out out/lab
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fsr4 import hq_scaler, media  # noqa: E402
from fsr4.ffx import find_sdk_dir  # noqa: E402
from fsr4.flow import create_flow  # noqa: E402
from fsr4.gpu import Gpu  # noqa: E402
from fsr4.pipeline import Pipeline  # noqa: E402


# (name, sr_amf algorithm, sr_amf sharpness); -1 leaves the sharpness to the driver. The value counts stops down
# from the strongest, as in FSR's RCAS: measured, detail energy is 2.9 times the reference at 0, 0.7 times at 2.
AMF_SETTINGS = [
    ("amf bicubic", "bicubic", -1.0),
    ("amf video SR 1.0", "sr1-0", -1.0),
    ("amf video SR 1.1", "sr1-1", -1.0),
    ("amf video SR 1.1, sharpness 0 (strongest)", "sr1-1", 0.0),
    ("amf video SR 1.1, sharpness 1", "sr1-1", 1.0),
    ("amf video SR 1.1, sharpness 2 (mildest)", "sr1-1", 2.0),
]

# (name, model, the scale it runs at); a model that only runs above the lab's scale is area averaged back to it.
REALESRGAN_SETTINGS = [
    ("real-esrgan animevideo v3", "realesr-animevideov3", 2),
    ("real-esrgan x4plus", "realesrgan-x4plus", 4),
]


def psnr(image: np.ndarray, reference: np.ndarray) -> float:
    difference = image[..., :3].astype(np.float32) - reference[..., :3].astype(np.float32)
    mse = float(np.mean(difference * difference))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


class FrameScores:
    """Scores one setting's frames, given in order, against the reference frames."""

    def __init__(self, truth_frames: list[np.ndarray]) -> None:
        self.truth_frames = truth_frames
        self.psnr: list[float] = []
        self.temporal: list[float] = []
        self.previous: np.ndarray | None = None

    def add(self, image: np.ndarray) -> None:
        """Scores the next frame, and its change from the one before."""
        index = len(self.psnr)
        truth = self.truth_frames[index]
        self.psnr.append(psnr(image, truth))
        current = image[..., :3].astype(np.int16)
        if self.previous is not None:
            before = self.truth_frames[index - 1][..., :3].astype(np.int16)
            self.temporal.append(psnr(current - self.previous, truth[..., :3].astype(np.int16) - before))
        self.previous = current


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


def has_amf_scaler() -> bool:
    """Whether this ffmpeg has `sr_amf`; says so when it does not, the lab then skips those rows."""
    if not hq_scaler.available():
        print("this ffmpeg has no sr_amf filter: the AMF rows are skipped", flush=True)
        return False
    return True


def amf_command(source: Path, width: int, height: int, algorithm: str, sharpness: float) -> list[str]:
    """The ffmpeg command up to its output: `source` through the HQ scaler on the GPU, RGBA in and out."""
    scaler = f"sr_amf=w={width}:h={height}:algorithm={algorithm}:sharpness={sharpness}"
    return [media.tool_path("ffmpeg"), "-v", "error", "-nostdin", "-init_hw_device", "amf=amf",
            "-filter_hw_device", "amf", "-i", str(source),
            "-vf", f"format=rgba,hwupload,{scaler},hwdownload,format=rgba"]


def amf_upscale(source: Path, width: int, height: int, algorithm: str, sharpness: float) -> np.ndarray:
    """Every frame of `source` upscaled by the HQ scaler, as RGBA of shape (frames, height, width, 4)."""
    command = amf_command(source, width, height, algorithm, sharpness) + ["-f", "rawvideo", "-pix_fmt", "rgba", "-"]
    output = subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(output, np.uint8).reshape(-1, height, width, 4)


def amf_ms_per_frame(source: Path, width: int, height: int, algorithm: str, sharpness: float, frames: int) -> float:
    """Decoding, upload, scaling and read back per frame; a one-frame run takes the start-up out."""
    def seconds(count: int) -> float:
        started = time.perf_counter()
        subprocess.run(amf_command(source, width, height, algorithm, sharpness)
                       + ["-frames:v", str(count), "-f", "null", "-"], check=True)
        return time.perf_counter() - started

    return (seconds(frames) - seconds(1)) * 1000 / (frames - 1)


def realesrgan_upscale(executable: Path, source: Path, destination: Path, model: str, scale: int) -> float:
    """Runs realesrgan-ncnn-vulkan on a file or a folder of PNG frames (GPU 0); returns the seconds it took."""
    started = time.perf_counter()
    subprocess.run([str(executable), "-i", str(source), "-o", str(destination), "-n", model, "-s", str(scale),
                    "-m", str(executable.parent / "models"), "-g", "0"], check=True, capture_output=True)
    return time.perf_counter() - started


def to_size(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """`image` at the lab's output size."""
    if image.shape[:2] == (height, width):
        return image
    import cv2
    return cv2.resize(np.ascontiguousarray(image), (width, height), interpolation=cv2.INTER_AREA)


def make_low_clip(truth_path: Path, low_path: Path, render: tuple[int, int], frames: int, crf: int | None) -> None:
    """Writes the lab's input: the clip downscaled (area), lossless, through H.264 at `crf` first when given."""
    ffmpeg = [media.tool_path("ffmpeg"), "-v", "error", "-y", "-i"]
    downscale = ["-vf", f"scale={render[0]}:{render[1]}:flags=area", "-frames:v", str(frames)]
    lossless = ["-c:v", "ffv1", "-pix_fmt", "rgb24", str(low_path)]
    if crf is None:
        subprocess.run([*ffmpeg, str(truth_path), *downscale, *lossless], check=True)
        return
    compressed = low_path.with_suffix(".mp4")
    subprocess.run([*ffmpeg, str(truth_path), *downscale, "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
                    str(compressed)], check=True)
    subprocess.run([*ffmpeg, str(compressed), *lossless], check=True)


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


def image_lab(gpu: Gpu, sdk_dir: Path, reference_path: Path, out_dir: Path, scale: float,
              realesrgan: Path | None) -> list[str]:
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
    if has_amf_scaler():
        for name, algorithm, sharpness in AMF_SETTINGS:
            image = amf_upscale(low_path, width, height, algorithm, sharpness)[0]
            media.write_image(out_dir / f"{reference_path.stem}_amf_{algorithm}_{sharpness}.png", image)
            report(name, image)
    if realesrgan:
        for name, model, model_scale in REALESRGAN_SETTINGS:
            path = out_dir / f"{reference_path.stem}_{model}.png"
            realesrgan_upscale(realesrgan, low_path, path, model, model_scale)
            image = to_size(media.read_image(path), width, height)
            media.write_image(path, image)
            report(name, image)
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


def video_lab(gpu: Gpu, sdk_dir: Path, truth_path: Path, out_dir: Path, scale: float, frames: int, warmup: int,
              realesrgan: Path | None, crf: int | None) -> list[str]:
    truth_info = media.probe(truth_path)
    width, height = truth_info.width, truth_info.height
    render = (int(round(width / scale)) // 2 * 2, int(round(height / scale)) // 2 * 2)
    low_path = out_dir / f"{truth_path.stem}_low.mkv"
    make_low_clip(truth_path, low_path, render, frames, crf)
    low_info = media.probe(low_path)
    truth_frames = [frame.copy() for frame in media.FrameReader(truth_path, truth_info, max_frames=frames)]
    low_frames = [frame.copy() for frame in media.FrameReader(low_path, low_info, max_frames=frames)]
    input_note = "" if crf is None else f", through H.264 CRF {crf}"
    lines = [f"## Video: {truth_path.name} ({width}x{height}), render {render[0]}x{render[1]}{input_note}, "
             f"{len(low_frames)} frames, PSNR over frames {warmup + 1}..{len(low_frames)}", "",
             "| setting | mean PSNR dB | min PSNR dB | temporal PSNR dB | ms per frame |", "|---|---|---|---|---|"]

    def report(name: str, scores: FrameScores, ms: float) -> None:
        values = scores.psnr[warmup:]
        temporal = scores.temporal[max(warmup - 1, 0):]
        lines.append(f"| {name} | {np.mean(values):.2f} | {np.min(values):.2f} | {np.mean(temporal):.2f} | {ms:.1f} |")
        print(lines[-1], flush=True)

    lanczos = FrameScores(truth_frames)
    upscale_seconds = 0.0
    import cv2
    for low in low_frames:
        started = time.perf_counter()
        up = cv2.resize(low, (width, height), interpolation=cv2.INTER_LANCZOS4)
        upscale_seconds += time.perf_counter() - started
        lanczos.add(up)
    report("lanczos (opencv)", lanczos, upscale_seconds * 1000 / len(low_frames))

    if has_amf_scaler():
        for name, algorithm, sharpness in AMF_SETTINGS:
            upscaled = amf_upscale(low_path, width, height, algorithm, sharpness)
            if len(upscaled) != len(truth_frames):
                raise RuntimeError(f"{name}: {len(upscaled)} frames for {len(truth_frames)}")
            media.write_image(out_dir / f"{truth_path.stem}_amf_{algorithm}_{sharpness}_last.png", upscaled[-1])
            scores = FrameScores(truth_frames)
            for image in upscaled:
                scores.add(image)
            report(name, scores, amf_ms_per_frame(low_path, width, height, algorithm, sharpness, len(low_frames)))

    if realesrgan:
        with tempfile.TemporaryDirectory(dir=out_dir) as frames_dir:
            subprocess.run([media.tool_path("ffmpeg"), "-v", "error", "-i", str(low_path),
                            str(Path(frames_dir) / "%05d.png")], check=True)
            for name, model, model_scale in REALESRGAN_SETTINGS:
                with tempfile.TemporaryDirectory(dir=out_dir) as upscaled_dir:
                    seconds = realesrgan_upscale(realesrgan, Path(frames_dir), Path(upscaled_dir), model, model_scale)
                    paths = sorted(Path(upscaled_dir).glob("*.png"))
                    if len(paths) != len(truth_frames):
                        raise RuntimeError(f"{name}: {len(paths)} frames for {len(truth_frames)}")
                    scores = FrameScores(truth_frames)
                    for path in paths:
                        image = to_size(media.read_image(path), width, height)
                        scores.add(image)
                media.write_image(out_dir / f"{truth_path.stem}_{model}_last.png", image)
                report(name, scores, seconds * 1000 / len(paths))

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
        scores = FrameScores(truth_frames)
        upscale_seconds = 0.0
        try:
            for index, low in enumerate(low_frames):
                started = time.perf_counter()
                result = pipeline.process(low)
                upscale_seconds += time.perf_counter() - started
                scores.add(result)
                if index == len(low_frames) - 1:
                    tag = f"{flow_name}_{preset}_s{flow_scale}{'_neg' if flow_sign < 0 else ''}"
                    media.write_image(out_dir / f"{truth_path.stem}_{tag}_last.png", result)
                    media.write_image(out_dir / f"{truth_path.stem}_low_last.png", low)
        finally:
            pipeline.destroy()
        flow_ms = pipeline.timings.stages.get("flow", 0.0) / max(pipeline.timings.frames, 1)
        report(f"{name} (flow {flow_ms:.1f} ms)", scores, upscale_seconds * 1000 / len(low_frames))
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
    parser.add_argument("--realesrgan", type=Path, help="realesrgan-ncnn-vulkan.exe, to measure it too")
    parser.add_argument("--compress", type=int, metavar="CRF", help="put the small clip through H.264 at this CRF")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sdk_dir = find_sdk_dir(args.sdk)
    gpu = Gpu(verbose=False)
    lines = ["# Quality lab", ""]
    try:
        for image in args.image:
            lines += image_lab(gpu, sdk_dir, image, args.out, args.scale, args.realesrgan)
        for video in args.video:
            lines += video_lab(gpu, sdk_dir, video, args.out, args.scale, args.frames, args.warmup, args.realesrgan,
                               args.compress)
    finally:
        gpu.destroy()
    (args.out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"written {args.out / 'results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
