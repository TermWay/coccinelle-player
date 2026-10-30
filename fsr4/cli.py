"""Command line: `fsr4 upscale <image or video>`, `fsr4 info`, `fsr4 selftest`."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .ffx import QUALITY_MODE_SCALES, find_sdk_dir
from .flow import FLOW_METHODS, create_flow
from .media import ENCODERS, FrameReader, FrameWriter, is_image, probe, read_image, write_image
from .pipeline import JITTER_MODES, Pipeline

VIDEO_SUFFIX = ".mp4"


def even(value: float) -> int:
    """Round to the nearest even integer (video encoders need even dimensions)."""
    return max(2, int(round(value / 2)) * 2)


def output_size(width: int, height: int, args) -> tuple[int, int]:
    if args.width or args.height:
        if args.width and args.height:
            return even(args.width), even(args.height)
        if args.width:
            return even(args.width), even(args.width * height / width)
        return even(args.height * width / height), even(args.height)
    scale = args.scale if args.scale else QUALITY_MODE_SCALES[args.mode]
    return even(width * scale), even(height * scale)


def default_output(path: Path) -> Path:
    suffix = path.suffix if is_image(path) else VIDEO_SUFFIX
    return path.with_name(f"{path.stem}.fsr4{suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fsr4", description="Upscale an image or a video with AMD FSR 4.")
    parser.add_argument("--version", action="version", version=f"fsr4 {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    upscale = commands.add_parser("upscale", help="upscale an image or video file")
    upscale.add_argument("input", type=Path, help="image (png, jpg, ...) or video (mp4, mkv, ...)")
    upscale.add_argument("-o", "--output", type=Path, help="output path (default: <input>.fsr4.<ext>)")
    upscale.add_argument("--mode", choices=sorted(QUALITY_MODE_SCALES, key=QUALITY_MODE_SCALES.get),
                         default="performance", help="FSR quality mode, sets the scale (default: performance = 2x)")
    upscale.add_argument("--scale", type=float, help="per-dimension scale factor, overrides --mode (1.0 to 3.0)")
    upscale.add_argument("--width", type=int, help="output width (overrides --scale)")
    upscale.add_argument("--height", type=int, help="output height (overrides --scale)")
    upscale.add_argument("--flow", choices=FLOW_METHODS, default="dis",
                         help="motion vectors: optical flow (dis, needs opencv) or none (zero)")
    upscale.add_argument("--flow-preset", choices=("ultrafast", "fast", "medium"), default="fast",
                         help="DIS optical flow preset (default: fast)")
    upscale.add_argument("--flow-scale", type=int, default=1,
                         help="compute optical flow at 1/N resolution (default: 1)")
    upscale.add_argument("--jitter", choices=JITTER_MODES, default="fake",
                         help="fake: resample the input on FSR's jitter sequence (needs opencv); none: no jitter")
    upscale.add_argument("--passes", type=int, default=8,
                         help="for images: feed the frame this many times so FSR accumulates (default: 8)")
    upscale.add_argument("--sharpness", type=float, help="RCAS sharpening 0..1 (default: off)")
    upscale.add_argument("--codec", choices=sorted(ENCODERS), default="amf",
                         help="video encoder: amf (HEVC on the GPU), h264_amf, x264, x265 (default: amf)")
    upscale.add_argument("--frames", type=int, help="stop after this many frames (for tests)")
    upscale.add_argument("--debug-view", action="store_true", help="draw FSR's debug overlay on the output")
    upscale.add_argument("--debug-checking", action="store_true", help="enable FSR's input checker messages")
    upscale.add_argument("--validate", action="store_true", help="enable the Vulkan validation layer")
    upscale.add_argument("--gpu", type=int, help="Vulkan device index (default: the discrete AMD GPU)")
    upscale.add_argument("--sdk", help="folder holding the FSR DLLs (default: bin/)")
    upscale.add_argument("--jitter-sign", type=float, default=-1.0, help=argparse.SUPPRESS)
    upscale.add_argument("-q", "--quiet", action="store_true", help="no progress output")

    info = commands.add_parser("info", help="list Vulkan devices and the FSR versions available")
    info.add_argument("--sdk", help="folder holding the FSR DLLs (default: bin/)")
    info.add_argument("--gpu", type=int)

    selftest = commands.add_parser("selftest", help="GPU round trip test without FSR")
    selftest.add_argument("--validate", action="store_true")
    return parser


def run_info(args) -> int:
    from . import vk_types as T
    from .gpu import Gpu
    gpu = Gpu(gpu_index=args.gpu, verbose=True)
    try:
        for index, (_, properties) in enumerate(gpu.physical_devices()):
            kind = {T.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: "discrete", T.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: "integrated"}
            print(f"  gpu {index}: {properties.deviceName.decode('utf-8', 'replace')} "
                  f"({kind.get(properties.deviceType, 'other')}, vendor {properties.vendorID:#06x})")
        from .ffx import Upscaler, FFX_UPSCALE_ENABLE_AUTO_EXPOSURE
        upscaler = Upscaler(gpu, (1920, 1080), (3840, 2160), FFX_UPSCALE_ENABLE_AUTO_EXPOSURE, find_sdk_dir(args.sdk),
                            verbose=True)
        upscaler.destroy()
    finally:
        gpu.destroy()
    return 0


def run_upscale(args) -> int:
    from .gpu import Gpu
    input_path: Path = args.input
    if not input_path.is_file():
        print(f"input not found: {input_path}", file=sys.stderr)
        return 2
    sdk_dir = find_sdk_dir(args.sdk)
    output_path = args.output or default_output(input_path)
    verbose = not args.quiet
    image_mode = is_image(input_path)
    info = probe(input_path)
    width, height = info.width, info.height
    out_width, out_height = output_size(width, height, args)
    scale = out_width / width
    if not 1.0 <= scale <= 3.0 + 1e-6 or not 1.0 <= out_height / height <= 3.0 + 1e-6:
        print(f"FSR supports scales from 1x to 3x; {width}x{height} -> {out_width}x{out_height} is {scale:.2f}x",
              file=sys.stderr)
        return 2
    if verbose:
        print(f"[fsr4] {input_path.name}: {width}x{height} -> {out_width}x{out_height} ({scale:.2f}x), "
              f"{'image' if image_mode else f'{info.codec} video, {float(info.frame_rate):.3f} fps'}")

    gpu = Gpu(gpu_index=args.gpu, validate=args.validate, verbose=verbose)
    pipeline = None
    try:
        flow = create_flow(args.flow if not image_mode else "zero", width, height, args.flow_preset, args.flow_scale)
        pipeline = Pipeline(gpu, (width, height), (out_width, out_height), sdk_dir, flow, jitter_mode=args.jitter,
                            sharpness=args.sharpness, frame_time_ms=info.frame_time_ms, debug_view=args.debug_view,
                            debug_checking=args.debug_checking, verbose=verbose, jitter_sign=args.jitter_sign)
        started = time.perf_counter()
        if image_mode:
            frame = read_image(input_path)
            passes = max(1, args.passes)
            for _ in range(passes):
                result = pipeline.process(frame)
            write_image(output_path, result)
            frames_done = passes
        else:
            reader = FrameReader(input_path, info, max_frames=args.frames)
            writer = FrameWriter(output_path, out_width, out_height, info.frame_rate, args.codec,
                                 audio_source=input_path if info.has_audio else None)
            frames_done = 0
            total = info.frame_count if args.frames is None else min(args.frames, info.frame_count or args.frames)
            last_report = started
            try:
                for frame in reader:
                    writer.write(pipeline.process(frame))
                    frames_done += 1
                    now = time.perf_counter()
                    if verbose and now - last_report >= 1.0:
                        fps = frames_done / (now - started)
                        remaining = f", {((total - frames_done) / fps):.0f} s left" if total and fps else ""
                        print(f"\r[fsr4] frame {frames_done}{f'/{total}' if total else ''}, {fps:.1f} fps{remaining}   ",
                              end="", file=sys.stderr, flush=True)
                        last_report = now
            finally:
                writer.close()
            if verbose:
                print("", file=sys.stderr)
        elapsed = time.perf_counter() - started
        if verbose:
            print(f"[fsr4] wrote {output_path} ({frames_done} frames in {elapsed:.1f} s, {frames_done / elapsed:.1f} fps)")
            print(f"[fsr4] {pipeline.timings.summary()}")
        return 0
    finally:
        if pipeline is not None:
            pipeline.destroy()
        gpu.destroy()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "selftest":
        from .selftest import run
        return run(validate=args.validate)
    if args.command == "info":
        return run_info(args)
    return run_upscale(args)


if __name__ == "__main__":
    sys.exit(main())
