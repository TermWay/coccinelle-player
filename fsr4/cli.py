"""Command line: `fsr4 upscale <image or video>`, `fsr4 ui`, `fsr4 info`, `fsr4 selftest`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, log, usage
from .ffx import QUALITY_MODE_SCALES, find_sdk_dir
from .flow import FLOW_METHODS
from .hq_scaler import FIRST_FFMPEG, HQ_SCALER
from .realesrgan import REAL_ESRGAN, REAL_ESRGAN_ANIME, REAL_ESRGAN_X4PLUS
from .job import JobOptions, run_job
from .media import ENCODERS


UPSCALERS = {"4": None, "3.1": "3.1.6", "2.3": "2.3.4", "hq": HQ_SCALER, "realesrgan": REAL_ESRGAN,
             "realesrgan-x4plus": REAL_ESRGAN_X4PLUS, "realesrgan-x4plus-anime": REAL_ESRGAN_ANIME}

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
    upscale.add_argument("--sharpness", type=float, help="RCAS sharpening 0..1 (default: off)")
    upscale.add_argument("--codec", choices=sorted(ENCODERS), default="amf",
                         help="video encoder: amf (HEVC on the GPU), av1_amf, h264_amf, x264, x265 (default: amf)")
    upscale.add_argument("--upscaler", choices=sorted(UPSCALERS), default="4",
                         help="4 (FSR 4, default), 3.1 or 2.3 (from the same DLL, to compare), hq (AMD's HQ "
                              f"scaler, through ffmpeg {FIRST_FFMPEG} or later; it takes no motion vectors), or "
                              "realesrgan (slow, redraws the picture; installed apart, in bin/realesrgan or on PATH), "
                              "with realesrgan-x4plus (general) and realesrgan-x4plus-anime (drawings) for still images")
    upscale.add_argument("--frames", type=int, help="stop after this many frames (for tests)")
    upscale.add_argument("--debug-view", action="store_true", help="draw FSR's debug overlay on the output")
    upscale.add_argument("--debug-checking", action="store_true", help="enable FSR's input checker messages")
    upscale.add_argument("--validate", action="store_true", help="enable the Vulkan validation layer")
    upscale.add_argument("--gpu", type=int, help="Vulkan device index (default: the discrete AMD GPU)")
    upscale.add_argument("--sdk", help="folder holding the FSR DLLs (default: bin/)")
    upscale.add_argument("-q", "--quiet", action="store_true", help="no progress output")

    ui = commands.add_parser("ui", help="open the window")
    ui.add_argument("input", type=Path, nargs="?", help="file to preselect")

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
    options = JobOptions(
        input=args.input, output=args.output, mode=args.mode, scale=args.scale, width=args.width,
        height=args.height, flow=args.flow, flow_preset=args.flow_preset, flow_scale=args.flow_scale,
        sharpness=args.sharpness, codec=args.codec, upscaler=UPSCALERS[args.upscaler], frames=args.frames,
        debug_view=args.debug_view,
        debug_checking=args.debug_checking, validate=args.validate, gpu=args.gpu, sdk=args.sdk,
        verbose=not args.quiet)

    line_open = False

    def progress(frames_done: int, total: int | None, fps: float) -> None:
        nonlocal line_open
        if args.quiet:
            return
        remaining = f", {((total - frames_done) / fps):.0f} s left" if total and fps else ""
        print(f"\r[fsr4] frame {frames_done}{f'/{total}' if total else ''}, {fps:.1f} fps{remaining}   ",
              end="", file=sys.stderr, flush=True)
        line_open = True

    def close_line() -> None:
        """End the progress line, so that what follows starts a line of its own."""
        nonlocal line_open
        if line_open:
            print("", file=sys.stderr, flush=True)
            line_open = False

    def emit(line: str) -> None:
        close_line()
        print(line, flush=True)

    # The whole processor at normal priority, the optical flow capped at the threads it gains from.
    usage.apply(usage.CPU_THREADS, "normal")
    log.set_sink(emit)
    try:
        result = run_job(options, progress=progress)
    except (ValueError, RuntimeError, OSError) as error:
        close_line()
        print(f"fsr4: {error}", file=sys.stderr)
        return 2 if isinstance(error, ValueError) else 1
    finally:
        log.set_sink(None)
        close_line()
    return 0 if result.frames else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ui":
        from .ui import run_ui
        return run_ui(args.input)
    if args.command == "upscale":
        return run_upscale(args)
    try:
        if args.command == "selftest":
            from .selftest import run
            return run(validate=args.validate)
        return run_info(args)
    except (RuntimeError, OSError) as error:  # no Vulkan driver, no GPU, no FSR DLLs: said in one line
        print(f"fsr4: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
