"""Record the window comparing the source enlarged with bicubic against FSR 4 at 3x, for the README.

    python tools/ui_record.py --video first.mkv --video second.mkv --out out/demo/ui_demo.mkv

Each clip is opened in the real window and upscaled whole by the preview, then shown frame by frame (paused
stills, so no frame is dropped or late) while the divider sweeps from the right edge to the left one and back:
clip changes and the loop have no jump. Every frame is captured with PrintWindow, which sees only this
process's window, and piped to ffmpeg as lossless FFV1 at --rate. The window pops up on the desktop while it
records; presets go to a scratch file next to the recording, never the user's.
"""
from __future__ import annotations

import argparse
import ctypes
import math
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import window_capture  # noqa: E402
from fsr4 import ui, ui_theme, usage  # noqa: E402
from fsr4.preferences import PreferenceStore  # noqa: E402
from fsr4.presets import PresetStore  # noqa: E402

POLL_MS = 15
WAIT_SECONDS = 180


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, action="append", required=True, help="a clip; repeat for more")
    parser.add_argument("--out", type=Path, required=True, help="the recording (.mkv, lossless FFV1)")
    parser.add_argument("--rate", type=float, default=30.0, help="frames per second of the recording")
    parser.add_argument("--skip", type=float, default=1.0,
                        help="seconds left out at each clip's start, while FSR's history fills up")
    parser.add_argument("--size", default="", help="window size as WIDTHxHEIGHT (default: the app's own)")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")
    for path in args.video:
        if not path.is_file():
            parser.error(f"not found: {path}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    ui_theme.enable_dark_dialogs()
    usage.apply(usage.DEFAULT.threads, usage.DEFAULT.priority)
    root = tk.Tk()
    root.withdraw()
    app = ui.App(root, preferences=PreferenceStore(args.out.with_name(args.out.stem + "_preferences.json")))
    presets_file = args.out.with_name(args.out.stem + "_presets.json")
    presets_file.unlink(missing_ok=True)
    app.presets = PresetStore(presets_file)
    app._update_preset_lists()
    ui_theme.style_title_bar(root)
    if args.size:
        root.geometry(args.size)
    root.deiconify()
    app.size_var.set("3")
    app._settings_changed()
    started = time.monotonic()
    failures: list[str] = []
    encoder: dict[str, subprocess.Popen | tuple] = {}

    def say(message: str) -> None:
        print(f"{time.monotonic() - started:6.1f}s {message}", flush=True)

    def right_ready() -> bool:
        still = app.view._still
        return still is not None and still[1].get(app.view.sides["right"]) is not None

    def write(image) -> None:
        height, width = image.shape[:2]
        if "process" not in encoder:
            encoder["size"] = (width, height)
            encoder["process"] = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                 "-r", f"{args.rate:g}", "-i", "-", "-c:v", "ffv1", str(args.out)], stdin=subprocess.PIPE)
        if encoder["size"] != (width, height):
            raise RuntimeError(f"the window changed size from {encoder['size']} to {(width, height)}")
        encoder["process"].stdin.write(image[..., :3].tobytes())

    def steps():
        yield lambda: app.gpu_ready is not None
        if app.gpu_ready is not True:
            raise RuntimeError(f"no GPU: {app.gpu_label.cget('text')}")
        for path in args.video:
            if not app.load(path):
                raise RuntimeError(f"the window would not open {path}: {app.banner_label.cget('text')}")
            yield lambda: app.info is not None and app.player is not None and app.renderer is not None
            say(f"{path.name} opened, upscaling it")
            yield lambda: app.renderer.timeline.complete() and app.view.comparing and right_ready()
            source_rate = float(app.info.frame_rate)
            total = app.renderer.timeline.total
            skip = int(round(args.skip * source_rate))
            usable = total - skip
            if usable <= 0:
                raise RuntimeError(f"{path.name} has {total} frames, none after the {skip} skipped")
            count = int(usable / source_rate * args.rate)
            say(f"{path.name}: {total} frames upscaled, recording {count}")
            shown = None
            for index in range(count):
                frame = skip + min(usable - 1, int(index / args.rate * source_rate))
                if frame != shown:
                    before = app.view._still
                    app.seek_to(frame, True)
                    yield lambda: (app.player.position == frame and app.transport.position == frame
                                   and app.view._still is not before and right_ready() and not app.view.spinner)
                    shown = frame
                with app.view._lock:
                    # from the right edge (all bicubic) to the left one (all FSR 4) and back
                    app.view.split = 0.5 + 0.5 * math.cos(2 * math.pi * index / count)
                app.view.redraw()
                write(window_capture.capture(root))
        say("recorded")

    sequence = steps()
    deadline = {"time": 0.0, "condition": None}

    def stop(message: str | None) -> None:
        if message is not None:
            failures.append(message)
            say(f"FAIL  {message}")
        process = encoder.get("process")
        if process is not None:
            process.stdin.close()
            if process.wait() != 0:
                failures.append("ffmpeg could not write the recording")
        app.close()

    def advance() -> None:
        try:
            condition = deadline["condition"]
            if condition is not None:
                if not condition():
                    if time.monotonic() > deadline["time"]:
                        stop("timed out waiting for the window")
                        return
                    root.after(POLL_MS, advance)
                    return
                deadline["condition"] = None
            step = next(sequence)
        except StopIteration:
            stop(None)
            return
        except Exception as error:  # noqa: BLE001 - report and stop
            stop(f"{error!r}")
            return
        deadline["condition"] = step
        deadline["time"] = time.monotonic() + WAIT_SECONDS
        root.after(POLL_MS, advance)

    root.after(500, advance)
    root.mainloop()
    presets_file.unlink(missing_ok=True)
    if not failures:
        say(f"wrote {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
