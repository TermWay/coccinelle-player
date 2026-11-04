"""Measure what the window really shows while its layout changes: folding the sidebar, a repaint, a resize.

    python tools/ui_flicker.py --video clip.mkv [--frames out/ui/flicker] [--runs 3]

The window opens the clip (a short one, so that it is done upscaling) and a thread copies its area of the screen once
per composed frame (DwmFlush), so a blink of one frame is seen, which PrintWindow never shows. For each action:

- frames that differ from the one before, when they start and how long they last: a clean change is one step, or
  the fade of the window's curtain (the square curtain over the window's rounded corners is the first change seen);
- stray pixels: those in neither the picture before nor the one after, in the worst frame, and how many frames have
  any (a widget in the wrong place, or erased, or a background showing through);
- blinks: widgets that are the same before and after but changed in between;
- for a resize, whether what follows the right edge (the Right dropdown, Upscale) stays whole and how far it lags.

The window stays on top while it is measured and the run refuses when another program's window overlaps it, or when
someone folds the sidebar: leave the window alone. Frames stay in memory; --frames saves the worst one of each
action, cut to the window (fold and repaint only). "repainted" repaints everything after a fold: any change there is
something the fold left unpainted.
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import window_capture  # noqa: E402
from fsr4 import ui, ui_theme, usage  # noqa: E402
from fsr4.presets import PresetStore  # noqa: E402
from fsr4.ui_art import png_bytes  # noqa: E402

POLL_MS = 15
WAIT_SECONDS = 120
CAPACITY = 700
TOLERANCE = 10  # of 255: smaller differences are not visible, and dithered edges produce them
BLINK_SHARE = 0.02  # of a widget's pixels
STRAY_SHARE = 0.0005  # of the window's pixels, for a frame to count
BEFORE_SECONDS = 0.15
AFTER_SECONDS = 0.7
RESIZE_PIXELS = 240
RESIZE_STEPS = 30
RESIZE_STEP_SECONDS = 0.016
BROKEN_DIFFERENCE = 12.0  # mean difference, of 255, from the widget at rest
SRCCOPY, CAPTUREBLT = 0x00CC0020, 0x40000000
dwmapi = ctypes.windll.dwmapi
dwmapi.DwmFlush.restype = ctypes.HRESULT
dwmapi.DwmFlush.argtypes = []


class Sampler:
    """Copies an area of the screen once per composed frame, on its own thread, into a preallocated buffer."""

    def __init__(self, handle: int, width: int, height: int):
        self.handle, self.width, self.height = handle, width, height
        self.frames = np.empty((CAPACITY, height, width, 4), np.uint8)
        self.times = np.zeros(CAPACITY)
        self.rights = np.zeros(CAPACITY, np.int32)
        self.count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.count = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        bounds = window_capture.frame_bounds(self.handle)
        left, top = bounds.left, bounds.top
        screen_dc = window_capture.user32.GetWindowDC(None)
        memory_dc = window_capture.gdi32.CreateCompatibleDC(screen_dc)
        bitmap = window_capture.gdi32.CreateCompatibleBitmap(screen_dc, self.width, self.height)
        previous = window_capture.gdi32.SelectObject(memory_dc, bitmap)
        header = window_capture.BITMAPINFOHEADER(biSize=ctypes.sizeof(window_capture.BITMAPINFOHEADER),
                                                 biWidth=self.width, biHeight=-self.height, biPlanes=1, biBitCount=32,
                                                 biCompression=0)
        try:
            while not self._stop.is_set() and self.count < CAPACITY:
                dwmapi.DwmFlush()
                # CAPTUREBLT: without it a copy of the screen leaves out layered windows, as the window's curtain.
                if not window_capture.gdi32.BitBlt(memory_dc, 0, 0, self.width, self.height, screen_dc, left, top,
                                                   SRCCOPY | CAPTUREBLT):
                    raise RuntimeError("BitBlt from the screen failed")
                target = self.frames[self.count]
                if window_capture.gdi32.GetDIBits(memory_dc, bitmap, 0, self.height, target.ctypes.data,
                                                  ctypes.byref(header), 0) != self.height:
                    raise RuntimeError("GetDIBits failed")
                self.times[self.count] = time.perf_counter()
                self.rights[self.count] = window_capture.frame_bounds(self.handle).right - left
                self.count += 1
        finally:
            window_capture.gdi32.SelectObject(memory_dc, previous)
            window_capture.gdi32.DeleteObject(bitmap)
            window_capture.gdi32.DeleteDC(memory_dc)
            window_capture.user32.ReleaseDC(None, screen_dc)


def describe(widget: tk.Misc) -> str:
    """A widget's class and text, to name it in the results."""
    try:
        text = str(widget.cget("text"))
    except tk.TclError:  # no text option on this class
        text = ""
    return f"{widget.winfo_class()} {text[:24]!r}" if text else widget.winfo_class()


def leaf_widgets(root: tk.Misc, left: int, top: int) -> dict[str, tuple[int, int, int, int, str]]:
    """Every mapped widget without mapped children: its rectangle in the capture and its description."""
    found = {}
    pending = [root]
    while pending:
        widget = pending.pop()
        children = [child for child in widget.winfo_children()
                    if child.winfo_ismapped() and not isinstance(child, tk.Toplevel)]
        if not children and widget is not root and widget.winfo_width() > 1:
            found[str(widget)] = (widget.winfo_rootx() - left, widget.winfo_rooty() - top, widget.winfo_width(),
                                  widget.winfo_height(), describe(widget))
        pending.extend(children)
    return found


def crop(frame: np.ndarray, rectangle: tuple) -> np.ndarray:
    x, y, width, height = rectangle[:4]
    return frame[y:y + height, x:x + width, :3]


def visibly_different(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (np.abs(first.astype(np.int16) - second.astype(np.int16)) > TOLERANCE).any(axis=2)


def static_widgets(before_widgets: dict, after_widgets: dict, before: np.ndarray, after: np.ndarray) -> dict:
    """Those that do not move and look the same at both ends: any change in between is a blink."""
    return {path: rectangle for path, rectangle in before_widgets.items()
            if after_widgets.get(path) == rectangle and not visibly_different(crop(before, rectangle),
                                                                              crop(after, rectangle)).any()}


def score(name: str, sampler: Sampler, action_time: float, before_widgets: dict, after_widgets: dict,
          frames_folder: Path | None) -> None:
    """Print one action's numbers."""
    count = sampler.count
    frames, times = sampler.frames[:count], sampler.times[:count]
    start = int(np.searchsorted(times, action_time))
    if start == 0 or start >= count:
        raise RuntimeError(f"{name}: {count} frames sampled, the action at frame {start}")
    before, after = frames[start - 1], frames[-1]
    low = np.minimum(before[..., :3], after[..., :3]).astype(np.int16) - TOLERANCE
    high = np.maximum(before[..., :3], after[..., :3]).astype(np.int16) + TOLERANCE
    changes = []
    stray_shares = []
    for index in range(start, count):
        if (frames[index, ..., :3] != frames[index - 1, ..., :3]).any():
            changes.append(index)
        picture = frames[index, ..., :3].astype(np.int16)
        stray_shares.append(float(((picture < low) | (picture > high)).any(axis=2).mean()))
    stray = np.array(stray_shares)
    worst = int(stray.argmax()) + start
    static = static_widgets(before_widgets, after_widgets, before, after)
    blinks = []
    for rectangle in static.values():
        reference = crop(before, rectangle)
        hits = [float(visibly_different(crop(frames[index], rectangle), reference).mean())
                for index in range(start, count)]
        blinked = sum(hit > BLINK_SHARE for hit in hits)
        if blinked:
            blinks.append((blinked, max(hits), rectangle[4]))
    blinks.sort(reverse=True)
    duration = (times[changes[-1]] - times[changes[0]]) * 1000 if changes else 0.0
    first = (times[changes[0]] - action_time) * 1000 if changes else 0.0
    print(f"{name:10} {len(changes):3} changing frames over {duration:5.0f} ms from {first:3.0f} ms, stray pixels in "
          f"{int((stray > STRAY_SHARE).sum()):3} frames (worst {stray.max() * 100:5.2f}% at "
          f"{(times[worst] - action_time) * 1000:4.0f} ms), blinks in {len(blinks):3} of {len(static)} still widgets, "
          f"{count / (times[-1] - times[0]):.0f} fps", flush=True)
    for blinked, share, label in blinks[:6]:
        print(f"           {blinked:3} frames, up to {share * 100:3.0f}%: {label}", flush=True)
    if frames_folder is not None:
        frames_folder.mkdir(parents=True, exist_ok=True)
        rgba = frames[worst][..., [2, 1, 0, 3]].copy()
        rgba[..., 3] = 255
        (frames_folder / f"{name}.png").write_bytes(png_bytes(rgba))


def score_resize(name: str, sampler: Sampler, action_time: float, followers: dict[str, tuple],
                 before_widgets: dict, after_widgets: dict) -> None:
    """Print how what follows the right edge holds up, and the blinks of what stays at the left."""
    count = sampler.count
    frames, times, rights = sampler.frames[:count], sampler.times[:count], sampler.rights[:count]
    start = int(np.searchsorted(times, action_time))
    before = frames[start - 1]
    print(f"{name:10} {count - start} frames, the right edge moved in "
          f"{int((np.diff(rights[start - 1:]) != 0).sum())} of them, "
          f"{count / (times[-1] - times[0]):.0f} fps", flush=True)
    for label, rectangle in followers.items():
        x, y, width, height = rectangle
        template = cv2.cvtColor(np.ascontiguousarray(before[y:y + height, x:x + width, :3]), cv2.COLOR_BGR2GRAY)
        broken = 0
        lags = []
        for index in range(start, count):
            band = cv2.cvtColor(np.ascontiguousarray(frames[index, y:y + height, :rights[index], :3]),
                                cv2.COLOR_BGR2GRAY)
            difference = cv2.matchTemplate(band, template, cv2.TM_SQDIFF)[0]
            found = int(difference.argmin())
            if (difference[found] / template.size) ** 0.5 > BROKEN_DIFFERENCE:
                broken += 1
            else:
                lags.append(x + int(rights[index] - rights[start - 1]) - found)
        lag = np.abs(np.array(lags or [0]))
        print(f"           {label}: not whole in {broken} frames, away from its place in "
              f"{int((lag > 0).sum())} frames (mean {lag.mean():.1f} px, worst {lag.max()} px)", flush=True)
    static = {path: rectangle for path, rectangle in before_widgets.items()
              if after_widgets.get(path) == rectangle and rectangle[0] + rectangle[2] < rights.min()
              and not visibly_different(crop(before, rectangle), crop(frames[-1], rectangle)).any()}
    blinks = []
    for rectangle in static.values():
        reference = crop(before, rectangle)
        blinked = sum(float(visibly_different(crop(frames[index], rectangle), reference).mean()) > BLINK_SHARE
                      for index in range(start, count))
        if blinked:
            blinks.append((blinked, rectangle[4]))
    blinks.sort(reverse=True)
    print(f"           blinks in {len(blinks)} of {len(static)} widgets that stay in place", flush=True)
    for blinked, label in blinks[:6]:
        print(f"           {blinked:3} frames: {label}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True, help="a short clip")
    parser.add_argument("--frames", type=Path, help="folder for the worst frame of each action")
    parser.add_argument("--runs", type=int, default=3, help="how many times each action runs")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    ui_theme.enable_dark_dialogs()
    usage.apply(usage.DEFAULT.threads, usage.DEFAULT.priority)
    root = tk.Tk()
    root.withdraw()
    app = ui.App(root)
    presets_file = args.video.resolve().parent / "ui_flicker_presets.json"
    presets_file.unlink(missing_ok=True)
    app.presets = PresetStore(presets_file)
    app._update_preset_lists()
    ui_theme.style_title_bar(root)
    root.deiconify()
    root.attributes("-topmost", True)
    failures: list[str] = []

    def right_ready() -> bool:
        still = app.view._still
        return still is not None and still[1].get(app.view.sides["right"]) is not None

    def steps():
        yield lambda: app.gpu_ready is not None
        if app.gpu_ready is not True:
            raise RuntimeError(f"no GPU: {app.gpu_label.cget('text')}")
        if not app.load(args.video):
            raise RuntimeError(f"the window would not open {args.video}")
        yield lambda: app.info is not None and app.renderer is not None
        yield lambda: app.renderer.timeline.complete() and app.view.comparing and right_ready()
        yield 3.0
        handle = int(root.wm_frame(), 16)
        covering = window_capture.covering_window(handle)
        if covering is not None:
            raise RuntimeError(f"a window of another program ({covering}) overlaps the window; not measuring")
        bounds = window_capture.frame_bounds(handle)
        left, top = bounds.left, bounds.top
        sampler = Sampler(handle, bounds.right - bounds.left, bounds.bottom - bounds.top)

        def measured(name: str, run, sidebar_after: bool):
            before_widgets = leaf_widgets(root, left, top)
            sampler.start()
            yield BEFORE_SECONDS
            action_time = time.perf_counter()
            run()
            yield AFTER_SECONDS
            sampler.stop()
            if window_capture.covering_window(handle) is not None:
                raise RuntimeError("another program's window came over the window during the measure")
            if app.sidebar_visible != sidebar_after:
                raise RuntimeError("someone folded the sidebar during the measure: leave the window alone")
            score(name, sampler, action_time, before_widgets, leaf_widgets(root, left, top), args.frames)

        for run_index in range(args.runs):
            yield from measured("idle", lambda: None, True)
            yield from measured("repaint", app._repaint_everything, True)
            yield from measured("fold", app.toggle_sidebar, False)
            yield 0.4
            yield from measured("repainted", app._repaint_everything, False)
            yield from measured("unfold", app.toggle_sidebar, True)
            yield 0.4
            yield from measured("repainted", app._repaint_everything, True)
        for run_index in range(args.runs):
            width, height = root.winfo_width(), root.winfo_height()
            before_widgets = leaf_widgets(root, left, top)

            def rectangle(widget: tk.Misc) -> tuple[int, int, int, int]:
                return (widget.winfo_rootx() - left, widget.winfo_rooty() - top, widget.winfo_width(),
                        widget.winfo_height())

            followers = {"Right dropdown": rectangle(app.side_boxes["right"]), "Upscale": rectangle(app.start_button)}
            sampler.start()
            yield BEFORE_SECONDS
            action_time = time.perf_counter()
            for direction in (-1, 1):
                for step in range(1, RESIZE_STEPS + 1):
                    moved = RESIZE_PIXELS * step // RESIZE_STEPS
                    root.geometry(f"{width - moved if direction < 0 else width - RESIZE_PIXELS + moved}x{height}")
                    yield RESIZE_STEP_SECONDS
            yield AFTER_SECONDS
            sampler.stop()
            if window_capture.covering_window(handle) is not None:
                raise RuntimeError("another program's window came over the window during the measure")
            score_resize("resize", sampler, action_time, followers, before_widgets, leaf_widgets(root, left, top))

    sequence = steps()
    state: dict = {"condition": None, "deadline": 0.0}

    def stop(message: str | None) -> None:
        if message is not None:
            failures.append(message)
            print(f"FAIL  {message}", flush=True)
        app.close()

    def advance() -> None:
        try:
            condition = state["condition"]
            if condition is not None:
                if not condition():
                    if time.monotonic() > state["deadline"]:
                        stop("timed out waiting for the window")
                        return
                    root.after(POLL_MS, advance)
                    return
                state["condition"] = None
            step = next(sequence)
        except StopIteration:
            stop(None)
            return
        except Exception as error:  # noqa: BLE001 - report and stop
            stop(f"{error!r}")
            return
        if callable(step):
            state["condition"] = step
            state["deadline"] = time.monotonic() + WAIT_SECONDS
            root.after(POLL_MS, advance)
        else:
            root.after(int(step * 1000), advance)

    root.after(500, advance)
    root.mainloop()
    presets_file.unlink(missing_ok=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
