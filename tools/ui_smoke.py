"""Drive the window through every state and screenshot each one; exit 0 when every step worked.

    python tools/ui_smoke.py --video clip.wmv --image picture.png --out out/ui

Steps: empty window, a video dropped on it (a real WM_DROPFILES message), its first upscaled frame, a tooltip
(gone once the window is minimized), a dropdown, the file dialog, playback, a seek into frames not upscaled yet
(buffering), a paused zoom with the original pixels on the left (the handle dragged to the pointer's height), the
zoom slider (a drag to 600% and back must keep the picture still around the middle) and wheel, a side's dropdown opened from its label, a preset saved and two presets upscaled side by side
(paused and playing), a frame step, playing backwards at a quarter speed, a side turned off, full screen on the
window's own monitor (the second one when there is one) with its controls showing near the bottom and hiding at
rest, then back where the window was, the folded sidebar, the whole-file upscale and save, AMD's HQ scaler beside
FSR 4 and through a whole-file upscale, Real-ESRGAN beside FSR 4 (a whole-file upscale stopped midway, and the note
when it is not installed), an image compared with a preset, saved, then cancelled (nothing written), the error
banner for a bad file, a video opened with the upscale-on-open switch off (its pieces beside the output),
and the minimum window size. Presets and preferences go to scratch files in `--out`, not the user's. Screenshots
go through PrintWindow, so the window may be covered; it does pop up on the desktop and must not be minimized.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import threading
import time
import tkinter as tk
from ctypes import wintypes
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import window_capture  # noqa: E402
import fsr4  # noqa: E402
from fsr4 import media, ui, ui_theme, ui_window, usage  # noqa: E402
from fsr4.preferences import PreferenceStore  # noqa: E402
from fsr4.presets import BEST, PresetStore  # noqa: E402
from fsr4.realesrgan import REAL_ESRGAN  # noqa: E402
from fsr4.ui_slider import KNOB_INSET  # noqa: E402
from fsr4.ui_transport import VOLUME_KEY_DECIBELS, speed_fraction, volume_fraction  # noqa: E402

WM_DROPFILES = 0x0233
WM_CLOSE = 0x0010
GMEM_MOVEABLE_ZEROINIT = 0x0042
DIALOG_CLASS = "#32770"
DIALOG_SECONDS = 30

kernel32 = ctypes.windll.kernel32
kernel32.GlobalAlloc.restype = ctypes.c_void_p
kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
kernel32.GlobalLock.restype = ctypes.c_void_p
kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
ctypes.windll.user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]


def own_windows() -> list[tuple[int, str, str]]:
    """This process's top-level windows as (handle, title, class name). Never look a window up by its title
    alone: the user may have the app open with the same dialog, and it would be captured and closed."""
    user32 = window_capture.user32
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    windows = []

    def visit(handle, _parameter):
        if window_capture.window_process(handle) == os.getpid():
            title, kind = ctypes.create_unicode_buffer(256), ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(handle, title, 256)
            user32.GetClassNameW(handle, kind, 256)
            windows.append((handle, title.value, kind.value))
        return True

    user32.EnumWindows(window_capture.WNDENUMPROC(visit), 0)
    return windows


def close_dialog(title: str, screenshot: Path, found: list, done: threading.Event) -> threading.Thread:
    """Wait for this process's native dialog with `title`, screenshot it, then close it: it blocks the Tk thread
    until closed. Past `DIALOG_SECONDS` without it, any dialog of this process is closed until `done` is set, so
    the run fails instead of hanging."""
    user32 = ctypes.windll.user32

    def watch():
        deadline = time.monotonic() + DIALOG_SECONDS
        while not done.is_set():
            windows = own_windows()
            handle = next((handle for handle, text, _ in windows if text == title), None)
            if handle and not found:
                time.sleep(1.0)
                try:
                    window_capture.save_handle(handle, screenshot)
                    found.append(handle)
                finally:
                    user32.PostMessageW(handle, WM_CLOSE, 0, 0)
            elif time.monotonic() > deadline:
                for other, _, kind in windows:
                    if kind == DIALOG_CLASS:
                        user32.PostMessageW(other, WM_CLOSE, 0, 0)
            time.sleep(0.1)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    return thread


class DROPFILES(ctypes.Structure):
    _fields_ = [("pFiles", wintypes.DWORD), ("pt", wintypes.POINT), ("fNC", wintypes.BOOL), ("fWide", wintypes.BOOL)]


def post_drop(window: tk.Misc, paths: list[Path]) -> None:
    """Send the window the same WM_DROPFILES message Explorer sends after a drag and drop."""
    names = ("\0".join(str(path.resolve()) for path in paths) + "\0\0").encode("utf-16-le")
    header = DROPFILES(pFiles=ctypes.sizeof(DROPFILES), fWide=True)
    handle = kernel32.GlobalAlloc(GMEM_MOVEABLE_ZEROINIT, ctypes.sizeof(header) + len(names))
    pointer = kernel32.GlobalLock(handle)
    ctypes.memmove(pointer, ctypes.byref(header), ctypes.sizeof(header))
    ctypes.memmove(pointer + ctypes.sizeof(header), names, len(names))
    kernel32.GlobalUnlock(handle)
    ctypes.windll.user32.PostMessageW(window.winfo_id(), WM_DROPFILES, handle, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("out/ui"))
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")
    args.out.mkdir(parents=True, exist_ok=True)
    bad_file = args.out / "not_a_video.mp4"
    bad_file.write_bytes(b"this is not a video")

    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    ui_theme.enable_dark_dialogs()
    usage.apply(usage.DEFAULT.threads, usage.DEFAULT.priority)
    root = tk.Tk()
    root.withdraw()
    preferences_file = args.out / "preferences_check.json"
    preferences_file.unlink(missing_ok=True)
    app = ui.App(root, preferences=PreferenceStore(preferences_file))
    presets_file = args.out / "presets_check.json"
    presets_file.unlink(missing_ok=True)
    app.presets = PresetStore(presets_file)
    app._update_preset_lists()
    ui_theme.style_title_bar(root)
    root.deiconify()
    failures: list[str] = []
    started = time.monotonic()

    def check(condition: bool, message: str) -> None:
        print(f"{time.monotonic() - started:6.1f}s " + ("ok    " if condition else "FAIL  ") + message, flush=True)
        if not condition:
            failures.append(message)

    def shot(name: str) -> None:
        window_capture.save(root, args.out / f"{name}.png")
        print(f"{time.monotonic() - started:6.1f}s       screenshot {name}.png", flush=True)

    def choose(side: str, choice: str) -> None:
        """Pick an entry of a side's dropdown, as a click does."""
        app.side_boxes[side].current([value for _, value in app._side_entries()].index(choice))
        app._side_chosen(side)

    def pick_preset(side: str, name: str) -> None:
        """Pick a preset in the sidebar's dropdown, as a click does."""
        app.side_blocks[side].preset_choice.set(name)
        app._preset_chosen(side)

    def side_ready(side: str) -> bool:
        """Whether the paused picture has the upscaled version that `side` shows."""
        still = app.view._still
        return still is not None and still[1].get(app.view.sides[side]) is not None

    def pointer_at(x: int, y: int, sequence: str = "<Motion>") -> None:
        """A pointer event at (x, y) on the picture (the real cursor stays where it is)."""
        view = app.view
        view.event_generate(sequence, x=x, y=y, rootx=view.winfo_rootx() + x, rooty=view.winfo_rooty() + y)

    def move_pointer(height_fraction: float, sequence: str = "<Motion>") -> None:
        """A pointer event over the middle of the picture."""
        pointer_at(app.view.winfo_width() // 2, int(app.view.winfo_height() * height_fraction), sequence)

    def center() -> tuple[int, int]:
        return app.view.winfo_width() // 2, app.view.winfo_height() // 2

    def zoom_drag_shifts(slider, start_x: int, end_x: int, y: int, count: int) -> tuple[float, float]:
        """Drag `slider` from `start_x` to `end_x` and back in `count` motions each way and measure how far the
        picture moves sideways: a view enlarged around the view's middle by the change of scale, against the
        next one (phase correlation on a patch left of the divider, clear of the labels and the minimap). Returns
        the largest move from one motion to the next and how far the picture is, back at the start, from where it
        was, in view pixels."""
        view = app.view
        width, height = view.winfo_width(), view.winfo_height()
        rows, columns = slice(int(height * 0.3), int(height * 0.7)), slice(int(width * 0.12), int(width * 0.42))
        window = cv2.createHanningWindow((columns.stop - columns.start, rows.stop - rows.start), cv2.CV_64F)
        middle_x, middle_y = width / 2 - 0.5, height / 2 - 0.5  # the view's middle, in pixel-centre coordinates

        def patch(canvas: np.ndarray) -> np.ndarray:
            return cv2.cvtColor(canvas[rows, columns], cv2.COLOR_BGRA2GRAY).astype(np.float64)

        def moved(before: np.ndarray, ratio: float, after: np.ndarray) -> float:
            matrix = np.array([[ratio, 0, (1 - ratio) * middle_x], [0, ratio, (1 - ratio) * middle_y]])
            expected = cv2.warpAffine(before, matrix, (width, height), flags=cv2.INTER_LINEAR)
            shift, _ = cv2.phaseCorrelate(patch(expected), patch(after), window)
            return float(np.hypot(*shift))

        slider.event_generate("<ButtonPress-1>", x=start_x, y=y)
        first, first_scale = view.snapshot(), view.zoom_scale(view.zoom)
        previous, previous_scale = first, first_scale
        largest = 0.0
        for index in list(range(1, count + 1)) + list(range(count - 1, -1, -1)):
            slider.event_generate("<B1-Motion>", x=int(round(start_x + (end_x - start_x) * index / count)), y=y)
            current, scale = view.snapshot(), view.zoom_scale(view.zoom)
            largest = max(largest, moved(previous, scale / previous_scale, current))
            previous, previous_scale = current, scale
        slider.event_generate("<ButtonRelease-1>", x=start_x, y=y)
        return largest, moved(first, previous_scale / first_scale, previous)

    def steps():
        yield lambda: app.gpu_ready is not None
        check(app.gpu_ready is True, f"GPU check: {app.gpu_label.cget('text')}")
        check("RAM" in app.cpu_label.cget("text") and "VRAM" in app.gpu_label.cget("text"),
              f"machine status: {app.cpu_label.cget('text')} | {app.gpu_label.cget('text')}")
        version = app.version_label.cget("text")
        check(version.startswith(f"v{fsr4.__version__} · ") and version.endswith(" · from the sources")
              and "unknown" not in version, f"the version names its commit: {version}")
        check(app.choose_button.cget("text") == "Choose file…", "button reads Choose file… without a file")
        shot("01_empty")
        post_drop(root, [args.video])
        yield lambda: app.info is not None or "Cannot" in app.banner_label.cget("text")
        check(app.info is not None and app.input_path.name == args.video.name, "dropped video loaded")
        check(app.choose_button.cget("text") == "Choose another file…", "button reads Choose another file…")
        loaded = time.monotonic()
        yield lambda: app.view.comparing and side_ready("right")
        check(time.monotonic() - loaded < 6, f"first upscaled frame shown {time.monotonic() - loaded:.1f} s after load")
        check(app.transport.winfo_ismapped(), "playback controls shown for a video")
        yield 0.5
        shot("02_loaded")
        chip = app.size_chips["1440"]
        chip.event_generate("<Enter>", x=5, y=5)
        yield 0.8
        tips = [child for child in chip.winfo_children() if isinstance(child, tk.Toplevel)]
        check(bool(tips), "hovering a size chip shows a tooltip")
        if tips:
            window_capture.save_handle(int(tips[0].wm_frame(), 16), args.out / "02a_tooltip.png")
            print("      screenshot 02a_tooltip.png", flush=True)
            root.iconify()  # no Leave event: the tooltip must notice by itself
            yield 0.6
            check(not tips[0].winfo_exists(), "minimizing the window takes its tooltip away")
            root.deiconify()
            yield 0.8
        chip.event_generate("<Leave>")
        root.tk.call("ttk::combobox::Post", app.encoder_choice)
        yield 0.4
        popdown = root.tk.call("ttk::combobox::PopdownWindow", app.encoder_choice)
        window_capture.save_handle(int(root.tk.call("wm", "frame", popdown), 16), args.out / "02b_dropdown.png")
        print("      screenshot 02b_dropdown.png", flush=True)
        root.tk.call("ttk::combobox::Unpost", app.encoder_choice)
        found: list = []
        dialog_done = threading.Event()
        watcher = close_dialog("Choose a video or an image", args.out / "02c_file_dialog.png", found, dialog_done)
        try:
            app.browse()
        finally:
            dialog_done.set()
        watcher.join(timeout=5)
        check(bool(found), "file dialog opened and closed")
        app.toggle_play()
        yield lambda: app.player.playing and not app.player.buffering and app.player.position > 0
        rate = float(app.info.frame_rate)  # the speed checks below ask for half the real speed
        first = app.player.position
        yield 2.0
        played = app.player.position - first
        check(played >= rate or app.player.buffering, f"playback advanced {played} frames in 2 s")
        shot("03_playing")
        bar = app.transport.bar
        bar.event_generate("<Motion>", x=bar.winfo_width() // 3, y=bar.winfo_height() // 2)
        yield 0.3
        strip = app.transport.strip
        texts = [strip.itemcget(item, "text") for item in strip.find_withtag("overlay") if strip.type(item) == "text"]
        check(any(text.startswith("frame ") and "." in text for text in texts),
              f"hovering the seek bar shows the frame and its time to the millisecond: {texts}")
        shot("03b_seek_bar_hover")
        bar.event_generate("<Leave>")
        transport = app.transport
        positions = []
        for frame in (5, app._total() - 1):
            transport.set_position(frame)
            root.update_idletasks()
            positions.append(transport.volume_bar.winfo_x())
        check(positions[0] == positions[1], f"the volume stays in place as the time grows: x {positions}")
        volume_bar = transport.volume_bar
        volume_y = volume_bar.winfo_height() // 2
        volume_bar.event_generate("<ButtonPress-1>", x=int(round(volume_bar.x_of(volume_fraction(0.2)))), y=volume_y)
        volume_bar.event_generate("<ButtonRelease-1>", x=0, y=0)
        percent = float(transport.volume_label.cget("text").rstrip("%"))
        check(abs(percent - 20) <= 1, f"a click where the volume slider reads 20% sets it: {percent}%")
        volume_labels = {mark.value: (left + right) / 2 for mark, left, right in volume_bar._labels}
        check(len(volume_labels) == 6, f"the volume slider labels 0, 1, 3, 10, 30 and 100 ({sorted(volume_labels)})")
        if 0.1 in volume_labels:
            for sequence in ("<ButtonPress-1>", "<ButtonRelease-1>"):
                volume_bar.event_generate(sequence, x=int(round(volume_labels[0.1])), y=volume_y + app.theme.px(13))
            check(transport.volume == 0.1, f"a click on the 10 label sets 10% ({transport.volume_label.cget('text')})")
        transport.change_volume(VOLUME_KEY_DECIBELS)
        check(transport.volume_label.cget("text") == "16%",
              f"up raises the volume 4 dB ({transport.volume_label.cget('text')})")
        shot("03c_volume")
        transport.toggle_mute()
        check(transport.volume_label.cget("style") == "Faint.TLabel", "muted, the percentage greys out")
        transport.toggle_mute()
        transport.set_volume(0.8)
        total = app._total()
        app.seek_to(int(total * 0.8), True)
        yield 0.4
        check(app.player.buffering or app.player.position >= int(total * 0.8), "seek into frames not upscaled yet")
        shot("04_buffering")
        yield lambda: not app.player.buffering or not app.player.playing
        yield 1.0
        app.toggle_play()
        yield lambda: not app.player.playing
        yield 0.8
        choose("left", "nearest")
        app.view.set_zoom(2, anchor=center())
        yield 0.6
        check(app.zoom_slider.current == 2 and app.zoom_label.cget("text") == "200%",
              f"the zoom slider follows the view ({app.zoom_label.cget('text')})")
        check(app.view.sides["left"] == "nearest", "left side shows the original pixels")
        shot("05_paused_zoom_original")
        view = app.view
        handle_before = view.handle_y
        move_pointer(0.25)
        yield 0.3
        check(view.handle_y == handle_before, "the pointer passing over the picture leaves the handle where it is")
        move_pointer(0.3, "<ButtonPress-1>")
        move_pointer(0.25, "<B1-Motion>")
        move_pointer(0.25, "<ButtonRelease-1>")
        yield 0.3
        shown_left, _, shown_width, _ = view._geometry.shown

        def handle_mismatch(x: int, y: int) -> int:
            """The largest difference between the picture and the handle centred on (x, y), over its opaque
            pixels: 0 when the handle is drawn there."""
            sprite = view._handle
            half = sprite.shape[0] // 2
            shown = view.snapshot()[y - half:y - half + sprite.shape[0], x - half:x - half + sprite.shape[1], :3]
            opaque = sprite[..., 3] == 255
            return int(np.abs(shown.astype(int) - sprite[..., :3].astype(int))[opaque].max())

        divider_x = shown_left + int(round(view.split * shown_width))
        handle_y = int(view.winfo_height() * 0.25)
        mismatch = handle_mismatch(divider_x, handle_y)
        check(view.handle_y == handle_y and mismatch <= 2,
              f"zoomed in, dragging the divider takes its handle to the pointer's height (y {handle_y}, largest "
              f"difference {mismatch})")
        move_pointer(0.6)
        yield 0.2
        check(view.handle_y == handle_y, "once released, the handle stays where it was dragged")
        shot("05_handle_at_pointer")

        def around_pointer(x: int, y: int) -> tuple[int, int, int]:
            """With the system's west-east pointer's hotspot at (x, y): the lightest picture pixel under the
            pointer's footprint (the handle's dark disc when the pointer fits inside its chevrons), then, along the
            hotspot's row, the pixels between each of the pointer's tips and the white beyond it."""
            arrow, (hotspot_x, hotspot_y) = view.handle_arrow, view.handle_hotspot
            height, width = arrow.shape[:2]
            snapshot = view.snapshot()
            shown = snapshot[y - hotspot_y:y - hotspot_y + height, x - hotspot_x:x - hotspot_x + width, :3]
            footprint = arrow[..., 3] > 0
            tips = np.nonzero(footprint[hotspot_y])[0]
            left_tip, right_tip = x - hotspot_x + int(tips[0]), x - hotspot_x + int(tips[-1])
            white = np.nonzero(snapshot[y, :, :3].min(axis=1) >= 200)[0]
            return (int(shown[footprint].max()), left_tip - int(white[white < left_tip].max()) - 1,
                    int(white[white > right_tip].min()) - right_tip - 1)

        center_y = view.handle_y
        reach = view._handle.shape[0] // 3
        press_x, press_y = divider_x - reach, center_y + reach // 2
        pointer_at(press_x, press_y, "<ButtonPress-1>")
        pressed = handle_mismatch(press_x, press_y), *around_pointer(press_x, press_y)
        pointer_at(press_x + 57, press_y - 41, "<B1-Motion>")
        dragged = handle_mismatch(press_x + 57, press_y - 41), *around_pointer(press_x + 57, press_y - 41)
        pointer_at(divider_x, center_y, "<B1-Motion>")
        pointer_at(divider_x, center_y, "<ButtonRelease-1>")
        check(all(mismatch <= 2 and under <= 60 and 1 <= left <= app.theme.px(4) and 1 <= right <= app.theme.px(4)
                  for mismatch, under, left, right in (pressed, dragged)),
              f"pressed beside its middle and dragged, the handle stays centred on the pointer, which fits inside its "
              f"chevrons (largest difference, lightest pixel under the pointer, gaps to the chevrons: {pressed} "
              f"pressed, {dragged} dragged)")
        zoom_slider = app.zoom_slider
        track_y = zoom_slider.winfo_height() // 2

        def zoom_x(scale: float) -> int:
            return int(round(zoom_slider.x_of(app._zoom_fraction(scale))))

        def click_zoom(x: float, y: int) -> None:
            for sequence in ("<ButtonPress-1>", "<ButtonRelease-1>"):
                zoom_slider.event_generate(sequence, x=int(round(x)), y=y)

        click_zoom(zoom_x(1.5), track_y)
        yield 0.6
        check(abs(app.view.zoom_scale(app.view.zoom) - 1.5) < 0.02 and app.zoom_label.cget("text") == "150%",
              f"a click on the zoom slider zooms there ({app.zoom_label.cget('text')})")
        shot("05a_zoom_slider")
        click_zoom(zoom_x(1.0) + app.theme.px(3), track_y)
        check(app.view.zoom == 1, f"a click next to the 100% mark lands on it ({app.view.zoom})")
        zoom_slider.event_generate("<ButtonPress-1>", x=zoom_x(1.0), y=track_y)
        zoom_slider.event_generate("<B1-Motion>", x=zoom_x(3.0), y=track_y)
        zoom_slider.event_generate("<ButtonRelease-1>", x=zoom_x(3.0), y=track_y)
        check(abs(app.view.zoom_scale(app.view.zoom) - 3.0) < 0.05, f"a drag zooms along ({app.zoom_label.cget('text')})")
        largest, drift = zoom_drag_shifts(zoom_slider, zoom_x(1.0), zoom_x(6.0), track_y, 40)
        check(largest <= 1.5 and drift <= 0.5,
              f"dragging the zoom slider to 600% and back keeps the picture still around the middle (largest step "
              f"{largest:.2f} px; back at 100%, {drift:.2f} px from where it was)")
        labels ={mark.value: (left + right) / 2 for mark, left, right in zoom_slider._labels}
        check({1, 16} <= set(labels), f"the zoom slider labels 100% and 1600% at least ({sorted(map(str, labels))})")
        check(app.zoom_fit_var.get() == "", "away from Fit, the Fit button is not lit")
        if "source" in labels:
            click_zoom(labels["source"], track_y + app.theme.px(13))
            check(app.view.zoom == "source", f"a click on the Original label zooms there ({app.view.zoom})")
        zoom_slider.event_generate("<MouseWheel>", delta=120, x=zoom_x(1.0), y=track_y)
        check(app.view.zoom_scale(app.view.zoom) > app.view.zoom_scale("source"),
              f"the wheel over the zoom slider zooms in ({app.zoom_label.cget('text')})")
        app.zoom_fit_button.invoke()
        check(app.view.zoom == "fit" and app.zoom_fit_var.get() == "fit", "the Fit button fits the picture and lights")
        app.view.set_zoom(app.view.zoom_step(1))
        check(app.view.fit_scale() < app.view.zoom_scale(app.view.zoom) < app.view.fit_scale() * 1.2,
              f"a wheel notch zooms in a quarter octave ({app.zoom_label.cget('text')})")
        app.view.set_zoom("source")
        yield 0.6
        check(app.zoom_slider.current == "source", "the zoom slider follows Original size")
        shot("05b_original_size")
        app.view.set_zoom("fit")
        choose("left", "cubic")
        left, top, right, bottom = app.view._pill_boxes["right"]
        for sequence in ("<ButtonPress-1>", "<ButtonRelease-1>"):
            app.view.event_generate(sequence, x=(left + right) // 2, y=(top + bottom) // 2)
        yield 0.5
        popdown = root.tk.call("ttk::combobox::PopdownWindow", app.side_boxes["right"])
        check(root.tk.call("winfo", "ismapped", popdown) == 1 and app.side_choices["right"] == ui.UPSCALED,
              "a click on the label on the picture opens its side's dropdown")
        window_capture.save_handle(int(root.tk.call("wm", "frame", popdown), 16), args.out / "05c_side_dropdown.png")
        print("      screenshot 05c_side_dropdown.png", flush=True)
        root.tk.call("ttk::combobox::Unpost", app.side_boxes["right"])
        # a preset of one's own, compared with FSR 3.1
        left_block, right_block = app.side_blocks["left"], app.side_blocks["right"]
        check(right_block.winfo_ismapped() and not left_block.winfo_ismapped()
              and right_block.preset_choice.get() == BEST.name, "the settings are the upscaled side's, the best ones")
        right_block.sharpen_var.set(True)
        right_block.sharpness_var.set(0.8)
        app._settings_changed()
        app._begin_save_preset("right")
        yield 0.3
        shot("05d_naming_a_preset")
        right_block.name_var.set("Smoke sharp")
        app._finish_save_preset("right")
        check(app.presets.find("Smoke sharp") is not None and app.side_boxes["right"].get() == "Smoke sharp"
              and right_block.preset_choice.get() == "Smoke sharp", "a preset is saved and named in both lists")
        yield 0.8
        choose("left", "preset:FSR 3.1")
        yield lambda: side_ready("left") and side_ready("right")
        app.view.set_zoom(2, anchor=center())
        yield 0.8
        check(app.view.side_labels == {"left": "FSR 3.1", "right": "Smoke sharp"},
              f"labels name the presets: {app.view.side_labels}")
        check(app.save_row.winfo_ismapped() and app._options().upscaler is None and app._options().sharpness == 0.8,
              "with both sides upscaled the window asks which one to set and save, the right one to begin with")
        shot("05e_presets_side_by_side")
        app.save_side_var.set("left")
        app._settings_changed()
        yield 0.8
        check(app._options().upscaler == "3.1.6" and left_block.winfo_ismapped() and not right_block.winfo_ismapped()
              and left_block.preset_choice.get() == "FSR 3.1", "the left side can be the one set and saved")
        shot("05e2_left_side_settings")
        app.save_side_var.set("right")
        app._settings_changed()
        app.view.set_zoom("fit")
        app.seek_to(int(app._total() * 0.6), True)  # the last 40%: a stall and the end, in seconds
        app.toggle_play()
        yield lambda: app.player.playing and not app.player.buffering and app.player.position > 0
        first = app.player.position
        yield 2.0
        check(app.player.position - first >= rate, f"plays two upscaled versions ({app.player.position - first} "
                                                   f"frames in 2 s)")
        shot("05f_presets_playing")
        give_up = time.monotonic() + 90
        yield lambda: not app.player.playing or time.monotonic() > give_up
        check(not app.player.playing and app.player.position >= app._total() - 2,
              f"two versions play to the end, waiting for the upscaler on the way ({app.player.position} of {app._total()})")
        pick_preset("right", "FSR 4")
        check(not right_block.sharpen_var.get() and app.side_boxes["right"].get() == "FSR 4",
              "choosing a preset in the sidebar loads it, and the side's list follows")
        choose("right", "preset:Smoke sharp")
        check(right_block.how().sharpness == 0.8 and right_block.preset_choice.get() == "Smoke sharp",
              "choosing a preset in the side's list loads it, and the sidebar follows")
        app._delete_preset("right")
        check(app.presets.find("Smoke sharp") is None and right_block.preset_choice.get() == "Custom settings"
              and right_block.how().sharpness == 0.8, "a preset of one's own is deleted, the side keeps its settings")
        choose("right", "preset:FSR 4")
        choose("left", "cubic")
        yield 1.0
        start = app.player.position
        for sequence in ("<ButtonPress-1>", "<ButtonRelease-1>"):
            app.transport.frame_back_button.event_generate(sequence)
        yield 0.6
        check(app.player.position == start - 1, f"the previous-frame button steps one frame ({start} -> "
                                                f"{app.player.position})")
        speed_bar = app.transport.speed_bar
        volume_track = app.transport.volume_bar.winfo_rooty() + app.transport.volume_bar.winfo_height() // 2
        speed_track = speed_bar.winfo_rooty() + speed_bar.winfo_height() // 2
        check(volume_track == speed_track, f"the volume and speed tracks share a line ({volume_track}, {speed_track})")

        def speed_x(speed: float) -> int:
            inset = app.theme.px(KNOB_INSET)
            return int(round(inset + (speed_bar.winfo_width() - 2 * inset) * speed_fraction(speed)))

        def click_speed(speed: float) -> None:
            for sequence in ("<ButtonPress-1>", "<ButtonRelease-1>"):
                speed_bar.event_generate(sequence, x=speed_x(speed), y=speed_bar.winfo_height() // 2)

        click_speed(0.25)
        check(app.transport.speed == 0.25, f"a click on the ¼ mark plays at 0.25× ({app.transport.speed})")
        app.transport.reverse_button.invoke()
        check(app.transport.speed_label.cget("text") == "−0.25×",
              f"the reverse toggle turns the speed round ({app.transport.speed_label.cget('text')})")
        app.toggle_play()
        yield lambda: app.player.playing and not app.player.buffering
        first = app.player.position
        yield 2.0
        back = first - app.player.position
        check(back >= rate / 4, f"plays backwards at a quarter speed ({back} frames back in 2 s)")
        shot("05g_backwards")
        speed_bar.event_generate("<ButtonPress-1>", x=speed_x(0.5), y=4)
        speed_bar.event_generate("<B1-Motion>", x=(speed_x(0.25) + speed_x(0.5)) // 2, y=4)
        check(app.transport.speed == -0.35, f"a drag stops between the marks too ({app.transport.speed})")
        shot("05g2_speed_dragged")
        speed_bar.event_generate("<B1-Motion>", x=speed_x(0.0), y=4)
        speed_bar.event_generate("<ButtonRelease-1>", x=speed_x(0.0), y=4)
        yield lambda: not app.player.playing
        check(app.transport.speed_label.cget("text") == "0×", "at 0 the video pauses")
        shot("05g3_speed_at_0")
        click_speed(0.5)
        yield lambda: app.player.playing and not app.player.buffering
        check(app.player.speed == -0.5, f"lifting the slider plays on, still backwards ({app.player.speed})")
        click_speed(0.0)
        yield lambda: not app.player.playing
        app.toggle_play()
        yield lambda: app.player.playing
        check(app.transport.speed == -1.0, f"Play at 0 goes back to normal speed ({app.transport.speed})")
        app.player.pause()
        yield lambda: not app.player.playing
        app.transport.speed_label.event_generate("<ButtonRelease-1>")
        yield lambda: app.player.speed == 1.0
        check(not app.transport.backwards, "a click on the value goes back to 1× forwards")
        choose("left", "off")
        yield 0.8
        check(app.view.single and app.view.needed_layers() == {app.view.sides["right"]},
              "left side off: the right one fills the picture")
        shot("05h_left_side_off")
        choose("right", "off")
        check(app.side_choices == {"left": "cubic", "right": "off"}, "turning the other side off brings the first back")
        check(app.start_button.instate(["disabled"]), "neither side upscaled: nothing to save")
        yield 0.3
        check(app.save_note.winfo_ismapped(), "the window says why Upscale is off")
        yield lambda: app.render_position is None
        check(app.status_var.get() == ui.NOTHING_UPSCALED, "with no side upscaled the upscaler rests")
        shot("05i_nothing_upscaled")
        choose("right", "preset:FSR 4")
        yield 0.5
        app.seek_to(0, True)
        windowed = root.geometry()
        other = next((screen for screen in ui_window.monitors() if not screen[4]), None)
        if other is not None:  # the second monitor: full screen must cover that one
            root.geometry(f"+{other[0] + 120}+{other[1] + 120}")
            yield 0.6
        placed = root.geometry()
        app.toggle_fullscreen()
        yield 1.0
        monitor = ui_window.monitor_of(root)
        covered = (app.view.winfo_rootx(), app.view.winfo_rooty(), app.view.winfo_width(), app.view.winfo_height())
        check(covered == monitor, f"full screen covers the window's monitor {monitor}: {covered}")
        shot("06_fullscreen")
        app.toggle_play()
        yield 3.5
        check(app.player.playing, "still playing after 3.5 s in full screen")
        check(not app.transport.winfo_ismapped(), "controls hide once the pointer rests")
        shot("06b_fullscreen_playing")
        move_pointer(0.97)
        yield 0.3
        check(app.transport.winfo_ismapped(), "moving the pointer near the bottom shows the controls")
        shot("06c_fullscreen_controls")
        yield 3.2
        check(not app.transport.winfo_ismapped(), "they hide when the pointer rests, even at the bottom")
        move_pointer(0.4)
        yield 0.3
        check(not app.transport.winfo_ismapped(), "moving in the middle of the picture leaves them hidden")
        app.toggle_play()
        yield 0.4
        check(app.transport.winfo_ismapped(), "pausing shows them")
        yield 3.2
        check(not app.transport.winfo_ismapped(), "they hide at rest while paused too")
        app.toggle_fullscreen()
        yield 0.8
        check(root.geometry() == placed, f"leaving full screen puts the window back as it was ({root.geometry()})")
        content, canvas = app.sidebar_content, app.sidebar_canvas
        check(content.winfo_width() == canvas.winfo_width(), "the settings keep their width after full screen")
        shot("06d_after_fullscreen")
        root.state("zoomed")  # maximized: full screen still covers the monitor, and leaving maximizes again
        yield 0.6
        app.toggle_fullscreen()
        yield 1.0
        covered = (app.view.winfo_rootx(), app.view.winfo_rooty(), app.view.winfo_width(), app.view.winfo_height())
        check(covered == ui_window.monitor_of(root), f"full screen from a maximized window covers the monitor: {covered}")
        app.toggle_fullscreen()
        yield 0.8
        check(root.state() == "zoomed", f"leaving it, the window is maximized again ({root.state()})")
        root.state("normal")
        root.geometry(windowed)
        yield 0.6
        redraw_styles = ui_window.CS_HREDRAW | ui_window.CS_VREDRAW
        classes = [ui_window.user32.GetClassLongPtrW(handle, ui_window.GCL_STYLE)
                   for handle in (root.winfo_id(), ui_window.frame_handle(root))]
        check(not any(style & redraw_styles for style in classes), f"a size change repaints no widget: {classes}")
        curtains = []
        make_curtain = ui.Curtain
        ui.Curtain = lambda window: curtains.append(make_curtain(window)) or curtains[-1]
        fold_place = app.fold_button.winfo_rootx(), app.fold_button.winfo_rooty()
        app.toggle_sidebar()
        yield 0.6
        ui.Curtain = make_curtain
        check(len(curtains) == 1 and curtains[0].closed, "the fold's curtain fades and is gone")
        check(not app.sidebar.winfo_ismapped(), "sidebar folds away")
        unfold_place = app.unfold_button.winfo_rootx(), app.unfold_button.winfo_rooty()
        check(fold_place == unfold_place, f"its button stays in place ({fold_place}, then {unfold_place})")
        shot("07_sidebar_hidden")
        app.toggle_sidebar()
        app.start()
        yield lambda: not app.busy
        check(app.last_saved is not None and app.last_saved.exists(), "whole video saved")
        if app.last_saved is not None and app.last_saved.exists():
            saved = media.probe(app.last_saved)
            check(abs((saved.frame_count or 0) - total) <= 1, f"saved {saved.frame_count} of {total} frames")
            check([track.codec for track in saved.audio] == ["aac"],
                  f"{'/'.join(track.codec for track in app.info.audio)} audio saved as AAC")
        yield 0.5
        shot("08_saved")
        # A default name never replaces a file, so each run saves a new one: remove it once checked.
        app.last_saved.unlink()
        # AMD's HQ scaler: no motion to set, the encoder's ffmpeg upscales; beside FSR 4, then the whole file.
        pick_preset("right", "AMD HQ scaler")
        yield 0.8  # the new settings apply after a moment
        check(right_block.sharpen_row.winfo_ismapped() and not right_block.motion_choice.winfo_ismapped(),
              "the HQ scaler shows its sharpening, no motion")
        choose("left", "preset:FSR 4")
        yield lambda: side_ready("left") and side_ready("right")
        app.view.set_zoom(2, anchor=center())
        yield 0.8
        check(app.view.side_labels == {"left": "FSR 4", "right": "AMD HQ scaler"},
              f"labels name the upscalers: {app.view.side_labels}")
        shot("08b_hq_scaler_beside_fsr4")
        app.view.set_zoom("fit")
        started = time.monotonic()
        app.start()
        yield lambda: not app.busy
        check(app.last_saved is not None and app.last_saved.exists(), "whole video saved with the HQ scaler")
        saved = media.probe(app.last_saved)
        check(abs((saved.frame_count or 0) - total) <= 1 and saved.width == app._plan().output_size[0],
              f"saved {saved.frame_count} of {total} frames at {saved.width} wide "
              f"in {time.monotonic() - started:.1f} s")
        app.last_saved.unlink()
        pick_preset("right", "Real-ESRGAN")
        yield 0.8  # the new settings apply after a moment
        check(right_block.upscaler_choice.winfo_ismapped() and not right_block.motion_choice.winfo_ismapped()
              and not right_block.sharpen_row.winfo_ismapped(), "Real-ESRGAN shows no motion and no sharpening")
        shot("08c0_realesrgan_settings")
        yield lambda: side_ready("left") and side_ready("right")
        app.view.set_zoom(2, anchor=center())
        yield 0.8
        check(app.view.side_labels == {"left": "FSR 4", "right": "Real-ESRGAN"},
              f"labels name the upscalers: {app.view.side_labels}")
        shot("08c_realesrgan_beside_fsr4")
        app.view.set_zoom("fit")
        app.start()
        yield 6.0
        check(app.batch and app.renderer.timeline.ready_until(0) >= 0, "a whole-file upscale with Real-ESRGAN runs")
        choose("right", "nearest")
        check(app.batch and app.renderer.settings.upscaler == REAL_ESRGAN and app._options().upscaler == REAL_ESRGAN,
              "a side changed during a whole-file upscale leaves it as it started")
        stopped = time.monotonic()
        app.cancel()
        yield lambda: not app.busy
        check(time.monotonic() - stopped < 8.0, f"it stops in {time.monotonic() - stopped:.1f} s")
        check(app.renderer.settings.upscaler is None, "once stopped, the side still upscaled is the saved one")
        choose("right", "preset:Real-ESRGAN")
        with mock.patch("fsr4.job.find_program", return_value=None):
            app._refresh()
            yield 0.5
            check("not installed" in app.size_note.cget("text") and app.start_button.instate(["disabled"]),
                  f"without its program the window says how to install it: {app.size_note.cget('text')}")
            shot("08d_realesrgan_missing")
        app._refresh()
        check(len(right_block.upscalers) == 5 and "Real-ESRGAN x4plus" not in app.side_boxes["left"].cget("values")
              and "Real-ESRGAN x4plus" not in right_block.preset_choice.cget("values"),
              "a video is not offered the models for still images")
        choose("right", "preset:FSR 4")
        choose("left", "cubic")
        app.load(args.image)
        yield lambda: app.info is not None
        yield lambda: side_ready("right")
        check(True, "the image is upscaled as soon as it opens")
        check(right_block.sharpen_row.winfo_ismapped() and not right_block.motion_choice.winfo_ismapped(),
              "an image has no motion to set")
        choose("left", "preset:FSR 2.3")
        yield lambda: side_ready("left")
        yield 0.5
        shot("09_image_presets")
        choose("left", "preset:AMD HQ scaler")
        yield lambda: side_ready("left")
        yield 0.5
        shot("09a_image_hq_scaler")
        choose("left", "preset:Real-ESRGAN")
        yield lambda: side_ready("left")
        yield 0.5
        shot("09a_image_realesrgan")
        for name, screenshot in (("Real-ESRGAN x4plus", "09a_image_x4plus"), ("Real-ESRGAN anime", "09a_image_anime")):
            choose("left", f"preset:{name}")
            yield lambda: side_ready("left")
            yield 0.5
            shot(screenshot)
        app.start()
        yield lambda: not app.busy and app.last_saved is not None
        check(app.last_saved.exists(), "image job finished")
        choose("left", "cubic")
        yield 0.5
        shot("09b_image_saved")
        app.last_saved.unlink()
        # Esc (stop) while the image is being upscaled cancels it: nothing is written.
        files_before = set(args.image.parent.iterdir())
        app.start()
        app.cancel()
        yield lambda: not app.busy
        check(app.status_var.get() == "Cancelled · nothing was saved" and app.last_saved is None
              and set(args.image.parent.iterdir()) == files_before,
              f"Esc cancels the image upscale ({app.status_var.get()})")
        app.load(bad_file)
        yield lambda: app.banner.winfo_ismapped()
        yield 0.3
        shot("10_error")
        check("Cannot open" in app.banner_label.cget("text"), "bad file shows the error banner")
        app._hide_banner()
        choose("left", "preset:Real-ESRGAN anime")
        app.load(args.video)
        yield lambda: app.info is not None
        check(left_block.how().upscaler == REAL_ESRGAN, "opening a video takes a model for still images off its side")
        choose("left", "cubic")
        # The switch off, a file opens as in a plain player: the source alone, nothing upscaled, until a preset is
        # chosen. Its pieces wait beside the output.
        app.upscale_on_open_switch.invoke()
        kept = PreferenceStore(preferences_file).preferences
        check(not kept.upscale_on_open, f"the preferences are kept in their file ({kept})")
        app.load(args.video)
        yield lambda: app.info is not None and app.renderer is not None
        check(app.side_choices == {"left": "cubic", "right": ui.OFF},
              f"opened without upscaling, the source alone ({app.side_choices})")
        output_folder = Path(app.output_var.get()).resolve().parent
        check(app.renderer.directory.parent == output_folder,
              f"the pieces wait beside the output ({app.renderer.directory})")
        app.toggle_play()
        yield 2.0
        check(app.player.position > 0 and app.render_position is None and app.renderer.timeline.ready_frames() == 0,
              f"it plays the source and nothing is upscaled (frame {app.player.position})")
        check(app.status_var.get() == ui.NOTHING_UPSCALED, f"the status says how to upscale ({app.status_var.get()})")
        shot("12_opened_without_upscaling")
        switch, choose_button, content = app.upscale_on_open_switch, app.choose_button, app.sidebar_content
        middle_offset = (switch.winfo_rooty() + switch.winfo_height() // 2
                         - choose_button.winfo_rooty() - choose_button.winfo_height() // 2)
        gap = switch.winfo_rootx() - choose_button.winfo_rootx() - choose_button.winfo_width()
        switch_right = switch.winfo_rootx() + switch.winfo_width() - content.winfo_rootx()
        inner_right = content.winfo_width() - app.theme.px(6)
        check(abs(middle_offset) <= 2 and abs(gap - app.theme.px(ui.SWITCH_GAP)) <= 1 and switch_right <= inner_right
              and switch.winfo_width() >= switch.winfo_reqwidth(),
              f"the switch sits whole just after {choose_button.cget('text')!r} (a gap of {gap}, "
              f"ends at {switch_right} of {inner_right})")
        app.toggle_play()
        yield lambda: not app.player.playing
        choose("right", "preset:FSR 4")
        yield lambda: side_ready("right")
        check(True, "choosing a preset starts the upscaler")
        app.upscale_on_open_switch.invoke()
        minimum_width, minimum_height = root.minsize()
        root.geometry(f"{minimum_width}x{minimum_height}")
        yield 1.0
        shot("11_minimum_size")
        button = app.transport.fullscreen_button
        check(button.winfo_x() + button.winfo_width() <= app.transport.controls.winfo_width(),
              f"the playback controls all fit at the minimum size ({button.winfo_x() + button.winfo_width()} "
              f"for {app.transport.controls.winfo_width()})")
        zoom_label = app.zoom_label
        check(zoom_label.winfo_x() + zoom_label.winfo_width() <= zoom_label.master.winfo_width(),
              f"the zoom slider and its value fit at the minimum size ({zoom_label.winfo_x() + zoom_label.winfo_width()}"
              f" for {zoom_label.master.winfo_width()})")
        content, canvas = app.sidebar_content, app.sidebar_canvas
        fits = content.winfo_reqheight() <= canvas.winfo_height() + 1
        check(fits or app.sidebar_scrollbar.winfo_ismapped(),
              f"sidebar fits or scrolls at the minimum size ({content.winfo_reqheight()} for {canvas.winfo_height()})")
        app.sidebar_canvas.yview_moveto(1.0)
        yield 0.4
        shot("11b_minimum_size_scrolled")

    sequence = steps()
    deadline = {"time": 0.0, "condition": None}

    def advance():
        condition = deadline["condition"]
        if condition is not None:
            try:
                met = condition()
            except Exception as error:  # noqa: BLE001 - report and stop
                check(False, f"waiting raised {error!r}")
                app.close()
                return
            if not met:
                if time.monotonic() > deadline["time"]:
                    check(False, "timed out waiting for the window")
                    app.close()
                    return
                root.after(100, advance)
                return
            deadline["condition"] = None
        try:
            step = next(sequence)
        except StopIteration:
            app.close()
            return
        except Exception as error:  # noqa: BLE001 - report and stop
            check(False, f"step raised {error!r}")
            app.close()
            return
        if callable(step):
            deadline["condition"] = step
            deadline["time"] = time.monotonic() + 180
            root.after(100, advance)
        else:
            root.after(int(step * 1000), advance)

    root.after(500, advance)
    root.mainloop()
    print(f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
