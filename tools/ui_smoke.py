"""Drive the window through every state and screenshot each one; exit 0 when every step worked.

    python tools/ui_smoke.py --video clip.wmv --image picture.png --out out/ui

Steps: empty window, a video dropped on it (a real WM_DROPFILES message), its first upscaled frame, a tooltip,
a dropdown, the file dialog, playback, a seek into frames not upscaled yet (buffering), a paused zoom with the
original pixels on the left, the zoom slider and wheel, a side's dropdown opened from its label, a preset saved
and two presets upscaled side by side (paused and playing), a frame step, playing backwards at a quarter speed, a
side turned off, full screen on the window's own monitor (the second one when there is one) with its controls
showing near the bottom and hiding at rest, then back where the window was, the folded sidebar, the whole-file
upscale and save, an image compared with a preset,
the error banner for a bad file, and the minimum window size. Presets go to a scratch file in `--out`, not the
user's. Screenshots go through PrintWindow, so the window may be covered; it does pop up on the desktop.
"""
from __future__ import annotations

import argparse
import ctypes
import math
import os
import sys
import threading
import time
import tkinter as tk
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import window_capture  # noqa: E402
from fsr4 import media, ui, ui_theme, ui_window, usage  # noqa: E402
from fsr4.presets import PresetStore  # noqa: E402

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
    app = ui.App(root)
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

    def side_ready(side: str) -> bool:
        """Whether the paused picture has the upscaled version that `side` shows."""
        still = app.view._still
        return still is not None and still[1].get(app.view.sides[side]) is not None

    def move_pointer(height_fraction: float) -> None:
        """A pointer motion over the picture (the real cursor stays where it is)."""
        view = app.view
        x, y = view.winfo_width() // 2, int(view.winfo_height() * height_fraction)
        view.event_generate("<Motion>", x=x, y=y, rootx=view.winfo_rootx() + x, rooty=view.winfo_rooty() + y)

    def center() -> tuple[int, int]:
        return app.view.winfo_width() // 2, app.view.winfo_height() // 2

    def steps():
        yield lambda: app.gpu_ready is not None
        check(app.gpu_ready is True, f"GPU check: {app.gpu_label.cget('text')}")
        check("RAM" in app.cpu_label.cget("text") and "VRAM" in app.gpu_label.cget("text"),
              f"header: {app.cpu_label.cget('text')} | {app.gpu_label.cget('text')}")
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
        inset = app.theme.px(8)
        volume_bar.event_generate("<ButtonPress-1>", x=inset + (volume_bar.winfo_width() - 2 * inset) // 4,
                                  y=volume_bar.winfo_height() // 2)
        volume_bar.event_generate("<ButtonRelease-1>", x=0, y=0)
        percent = int(transport.volume_label.cget("text").rstrip("%"))
        check(abs(percent - 25) <= 2, f"a click a quarter along the volume slider sets about 25%: {percent}%")
        transport.change_volume(0.1)
        check(transport.volume_label.cget("text") == "30%", "up steps the volume to the next 10%")
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
        check(app.zoom_var.get() == "2", "zoom chip follows the view")
        check(app.view.sides["left"] == "nearest", "left side shows the original pixels")
        shot("05_paused_zoom_original")
        app.zoom_slider_var.set(math.log2(1.5))
        app._zoom_slid()
        yield 0.6
        check(abs(app.view.zoom_scale(app.view.zoom) - 1.5) < 0.01 and app.zoom_var.get() == "",
              f"the slider zooms to {app.zoom_label.cget('text')}")
        shot("05a_zoom_slider")
        app.zoom_slider_var.set(math.log2(1.02))
        app._zoom_slid()
        check(app.view.zoom == 1 and app.zoom_var.get() == "1", "the slider sticks to 100%")
        app.view.set_zoom("fit")
        app.view.set_zoom(app.view.zoom_step(1))
        check(app.view.fit_scale() < app.view.zoom_scale(app.view.zoom) < app.view.fit_scale() * 1.2,
              f"a wheel notch zooms in a quarter octave ({app.zoom_label.cget('text')})")
        app.view.set_zoom("source")
        yield 0.6
        check(app.zoom_var.get() == "source", "Original size chip follows the view")
        shot("05b_original_size")
        app.view.set_zoom("fit")
        choose("left", "cubic")
        left, top, right, bottom = app.view._pill_boxes["right"]
        for sequence in ("<ButtonPress-1>", "<ButtonRelease-1>"):
            app.view.event_generate(sequence, x=(left + right) // 2, y=(top + bottom) // 2)
        yield 0.5
        popdown = root.tk.call("ttk::combobox::PopdownWindow", app.side_boxes["right"])
        check(root.tk.call("winfo", "ismapped", popdown) == 1 and app.side_choices["right"] == "current",
              "a click on the label on the picture opens its side's dropdown")
        window_capture.save_handle(int(root.tk.call("wm", "frame", popdown), 16), args.out / "05c_side_dropdown.png")
        print("      screenshot 05c_side_dropdown.png", flush=True)
        root.tk.call("ttk::combobox::Unpost", app.side_boxes["right"])
        # a preset of one's own, compared with FSR 3.1
        app.sharpen_var.set(True)
        app.sharpness_var.set(0.8)
        app._settings_changed()
        app._begin_save_preset()
        yield 0.3
        shot("05d_naming_a_preset")
        app.preset_name_var.set("Smoke sharp")
        app._finish_save_preset()
        check(app.presets.find("Smoke sharp") is not None and app.preset_choice.get() == "Smoke sharp",
              "a preset is saved and named in the settings")
        yield 0.8
        choose("left", "preset:FSR 3.1")
        yield lambda: side_ready("left") and side_ready("right")
        app.view.set_zoom(2, anchor=center())
        yield 0.8
        check(app.view.side_labels == {"left": "FSR 3.1", "right": "Smoke sharp"},
              f"labels name the presets: {app.view.side_labels}")
        shot("05e_presets_side_by_side")
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
        app.preset_choice.set("FSR 4")
        app._preset_chosen()
        check(not app.sharpen_var.get() and app.preset_choice.get() == "FSR 4", "choosing a preset loads it")
        app.preset_choice.set("Smoke sharp")
        app._preset_chosen()
        app._delete_preset()
        check(app.presets.find("Smoke sharp") is None and app.preset_choice.get() != "Smoke sharp",
              "a preset of one's own is deleted")
        choose("left", "cubic")
        yield 1.0
        start = app.player.position
        for sequence in ("<ButtonPress-1>", "<ButtonRelease-1>"):
            app.transport.frame_back_button.event_generate(sequence)
        yield 0.6
        check(app.player.position == start - 1, f"the previous-frame button steps one frame ({start} -> "
                                                f"{app.player.position})")
        app.transport.set_speed(-0.25, notify=True)
        app.toggle_play()
        yield lambda: app.player.playing and not app.player.buffering
        first = app.player.position
        yield 2.0
        back = first - app.player.position
        check(back >= rate / 4, f"plays backwards at a quarter speed ({back} frames back in 2 s)")
        shot("05g_backwards")
        app.player.pause()
        yield lambda: not app.player.playing
        app.transport.set_speed(1.0, notify=True)
        choose("left", "off")
        yield 0.8
        check(app.view.single and app.view.needed_layers() == {app.view.sides["right"]},
              "left side off: the right one fills the picture")
        shot("05h_left_side_off")
        choose("right", "off")
        check(app.side_choices == {"left": "cubic", "right": "off"}, "turning the other side off brings the first back")
        choose("right", "current")
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
        app.toggle_sidebar()
        yield 0.6
        check(not app.sidebar.winfo_ismapped(), "sidebar folds away")
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
        app.load(args.image)
        yield lambda: app.info is not None
        yield lambda: side_ready("right")
        check(True, "the image is upscaled as soon as it opens")
        choose("left", "preset:FSR 2.3")
        yield lambda: side_ready("left")
        yield 0.5
        shot("09_image_presets")
        app.start()
        yield lambda: not app.busy and app.last_saved is not None
        check(app.last_saved.exists(), "image job finished")
        choose("left", "cubic")
        yield 0.5
        shot("09b_image_saved")
        app.last_saved.unlink()
        app.load(bad_file)
        yield lambda: app.banner.winfo_ismapped()
        yield 0.3
        shot("10_error")
        check("Cannot open" in app.banner_label.cget("text"), "bad file shows the error banner")
        app._hide_banner()
        app.load(args.video)
        yield lambda: app.info is not None
        minimum_width, minimum_height = root.minsize()
        root.geometry(f"{minimum_width}x{minimum_height}")
        yield 1.0
        shot("11_minimum_size")
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
