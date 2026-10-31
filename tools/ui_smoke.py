"""Drive the window through every state and screenshot each one; exit 0 when every step worked.

    python tools/ui_smoke.py --video out/ui/sample.wmv --image out/lab/in.png --out out/ui

Steps: empty window, a video dropped on it (a real WM_DROPFILES message), its first upscaled frame, a tooltip,
a dropdown, the file dialog, playback, a seek into frames not upscaled yet (buffering), a paused zoom with the
original pixels on the left, full screen with and without its controls, the folded sidebar, the whole-file
upscale and save, an image, the error banner for a bad file, and the minimum window size. Screenshots go
through PrintWindow, so the window may be covered; it does pop up on the desktop.
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import threading
import time
import tkinter as tk
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import window_capture  # noqa: E402
from fsr4 import media, ui, ui_theme, usage  # noqa: E402

WM_DROPFILES = 0x0233
GMEM_MOVEABLE_ZEROINIT = 0x0042

kernel32 = ctypes.windll.kernel32
kernel32.GlobalAlloc.restype = ctypes.c_void_p
kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
kernel32.GlobalLock.restype = ctypes.c_void_p
kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
ctypes.windll.user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]


def close_dialog_later(title: str, screenshot: Path, found: list) -> threading.Thread:
    """Wait for a native dialog with `title`, screenshot it, then close it (it blocks the Tk thread)."""
    user32 = ctypes.windll.user32
    user32.FindWindowW.restype = wintypes.HWND
    user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]

    def watch():
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            handle = user32.FindWindowW(None, title)
            if handle:
                time.sleep(1.0)
                window_capture.save_handle(handle, screenshot)
                found.append(handle)
                user32.PostMessageW(handle, 0x0010, 0, 0)  # WM_CLOSE
                return
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

    def steps():
        yield lambda: app.gpu_ready is not None
        check(app.gpu_ready is True, f"GPU check: {app.gpu_label.cget('text')}")
        check(app.choose_button.cget("text") == "Choose file…", "button reads Choose file… without a file")
        shot("01_empty")
        post_drop(root, [args.video])
        yield lambda: app.info is not None or "Cannot" in app.banner_label.cget("text")
        check(app.info is not None and app.input_path.name == args.video.name, "dropped video loaded")
        check(app.choose_button.cget("text") == "Choose another file…", "button reads Choose another file…")
        loaded = time.monotonic()
        yield lambda: app.view.comparing and app.view._still is not None             and app.view._still[1].get("current") is not None
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
        watcher = close_dialog_later("Choose a video or an image", args.out / "02c_file_dialog.png", found)
        app.browse()
        watcher.join(timeout=5)
        check(bool(found), "file dialog opened and closed")
        app.toggle_play()
        yield lambda: app.player.playing and not app.player.buffering and app.player.position > 0
        first = app.player.position
        yield 2.0
        played = app.player.position - first
        check(played >= 60 or app.player.buffering, f"playback advanced {played} frames in 2 s")
        shot("03_playing")
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
        app._choose_layer("left", "nearest")
        app.view.set_zoom(2, anchor=(app.view.winfo_width() // 2, app.view.winfo_height() // 2))
        yield 0.6
        check(app.zoom_var.get() == "2", "zoom chip follows the view")
        check(app.view.sides["left"] == "nearest", "left side shows the original pixels")
        shot("05_paused_zoom_original")
        app.zoom_var.set("source")
        app.view.set_zoom("source")
        yield 0.6
        shot("05b_original_size")
        app.view.set_zoom("fit")
        app._choose_layer("left", "cubic")
        # two settings side by side: sharpening on, the version without it kept as "before"
        app.sharpen_var.set(True)
        app.sharpness_var.set(0.8)
        app._settings_changed()
        yield lambda: app.renderer.before is not None
        yield lambda: app.renderer.timeline.ready_until(app.player.position) > app.player.position
        app._choose_layer("left", "before")
        yield lambda: app.view._still is not None and app.view._still[1].get("before") is not None             and app.view._still[1].get("current") is not None
        app.view.set_zoom(2, anchor=(app.view.winfo_width() // 2, app.view.winfo_height() // 2))
        yield 0.8
        check("Before" in app.view.layer_labels.get("before", ""), "previous settings available as a layer")
        shot("05c_before_and_after_settings")
        app.view.set_zoom("fit")
        app._choose_layer("left", "cubic")
        app.sharpen_var.set(False)
        app._settings_changed()
        yield 1.0
        check(app.renderer.settings.sharpness is None and app.renderer.before is not None,
              "switching back reuses the earlier version")
        app.seek_to(0, True)
        app.toggle_fullscreen()
        yield 1.0
        check(app.view.winfo_width() == root.winfo_screenwidth(), "full screen video fills the screen width")
        shot("06_fullscreen")
        app.toggle_play()
        yield 3.5
        check(app.player.playing, "still playing after 3.5 s in full screen")
        check(not app.transport.winfo_ismapped(), "controls hide while playing full screen")
        shot("06b_fullscreen_playing")
        app.toggle_play()
        app.toggle_fullscreen()
        yield 0.8
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
            check([track.codec for track in saved.audio] == ["aac"], "WMA audio converted to AAC")
        yield 0.5
        shot("08_saved")
        app.load(args.image)
        yield lambda: app.info is not None
        app.start()
        yield lambda: not app.busy and app.last_saved is not None
        check(app.last_saved.exists(), "image job finished")
        yield 0.5
        shot("09_image")
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
            if not condition():
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
