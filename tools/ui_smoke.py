"""Drive the window through every state and screenshot each one; exit 0 when every step worked.

    python tools/ui_smoke.py --video out/ui/sample.wmv --image out/lab/in.png --out out/ui

Steps: empty window, a file dropped on it (a real WM_DROPFILES message), the loaded preview, the live preview
while upscaling, the finished comparison, zoom plus log, an image job, and the error banner for a bad file.
Screenshots go through PrintWindow, so the window may be covered; it does pop up on the desktop.
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
from fsr4 import ui, ui_theme  # noqa: E402

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
    parser.add_argument("--frames", type=int, default=0, help="limit the video job (0: whole file)")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")
    args.out.mkdir(parents=True, exist_ok=True)
    bad_file = args.out / "not_a_video.mp4"
    bad_file.write_bytes(b"this is not a video")

    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    ui_theme.enable_dark_dialogs()
    root = tk.Tk()
    root.withdraw()
    app = ui.App(root)
    ui_theme.style_title_bar(root)
    root.deiconify()
    original_options = app._options

    def limited_options():
        options = original_options()
        options.frames = args.frames or None
        return options

    app._options = limited_options
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
        shot("01_empty")
        post_drop(root, [args.video])
        yield lambda: app.info is not None or "Cannot" in app.banner_label.cget("text")
        check(app.info is not None and app.input_path.name == args.video.name, "dropped video loaded")
        yield 0.4
        shot("02_loaded")
        check(app.start_button.instate(["!disabled"]), "Upscale enabled after loading")
        root.tk.call("ttk::combobox::Post", app.motion_choice)
        yield 0.4
        popdown = root.tk.call("ttk::combobox::PopdownWindow", app.motion_choice)
        window_capture.save_handle(int(root.tk.call("wm", "frame", popdown), 16), args.out / "02b_dropdown.png")
        print("      screenshot 02b_dropdown.png", flush=True)
        root.tk.call("ttk::combobox::Unpost", app.motion_choice)
        found: list = []
        watcher = close_dialog_later("Choose a video or an image", args.out / "02c_file_dialog.png", found)
        app.browse()
        watcher.join(timeout=5)
        check(bool(found), "file dialog opened and closed")
        job_started = time.monotonic()
        app.start()
        yield lambda: app.view.comparing
        first_preview = time.monotonic() - job_started
        check(first_preview < 3.0, f"first live preview {first_preview:.1f} s after start")
        yield 0.6
        shot("03_running")
        check(app.running, "job running with a live comparison")
        yield lambda: not app.running
        check(app.last_result is not None and app.last_result.output.exists(), "video job finished")
        yield 0.5
        shot("04_done")
        app.view.split = 0.35
        app.view.set_zoom(2, anchor=(app.view.winfo_width() // 2, app.view.winfo_height() // 2))
        app.toggle_log()
        yield 0.5
        shot("05_zoom_log")
        check(app.zoom_var.get() == 2, "zoom chip follows the view")
        app.toggle_log()
        app.load(args.image)
        yield lambda: app.info is not None
        app.start()
        yield lambda: not app.running and app.last_result is not None
        check(app.last_result.output.exists(), "image job finished")
        yield 0.5
        shot("06_image")
        app.load(bad_file)
        yield lambda: app.banner.winfo_ismapped()
        yield 0.3
        shot("07_error")
        check("Cannot open" in app.banner_label.cget("text"), "bad file shows the error banner")
        app._hide_banner()
        app.load(args.video)
        yield lambda: app.info is not None
        minimum_width, minimum_height = root.minsize()
        root.geometry(f"{minimum_width}x{minimum_height}")
        yield 0.8
        shot("08_minimum_size")
        sidebar_bottom = app.audio_note.winfo_rooty() + app.audio_note.winfo_height()
        footer_top = app.progress.winfo_rooty()
        check(sidebar_bottom <= footer_top, f"sidebar fits at the minimum size ({sidebar_bottom} <= {footer_top})")

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
