"""Drive the window automatically and screenshot it: load a file, run the job, capture before and after.

    python tools/ui_smoke.py clip.mp4 --frames 60 --out out/ui

Writes `loaded.png`, `running.png` and `done.png` into `--out` (captures of the real window, so it must be
visible on a screen) and exits 0 when the job finished. Used to check the UI without clicking through it.
"""
from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fsr4 import ui  # noqa: E402

CAPTURE_SCRIPT = r"""
Add-Type -AssemblyName System.Drawing
$bmp = New-Object System.Drawing.Bitmap({width}, {height})
$graphics = [System.Drawing.Graphics]::FromImage($bmp)
$graphics.CopyFromScreen({left}, {top}, 0, 0, $bmp.Size)
$bmp.Save("{path}")
"""


def capture(root: tk.Tk, path: Path) -> None:
    """Screenshot the window's rectangle through GDI (PowerShell, no extra packages)."""
    window = int(root.wm_frame(), 16)
    rect = (ctypes.c_long * 4)()
    ctypes.windll.user32.SetForegroundWindow(window)
    ctypes.windll.user32.GetWindowRect(window, rect)
    left, top, right, bottom = rect
    script = CAPTURE_SCRIPT.format(width=right - left, height=bottom - top, left=left, top=top,
                                   path=str(path).replace("\\", "\\\\"))
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("out/ui"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # The status line contains an arrow; a cp1252 console must not kill the Tk callback that prints it.
    sys.stdout.reconfigure(errors="replace")

    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    root = tk.Tk()
    app = ui.App(root, args.input)
    original_options = app._options

    def limited_options():
        options = original_options()
        options.frames = args.frames
        return options

    app._options = limited_options
    outcome = {"code": 1}

    def after_load():
        capture(root, args.out / "loaded.png")
        app._run()
        root.after(1500, lambda: capture(root, args.out / "running.png"))
        root.after(300, wait_done)

    def wait_done():
        if app.worker is not None and app.worker.is_alive():
            root.after(300, wait_done)
            return
        root.after(1500, finish)

    def finish():
        capture(root, args.out / "done.png")
        outcome["code"] = 0 if app.result_path is not None else 1
        print(f"status: {app.status_var.get()}")
        print(f"gpu: {app.gpu_var.get()}")
        app._close()

    root.after(2500, after_load)
    root.mainloop()
    return outcome["code"]


if __name__ == "__main__":
    sys.exit(main())
