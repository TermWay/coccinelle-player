"""Entry point of both packaged executables (PyInstaller cannot start a package's `__main__` directly).

"Coccinelle Player.exe" has no console; fsr4.exe has one. Either opens the window when started with no arguments
(a double click) or with a single existing file (a file dropped on the exe icon, or "Open with"); fsr4.exe
hides the console it was given in that case. Any other arguments are the command line
(`fsr4.exe upscale clip.mp4`, `fsr4.exe info`, ...).
"""
import ctypes
import os
import sys
from pathlib import Path

from fsr4.cli import main


def hide_own_console() -> None:
    """Hide the console window when this process is the only one attached to it (started from Explorer)."""
    if os.name != "nt":
        return
    kernel32 = ctypes.windll.kernel32
    window = kernel32.GetConsoleWindow()
    if not window:
        return
    processes = (ctypes.c_uint32 * 4)()
    if kernel32.GetConsoleProcessList(processes, 4) == 1:
        ctypes.windll.user32.ShowWindow(window, 0)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if not arguments or (len(arguments) == 1 and Path(arguments[0]).is_file()):
        hide_own_console()
        sys.exit(main(["ui", *arguments]))
    sys.exit(main())
