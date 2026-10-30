"""Entry point for the packaged exe (PyInstaller cannot start a package's `__main__` directly).

With no arguments (a double click) it opens the window and hides the console it was given; with arguments
it is the command line (`fsr4.exe upscale clip.mp4`, `fsr4.exe ui photo.png`, ...).
"""
import ctypes
import os
import sys

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
    if len(sys.argv) == 1:
        hide_own_console()
        sys.exit(main(["ui"]))
    sys.exit(main())
