"""Files dropped from Explorer onto a Tk window (Windows WM_DROPFILES; no drag-and-drop package needed).

The window procedure of the toplevel's client window (`winfo_id`, which exists from creation, unlike the outer
frame Tk adds when the window is first shown) is subclassed: WM_DROPFILES is answered here, every other message
goes to Tk's own procedure. Drops on any widget reach it, as the first ancestor that accepts files.
"""
from __future__ import annotations

import ctypes
import os
import tkinter as tk
from ctypes import wintypes
from typing import Callable

from . import log

WM_DROPFILES = 0x0233
GWLP_WNDPROC = -4
QUERY_COUNT = 0xFFFFFFFF

if os.name == "nt":
    LRESULT = ctypes.c_ssize_t
    WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
    _user32 = ctypes.windll.user32
    _shell32 = ctypes.windll.shell32
    _user32.SetWindowLongPtrW.restype = ctypes.c_void_p
    _user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
    _user32.CallWindowProcW.restype = LRESULT
    _user32.CallWindowProcW.argtypes = [ctypes.c_void_p, wintypes.HWND, wintypes.UINT, wintypes.WPARAM,
                                        wintypes.LPARAM]
    _shell32.DragAcceptFiles.argtypes = [wintypes.HWND, wintypes.BOOL]
    _shell32.DragQueryFileW.restype = wintypes.UINT
    _shell32.DragQueryFileW.argtypes = [ctypes.c_void_p, wintypes.UINT, ctypes.c_wchar_p, wintypes.UINT]
    _shell32.DragFinish.argtypes = [ctypes.c_void_p]


def dropped_paths(drop_handle: int) -> list[str]:
    """The file paths held by an HDROP."""
    paths = []
    for index in range(_shell32.DragQueryFileW(drop_handle, QUERY_COUNT, None, 0)):
        length = _shell32.DragQueryFileW(drop_handle, index, None, 0)
        buffer = ctypes.create_unicode_buffer(length + 1)
        _shell32.DragQueryFileW(drop_handle, index, buffer, length + 1)
        paths.append(buffer.value)
    return paths


class FileDrop:
    """Calls `on_drop(paths)` (on the Tk thread) when files are dropped on `window`. Windows only."""

    def __init__(self, window: tk.Misc, on_drop: Callable[[list[str]], None]):
        self.on_drop = on_drop
        self.handle = window.winfo_id()
        self._procedure = WNDPROC(self._window_procedure)
        self._previous = _user32.SetWindowLongPtrW(self.handle, GWLP_WNDPROC,
                                                   ctypes.cast(self._procedure, ctypes.c_void_p))
        if not self._previous:
            raise OSError(f"could not subclass the window for file drops (error {ctypes.GetLastError()})")
        _shell32.DragAcceptFiles(self.handle, True)

    def _window_procedure(self, handle, message, wparam, lparam):
        if message == WM_DROPFILES:
            try:
                paths = dropped_paths(wparam)
                self.on_drop(paths)
            except Exception as error:  # noqa: BLE001 - an exception must not cross into Windows
                log.emit(f"[ui] file drop failed: {error}")
            finally:
                _shell32.DragFinish(wparam)
            return 0
        return _user32.CallWindowProcW(self._previous, handle, message, wparam, lparam)

    def close(self) -> None:
        """Stop accepting files and give the window its own procedure back."""
        if self._previous:
            _shell32.DragAcceptFiles(self.handle, False)
            _user32.SetWindowLongPtrW(self.handle, GWLP_WNDPROC, self._previous)
            self._previous = None

