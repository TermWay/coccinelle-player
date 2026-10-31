"""Full screen on the monitor the window is on, and a repaint of everything after it (Windows).

Tk's `-fullscreen` sizes the window to the primary screen at 0,0 whatever monitor it is on, and rebuilds the
window frame on the way. Here the window keeps its frame: entering takes off the border styles and covers the
window's own monitor; leaving puts back the styles and the exact placement it had (position, size, maximized),
on the monitor it was on. Windows treats a borderless window that covers a monitor as full screen and hides
the taskbar while it is in front.
"""
from __future__ import annotations

import ctypes
import tkinter as tk
from ctypes import wintypes

GWL_STYLE = -16
SW_RESTORE = 9
WS_OVERLAPPEDWINDOW = 0x00CF0000
MONITOR_DEFAULTTONEAREST = 2
MONITORINFOF_PRIMARY = 1
SWP_NOSIZE, SWP_NOMOVE, SWP_NOZORDER, SWP_FRAMECHANGED, SWP_NOOWNERZORDER = 0x1, 0x2, 0x4, 0x20, 0x200
RDW_INVALIDATE, RDW_ERASE, RDW_ALLCHILDREN, RDW_UPDATENOW, RDW_FRAME = 0x1, 0x4, 0x80, 0x100, 0x400


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD)]


class WINDOWPLACEMENT(ctypes.Structure):
    _fields_ = [("length", wintypes.UINT), ("flags", wintypes.UINT), ("showCmd", wintypes.UINT),
                ("ptMinPosition", wintypes.POINT), ("ptMaxPosition", wintypes.POINT),
                ("rcNormalPosition", wintypes.RECT)]


_MonitorEnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT),
                                      wintypes.LPARAM)
user32 = ctypes.WinDLL("user32", use_last_error=True)
for _function, _restype, _argtypes in (
        (user32.MonitorFromWindow, wintypes.HMONITOR, [wintypes.HWND, wintypes.DWORD]),
        (user32.GetMonitorInfoW, wintypes.BOOL, [wintypes.HMONITOR, ctypes.POINTER(MONITORINFO)]),
        (user32.EnumDisplayMonitors, wintypes.BOOL, [wintypes.HDC, ctypes.POINTER(wintypes.RECT), _MonitorEnumProc,
                                                     wintypes.LPARAM]),
        (user32.GetWindowLongW, ctypes.c_long, [wintypes.HWND, ctypes.c_int]),
        (user32.SetWindowLongW, ctypes.c_long, [wintypes.HWND, ctypes.c_int, ctypes.c_long]),
        (user32.GetWindowPlacement, wintypes.BOOL, [wintypes.HWND, ctypes.POINTER(WINDOWPLACEMENT)]),
        (user32.SetWindowPlacement, wintypes.BOOL, [wintypes.HWND, ctypes.POINTER(WINDOWPLACEMENT)]),
        (user32.SetWindowPos, wintypes.BOOL, [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int, wintypes.UINT]),
        (user32.RedrawWindow, wintypes.BOOL, [wintypes.HWND, ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]),
        (user32.IsZoomed, wintypes.BOOL, [wintypes.HWND]),
        (user32.ShowWindow, wintypes.BOOL, [wintypes.HWND, ctypes.c_int])):
    _function.restype = _restype
    _function.argtypes = _argtypes


def _check(succeeded: int, what: str) -> None:
    if not succeeded:
        raise ctypes.WinError(ctypes.get_last_error(), f"cannot {what}")


def _rect(rect: wintypes.RECT) -> tuple[int, int, int, int]:
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top


def monitors() -> list[tuple[int, int, int, int, bool]]:
    """Every monitor as (left, top, width, height, primary), in desktop pixels."""
    found = []

    def add(handle, _dc, _rect_pointer, _data):
        info = MONITORINFO(cbSize=ctypes.sizeof(MONITORINFO))
        _check(user32.GetMonitorInfoW(handle, ctypes.byref(info)), "read a monitor's size")
        found.append(_rect(info.rcMonitor) + (bool(info.dwFlags & MONITORINFOF_PRIMARY),))
        return True

    _check(user32.EnumDisplayMonitors(None, None, _MonitorEnumProc(add), 0), "list the monitors")
    return found


def frame_handle(window: tk.Misc) -> int:
    """The Windows handle of a Tk toplevel's frame window."""
    return int(window.wm_frame(), 16)


def monitor_of(window: tk.Misc) -> tuple[int, int, int, int]:
    """(left, top, width, height) of the monitor that holds most of `window`."""
    info = MONITORINFO(cbSize=ctypes.sizeof(MONITORINFO))
    monitor = user32.MonitorFromWindow(frame_handle(window), MONITOR_DEFAULTTONEAREST)
    _check(user32.GetMonitorInfoW(monitor, ctypes.byref(info)), "read the window's monitor")
    return _rect(info.rcMonitor)


def repaint(window: tk.Misc) -> None:
    """Have Windows repaint every part of `window` now (frame and child windows included)."""
    user32.RedrawWindow(frame_handle(window), None, None,
                        RDW_INVALIDATE | RDW_ERASE | RDW_FRAME | RDW_ALLCHILDREN | RDW_UPDATENOW)


class Fullscreen:
    """Borderless full screen over the monitor `window` is on; `leave` restores the window as it was."""

    def __init__(self, window: tk.Misc):
        self.window = window
        self._style: int | None = None
        self._placement: WINDOWPLACEMENT | None = None

    @property
    def active(self) -> bool:
        return self._style is not None

    def enter(self) -> None:
        if self.active:
            return
        handle = frame_handle(self.window)
        placement = WINDOWPLACEMENT(length=ctypes.sizeof(WINDOWPLACEMENT))
        _check(user32.GetWindowPlacement(handle, ctypes.byref(placement)), "read the window placement")
        left, top, width, height = monitor_of(self.window)
        if user32.IsZoomed(handle):
            # Windows keeps the taskbar over a maximized window: restore it first (leaving maximizes it again).
            user32.ShowWindow(handle, SW_RESTORE)
        style = user32.GetWindowLongW(handle, GWL_STYLE)
        user32.SetWindowLongW(handle, GWL_STYLE, style & ~WS_OVERLAPPEDWINDOW)
        self._style, self._placement = style, placement
        if not user32.SetWindowPos(handle, None, left, top, width, height, SWP_NOOWNERZORDER | SWP_FRAMECHANGED):
            error = ctypes.WinError(ctypes.get_last_error(), "cannot cover the monitor")
            self.leave()
            raise error

    def leave(self) -> None:
        if not self.active:
            return
        handle = frame_handle(self.window)
        user32.SetWindowLongW(handle, GWL_STYLE, self._style)
        _check(user32.SetWindowPlacement(handle, ctypes.byref(self._placement)), "restore the window placement")
        _check(user32.SetWindowPos(handle, None, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER
                                   | SWP_NOOWNERZORDER | SWP_FRAMECHANGED), "restore the window frame")
        self._style = self._placement = None
