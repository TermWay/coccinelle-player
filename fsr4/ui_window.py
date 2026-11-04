"""Full screen on the monitor the window is on, a repaint of everything after it, and a picture of the window that
hides a layout change (Windows).

Tk's `-fullscreen` sizes the window to the primary screen at 0,0 whatever monitor it is on, and rebuilds the
window frame on the way. Here the window keeps its frame: entering takes off the border styles and covers the
window's own monitor; leaving puts back the styles and the exact placement it had (position, size, maximized),
on the monitor it was on. Windows treats a borderless window that covers a monitor as full screen and hides
the taskbar while it is in front.
"""
from __future__ import annotations

import ctypes
import time
import tkinter as tk
from ctypes import wintypes

GWL_STYLE = -16
GCL_STYLE = -26
CS_VREDRAW, CS_HREDRAW = 0x1, 0x2
SW_RESTORE = 9
WS_OVERLAPPEDWINDOW = 0x00CF0000
MONITOR_DEFAULTTONEAREST = 2
MONITORINFOF_PRIMARY = 1
SWP_NOSIZE, SWP_NOMOVE, SWP_NOZORDER, SWP_FRAMECHANGED, SWP_NOOWNERZORDER = 0x1, 0x2, 0x4, 0x20, 0x200
RDW_INVALIDATE, RDW_ERASE, RDW_ALLCHILDREN, RDW_UPDATENOW, RDW_FRAME = 0x1, 0x4, 0x80, 0x100, 0x400
DCX_CACHE = 0x2
SRCCOPY = 0x00CC0020
WS_POPUP = 0x80000000
WS_EX_LAYERED, WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE = 0x80000, 0x20, 0x80, 0x08000000
SW_SHOWNOACTIVATE = 4
ULW_ALPHA = 2
CURTAIN_FADE_SECONDS = 0.12
CURTAIN_STEP_MS = 8


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD)]


class WINDOWPLACEMENT(ctypes.Structure):
    _fields_ = [("length", wintypes.UINT), ("flags", wintypes.UINT), ("showCmd", wintypes.UINT),
                ("ptMinPosition", wintypes.POINT), ("ptMaxPosition", wintypes.POINT),
                ("rcNormalPosition", wintypes.RECT)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_ubyte), ("BlendFlags", ctypes.c_ubyte), ("SourceConstantAlpha", ctypes.c_ubyte),
                ("AlphaFormat", ctypes.c_ubyte)]


_MonitorEnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT),
                                      wintypes.LPARAM)
user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
dwmapi = ctypes.WinDLL("dwmapi")
for _function, _restype, _argtypes in (
        (user32.MonitorFromWindow, wintypes.HMONITOR, [wintypes.HWND, wintypes.DWORD]),
        (user32.MonitorFromPoint, wintypes.HMONITOR, [wintypes.POINT, wintypes.DWORD]),
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
        (user32.GetClassLongPtrW, ctypes.c_size_t, [wintypes.HWND, ctypes.c_int]),
        (user32.SetClassLongPtrW, ctypes.c_size_t, [wintypes.HWND, ctypes.c_int, ctypes.c_size_t]),
        (user32.GetDCEx, wintypes.HDC, [wintypes.HWND, wintypes.HRGN, wintypes.DWORD]),
        (user32.ReleaseDC, ctypes.c_int, [wintypes.HWND, wintypes.HDC]),
        (user32.CreateWindowExW, wintypes.HWND, [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                                 wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]),
        (user32.DestroyWindow, wintypes.BOOL, [wintypes.HWND]),
        (user32.UpdateLayeredWindow, wintypes.BOOL, [wintypes.HWND, wintypes.HDC, ctypes.POINTER(wintypes.POINT),
                                                     ctypes.POINTER(wintypes.SIZE), wintypes.HDC,
                                                     ctypes.POINTER(wintypes.POINT), wintypes.COLORREF,
                                                     ctypes.POINTER(BLENDFUNCTION), wintypes.DWORD]),
        (gdi32.CreateCompatibleDC, wintypes.HDC, [wintypes.HDC]),
        (gdi32.CreateCompatibleBitmap, wintypes.HBITMAP, [wintypes.HDC, ctypes.c_int, ctypes.c_int]),
        (gdi32.SelectObject, wintypes.HGDIOBJ, [wintypes.HDC, wintypes.HGDIOBJ]),
        (gdi32.DeleteObject, wintypes.BOOL, [wintypes.HGDIOBJ]),
        (gdi32.DeleteDC, wintypes.BOOL, [wintypes.HDC]),
        (gdi32.BitBlt, wintypes.BOOL, [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD]),
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


def work_area_at(x: int, y: int) -> tuple[int, int, int, int]:
    """(left, top, width, height) of what the taskbar leaves free on the monitor at the desktop point (x, y), or on
    the nearest monitor."""
    info = MONITORINFO(cbSize=ctypes.sizeof(MONITORINFO))
    monitor = user32.MonitorFromPoint(wintypes.POINT(x, y), MONITOR_DEFAULTTONEAREST)
    _check(user32.GetMonitorInfoW(monitor, ctypes.byref(info)), "read the monitor at a point")
    return _rect(info.rcWork)


def keep_contents_on_resize(window: tk.Misc) -> None:
    """Stop Windows from invalidating a window and all its children whenever it changes size.

    Tk registers its window classes with CS_HREDRAW and CS_VREDRAW, so one size change of the window repainted every
    widget once per level of nesting (228 Expose events, 290 ms). Tk does not need them: widgets redraw on their own
    Configure and Windows still sends what a bigger window uncovers. A class is shared, so this covers all of Tk's
    windows in this process: the child class through `window`, the frame class through its frame window, which Tk
    only creates when `window` is first mapped.
    """
    def drop_styles() -> None:
        for handle in (window.winfo_id(), frame_handle(window)):
            style = user32.GetClassLongPtrW(handle, GCL_STYLE)
            if style & (CS_HREDRAW | CS_VREDRAW):
                _check(user32.SetClassLongPtrW(handle, GCL_STYLE, style & ~(CS_HREDRAW | CS_VREDRAW)),
                       "change a window class")

    window.bind("<Map>", lambda event: drop_styles() if event.widget is window else None, add="+")
    if window.winfo_ismapped():
        drop_styles()


def repaint(window: tk.Misc) -> None:
    """Have Windows repaint every part of `window` now (frame and child windows included)."""
    user32.RedrawWindow(frame_handle(window), None, None,
                        RDW_INVALIDATE | RDW_ERASE | RDW_FRAME | RDW_ALLCHILDREN | RDW_UPDATENOW)


class Curtain:
    """A picture of `window` held in front of it while its layout changes, then faded away.

    Tk moves a parent window first and its children in later passes, and Windows shows every stage: when the sidebar
    folds, what is anchored to the right rides with its parent 340 px away for 5 frames, then jumps back. Painting
    cannot be held back instead: Tk draws on idle, after any WM_SETREDRAW would have ended. Behind the picture, the
    stages are never seen and the old layout dissolves into the finished one.

    Create it on Tk's thread before the change and call `fade` once the change is drawn. It lets clicks through.
    """

    def __init__(self, window: tk.Misc):
        self.window = window
        self.closed = False
        self._place = wintypes.POINT(window.winfo_rootx(), window.winfo_rooty())
        self._size = wintypes.SIZE(window.winfo_width(), window.winfo_height())
        self._origin = wintypes.POINT(0, 0)
        self._blend = BLENDFUNCTION(BlendOp=0, BlendFlags=0, SourceConstantAlpha=255, AlphaFormat=0)
        source = window.winfo_id()
        # Without DCX_CLIPCHILDREN: the child windows' pixels are what this picture is for.
        window_dc = user32.GetDCEx(source, None, DCX_CACHE)
        _check(window_dc, "read the window's pixels")
        self._picture_dc = gdi32.CreateCompatibleDC(window_dc)
        self._bitmap = gdi32.CreateCompatibleBitmap(window_dc, self._size.cx, self._size.cy)
        self._previous = gdi32.SelectObject(self._picture_dc, self._bitmap)
        copied = gdi32.BitBlt(self._picture_dc, 0, 0, self._size.cx, self._size.cy, window_dc, 0, 0, SRCCOPY)
        user32.ReleaseDC(source, window_dc)
        _check(copied, "copy the window's pixels")
        self._handle = user32.CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE, "Static", None, WS_POPUP,
            self._place.x, self._place.y, self._size.cx, self._size.cy, frame_handle(window), None, None, None)
        _check(self._handle, "create the window's curtain")
        self._show(255)
        user32.ShowWindow(self._handle, SW_SHOWNOACTIVATE)
        # Once this returns the picture is on the screen, so no stage of the change can come before it.
        dwmapi.DwmFlush()

    def _show(self, alpha: int) -> None:
        self._blend.SourceConstantAlpha = alpha
        _check(user32.UpdateLayeredWindow(self._handle, None, ctypes.byref(self._place), ctypes.byref(self._size),
                                          self._picture_dc, ctypes.byref(self._origin), 0, ctypes.byref(self._blend),
                                          ULW_ALPHA), "show the window's curtain")

    def _close(self) -> None:
        user32.DestroyWindow(self._handle)
        gdi32.SelectObject(self._picture_dc, self._previous)
        gdi32.DeleteObject(self._bitmap)
        gdi32.DeleteDC(self._picture_dc)
        self.closed = True

    def fade(self) -> None:
        """Dissolve the picture into the window, on Tk's timers, and free it."""
        start = time.perf_counter()

        def step() -> None:
            left = 1 - (time.perf_counter() - start) / CURTAIN_FADE_SECONDS
            if left <= 0:
                self._close()
                return
            self._show(round(255 * left))
            self.window.after(CURTAIN_STEP_MS, step)

        step()


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
