"""Screenshot a Tk window through PrintWindow (works when the window is covered), saved as PNG.

Used by the UI checks in `tools/`; Windows only. Only this process's windows are captured: PrintWindow takes
one of its windows, and a capture of the screen refuses when another program's window overlaps it.
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fsr4.ui_art import png_bytes  # noqa: E402

PW_RENDERFULLCONTENT = 2
DWMWA_EXTENDED_FRAME_BOUNDS = 9
DWMWA_CLOAKED = 14

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
dwmapi = ctypes.windll.dwmapi
WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
for function, restype, argtypes in (
        (user32.GetWindowDC, ctypes.c_void_p, [wintypes.HWND]),
        (user32.ReleaseDC, ctypes.c_int, [wintypes.HWND, ctypes.c_void_p]),
        (user32.PrintWindow, wintypes.BOOL, [wintypes.HWND, ctypes.c_void_p, wintypes.UINT]),
        (user32.GetWindowRect, wintypes.BOOL, [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]),
        (user32.EnumWindows, wintypes.BOOL, [WNDENUMPROC, wintypes.LPARAM]),
        (user32.IsWindowVisible, wintypes.BOOL, [wintypes.HWND]),
        (user32.IsIconic, wintypes.BOOL, [wintypes.HWND]),
        (user32.GetWindowThreadProcessId, wintypes.DWORD, [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]),
        (user32.GetClassNameW, ctypes.c_int, [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]),
        # HRESULT as the result type raises OSError on failure.
        (dwmapi.DwmGetWindowAttribute, ctypes.HRESULT, [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p,
                                                        wintypes.DWORD]),
        (gdi32.CreateCompatibleDC, ctypes.c_void_p, [ctypes.c_void_p]),
        (gdi32.CreateCompatibleBitmap, ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
        (gdi32.BitBlt, wintypes.BOOL, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_int, ctypes.c_int, wintypes.DWORD]),
        (gdi32.SelectObject, ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p]),
        (gdi32.DeleteObject, wintypes.BOOL, [ctypes.c_void_p]),
        (gdi32.DeleteDC, wintypes.BOOL, [ctypes.c_void_p]),
        (gdi32.GetDIBits, ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT, wintypes.UINT,
                                         ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT])):
    function.restype = restype
    function.argtypes = argtypes


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]


def window_process(handle: int) -> int:
    """The id of the process that owns a window (0 when the window no longer exists)."""
    owner = wintypes.DWORD()
    user32.GetWindowThreadProcessId(handle, ctypes.byref(owner))
    return owner.value


def frame_bounds(handle: int) -> wintypes.RECT:
    """A window's bounds as drawn, without the invisible resize borders `GetWindowRect` adds around it."""
    bounds = wintypes.RECT()
    dwmapi.DwmGetWindowAttribute(handle, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(bounds), ctypes.sizeof(bounds))
    return bounds


def covering_window(handle: int) -> str | None:
    """The class name of another program's window that shows over `handle` on screen, or None. EnumWindows lists
    top-level windows from the top of the z-order, so every window listed before `handle` is above it."""
    above: list[int] = []
    found: list[int] = []

    def visit(other, _parameter):
        if other == handle:
            found.append(other)
            return False
        above.append(other)
        return True

    user32.EnumWindows(WNDENUMPROC(visit), 0)
    if not found:
        raise RuntimeError(f"window {handle:#x} is not a top-level window")
    bounds = frame_bounds(handle)
    own = os.getpid()
    for other in above:
        if not user32.IsWindowVisible(other) or user32.IsIconic(other) or window_process(other) in (own, 0):
            continue
        cloaked = wintypes.DWORD()
        dwmapi.DwmGetWindowAttribute(other, DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
        if cloaked.value:  # kept off screen: on another virtual desktop, or a shell window not shown
            continue
        box = frame_bounds(other)
        if box.left < bounds.right and bounds.left < box.right and box.top < bounds.bottom and bounds.top < box.bottom:
            name = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(other, name, 256)
            return name.value
    return None


def capture(window) -> np.ndarray:
    """RGBA pixels of a Tk toplevel, title bar included, without the invisible resize borders."""
    window.update()
    return capture_handle(int(window.wm_frame(), 16))


def capture_handle(handle: int) -> np.ndarray:
    """RGBA pixels of one of this process's top-level windows (a native dialog, a popup list) by its HWND."""
    if window_process(handle) != os.getpid():
        raise RuntimeError(f"window {handle:#x} belongs to another program; only this one's windows are captured")
    rect = wintypes.RECT()
    if not user32.GetWindowRect(handle, ctypes.byref(rect)):
        raise RuntimeError("GetWindowRect failed")
    bounds = frame_bounds(handle)
    width, height = rect.right - rect.left, rect.bottom - rect.top
    window_dc = user32.GetWindowDC(handle)
    memory_dc = gdi32.CreateCompatibleDC(window_dc)
    bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
    previous = gdi32.SelectObject(memory_dc, bitmap)
    try:
        if not user32.PrintWindow(handle, memory_dc, PW_RENDERFULLCONTENT):
            raise RuntimeError("PrintWindow failed")
        header = BITMAPINFOHEADER(biSize=ctypes.sizeof(BITMAPINFOHEADER), biWidth=width, biHeight=-height,
                                  biPlanes=1, biBitCount=32, biCompression=0)
        pixels = np.empty((height, width, 4), np.uint8)
        if gdi32.GetDIBits(memory_dc, bitmap, 0, height, pixels.ctypes.data, ctypes.byref(header), 0) != height:
            raise RuntimeError("GetDIBits failed")
    finally:
        gdi32.SelectObject(memory_dc, previous)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(handle, window_dc)
    left, top = bounds.left - rect.left, bounds.top - rect.top
    right, bottom = left + (bounds.right - bounds.left), top + (bounds.bottom - bounds.top)
    image = pixels[top:bottom, left:right][..., [2, 1, 0, 3]].copy()
    image[..., 3] = 255
    return image


def capture_screen(window, settle: bool = True) -> np.ndarray:
    """RGBA pixels of the screen where a Tk toplevel is: what is really displayed, unlike PrintWindow, which
    asks the window to draw itself and so hides areas that were never repainted. No other program's window may
    overlap it. `settle` lets Tk handle its pending events first; without it, the capture shows what is on
    screen this instant (paints Tk has not done yet included). Never during Windows' maximize or restore
    animation: the window is already in place for the check, but the screen there shows the animation and what
    is behind."""
    if settle:
        window.update()
    handle = int(window.wm_frame(), 16)
    covering = covering_window(handle)
    if covering is not None:
        # The capture would show that window, which may be private.
        raise RuntimeError(f"a window of another program ({covering}) overlaps the window; not capturing it")
    bounds = frame_bounds(handle)
    width, height = bounds.right - bounds.left, bounds.bottom - bounds.top
    screen_dc = user32.GetWindowDC(None)
    memory_dc = gdi32.CreateCompatibleDC(screen_dc)
    bitmap = gdi32.CreateCompatibleBitmap(screen_dc, width, height)
    previous = gdi32.SelectObject(memory_dc, bitmap)
    try:
        if not gdi32.BitBlt(memory_dc, 0, 0, width, height, screen_dc, bounds.left, bounds.top, 0x00CC0020):
            raise RuntimeError("BitBlt from the screen failed")
        header = BITMAPINFOHEADER(biSize=ctypes.sizeof(BITMAPINFOHEADER), biWidth=width, biHeight=-height,
                                  biPlanes=1, biBitCount=32, biCompression=0)
        pixels = np.empty((height, width, 4), np.uint8)
        if gdi32.GetDIBits(memory_dc, bitmap, 0, height, pixels.ctypes.data, ctypes.byref(header), 0) != height:
            raise RuntimeError("GetDIBits failed")
    finally:
        gdi32.SelectObject(memory_dc, previous)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(None, screen_dc)
    image = pixels[..., [2, 1, 0, 3]].copy()
    image[..., 3] = 255
    return image


def save(window, path: Path, screen: bool = False, settle: bool = True) -> None:
    """Save a Tk toplevel as PNG, drawn by itself (PrintWindow) or as the screen shows it (`screen`, see
    `capture_screen` for `settle`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(capture_screen(window, settle) if screen else capture(window)))


def save_handle(handle: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(capture_handle(handle)))
