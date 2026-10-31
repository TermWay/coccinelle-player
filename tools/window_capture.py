"""Screenshot a Tk window through PrintWindow (works when the window is covered), saved as PNG.

Used by the UI checks in `tools/`; Windows only.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fsr4.ui_art import png_bytes  # noqa: E402

PW_RENDERFULLCONTENT = 2
GA_ROOT = 2
DWMWA_EXTENDED_FRAME_BOUNDS = 9

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
dwmapi = ctypes.windll.dwmapi
for function, restype, argtypes in (
        (user32.GetWindowDC, ctypes.c_void_p, [wintypes.HWND]),
        (user32.ReleaseDC, ctypes.c_int, [wintypes.HWND, ctypes.c_void_p]),
        (user32.PrintWindow, wintypes.BOOL, [wintypes.HWND, ctypes.c_void_p, wintypes.UINT]),
        (user32.GetWindowRect, wintypes.BOOL, [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]),
        (user32.WindowFromPoint, wintypes.HWND, [wintypes.POINT]),
        (user32.GetAncestor, wintypes.HWND, [wintypes.HWND, wintypes.UINT]),
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


def capture(window) -> np.ndarray:
    """RGBA pixels of a Tk toplevel, title bar included, without the invisible resize borders."""
    window.update()
    return capture_handle(int(window.wm_frame(), 16))


def capture_handle(handle: int) -> np.ndarray:
    """RGBA pixels of any top-level window (a native dialog, a popup list) by its HWND."""
    rect = wintypes.RECT()
    user32.GetWindowRect(handle, ctypes.byref(rect))
    bounds = wintypes.RECT()
    dwmapi.DwmGetWindowAttribute(handle, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(bounds), ctypes.sizeof(bounds))
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
    asks the window to draw itself and so hides areas that were never repainted. The window must be on top.
    `settle` lets Tk handle its pending events first; without it, the capture shows what is on screen this
    instant (paints Tk has not done yet included). Never during Windows' maximize or restore animation: the
    window is already in place for the check, but the screen there shows the animation and what is behind."""
    if settle:
        window.update()
    handle = int(window.wm_frame(), 16)
    bounds = wintypes.RECT()
    dwmapi.DwmGetWindowAttribute(handle, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(bounds), ctypes.sizeof(bounds))
    width, height = bounds.right - bounds.left, bounds.bottom - bounds.top
    points = [(bounds.left + 4 + (width - 8) * column // 8, bounds.top + 4 + (height - 8) * row // 8)
              for row in range(9) for column in range(9)]
    for x, y in points:
        under = user32.GetAncestor(user32.WindowFromPoint(wintypes.POINT(x, y)), GA_ROOT)
        if under != handle:
            # Another window covers it: the capture would show that window, which may be private.
            raise RuntimeError("the window is covered on screen; not capturing what is over it")
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
