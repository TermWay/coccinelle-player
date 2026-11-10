"""The preview area, drawn with GDI so video plays smoothly at window and full screen sizes.

Tk's image path takes 20 ms for a 1600x900 frame and 120 ms at 4K; blitting a numpy BGRA buffer with GDI's
SetDIBitsToDevice takes 0.5-2 ms. So the view is a Tk frame without a background that this module paints:
every state (drop target, message, comparison of a still frame or of playing video) is composed into one
BGRA canvas with numpy (picture, divider, labels, minimap, spinner) and blitted. Composition runs on
whichever thread presents (the Tk thread for interaction, the player thread for playback) under one lock.

Each side of the divider shows a layer: an upscaled version (the window names them) or the source
enlarged by a filter (original pixels, bilinear, bicubic or Lanczos), all at the same
geometry; clicking a side's label tells the window, which opens that side's dropdown. The divider's handle goes to
the height where the divider is pressed or dragged. A side can be OFF: the other one fills the picture and there is no divider.
Zoom is Fit, Original
size (one screen pixel per source pixel, the upscaled side shrunk to match) or any scale up to 800%; from
100% up every output pixel is a block of screen pixels (nearest neighbour), so what is drawn is what the file
holds. The right or middle button pans, the wheel zooms around the cursor in quarter octaves (stopping at
Fit, Original size and 100% to 800%), double-click switches between Fit and 100%, and the minimap moves the
visible area.
"""
from __future__ import annotations

import ctypes
import functools
import math
import threading
import tkinter as tk
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from . import log
from .ui_art import Sketch
from .ui_theme import Theme

ZOOM_LEVELS: tuple = ("fit", "source", 1, 2, 4, 8)
MAX_ZOOM = 8.0
WHEEL_STEP = 2 ** 0.25
SNAP_TOLERANCE = 0.005
REFERENCE_FILTERS = {
    "nearest": ("Original pixels", cv2.INTER_NEAREST),
    "linear": ("Bilinear", cv2.INTER_LINEAR),
    "cubic": ("Bicubic", cv2.INTER_CUBIC),
    "lanczos": ("Lanczos", cv2.INTER_LANCZOS4),
}
MINIMAP_WIDTH = 180
MINIMAP_MIN = 8
SPINNER_FRAMES = 24
DIVIDER_COLOR = (0xf5, 0xf4, 0xf4)
OFF = "off"
PLACEHOLDER_DIM = 0.45


def hex_to_bgr(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    red, green, blue = (int(color[index:index + 2], 16) for index in (0, 2, 4))
    return blue, green, red


# ---- GDI -----------------------------------------------------------------------------------------------


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]


def _bitmap_header(width: int, height: int) -> BITMAPINFOHEADER:
    """A top-down 32-bit BGRX bitmap header."""
    return BITMAPINFOHEADER(biSize=ctypes.sizeof(BITMAPINFOHEADER), biWidth=width, biHeight=-height, biPlanes=1,
                            biBitCount=32, biCompression=0)


class Gdi:
    """The few user32/gdi32 calls the view needs: blit a BGRA buffer, render text to a coverage mask."""

    def __init__(self):
        user32, gdi32 = ctypes.WinDLL("user32"), ctypes.WinDLL("gdi32")
        handle = ctypes.c_void_p
        user32.GetDC.restype = handle
        user32.GetDC.argtypes = [handle]
        user32.ReleaseDC.argtypes = [handle, handle]
        gdi32.SetDIBitsToDevice.argtypes = [handle, ctypes.c_int, ctypes.c_int, wintypes.DWORD, wintypes.DWORD,
                                            ctypes.c_int, ctypes.c_int, wintypes.UINT, wintypes.UINT, ctypes.c_void_p,
                                            ctypes.c_void_p, wintypes.UINT]
        gdi32.CreateCompatibleDC.restype = handle
        gdi32.CreateCompatibleDC.argtypes = [handle]
        gdi32.CreateDIBSection.restype = handle
        gdi32.CreateDIBSection.argtypes = [handle, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p),
                                           handle, wintypes.DWORD]
        gdi32.SelectObject.restype = handle
        gdi32.SelectObject.argtypes = [handle, handle]
        gdi32.DeleteObject.argtypes = [handle]
        gdi32.DeleteDC.argtypes = [handle]
        gdi32.CreateFontW.restype = handle
        gdi32.CreateFontW.argtypes = [ctypes.c_int] * 5 + [wintypes.DWORD] * 8 + [wintypes.LPCWSTR]
        gdi32.SetTextColor.argtypes = [handle, wintypes.DWORD]
        gdi32.SetBkMode.argtypes = [handle, ctypes.c_int]
        gdi32.TextOutW.argtypes = [handle, ctypes.c_int, ctypes.c_int, wintypes.LPCWSTR, ctypes.c_int]
        gdi32.GetTextExtentPoint32W.argtypes = [handle, wintypes.LPCWSTR, ctypes.c_int, ctypes.POINTER(wintypes.SIZE)]
        self.user32, self.gdi32 = user32, gdi32

    def blit(self, window: int, canvas: np.ndarray) -> bool:
        """Copy a (height, width, 4) BGRA canvas to the window's top-left corner; False when GDI refused."""
        height, width = canvas.shape[:2]
        header = _bitmap_header(width, height)
        dc = self.user32.GetDC(window)
        if not dc:
            return False
        try:
            lines = self.gdi32.SetDIBitsToDevice(dc, 0, 0, width, height, 0, 0, 0, height, canvas.ctypes.data,
                                                 ctypes.byref(header), 0)
        finally:
            self.user32.ReleaseDC(window, dc)
        return lines != 0

    def text_coverage(self, text: str, family: str, pixel_height: int) -> np.ndarray:
        """Grayscale anti-aliased coverage (0..1 float32) of `text` in `family` at `pixel_height` em size."""
        gdi32 = self.gdi32
        font = gdi32.CreateFontW(-pixel_height, 0, 0, 0, 400, 0, 0, 0, 1, 0, 0, 4, 0, family)
        dc = gdi32.CreateCompatibleDC(None)
        try:
            if not font or not dc:
                raise OSError("GDI could not create a font or a drawing context")
            previous_font = gdi32.SelectObject(dc, font)
            size = wintypes.SIZE()
            gdi32.GetTextExtentPoint32W(dc, text, len(text), ctypes.byref(size))
            width, height = max(1, size.cx + 2), max(1, size.cy)
            bits = ctypes.c_void_p()
            bitmap = gdi32.CreateDIBSection(dc, ctypes.byref(_bitmap_header(width, height)), 0, ctypes.byref(bits),
                                            None, 0)
            if not bitmap:
                gdi32.SelectObject(dc, previous_font)
                raise OSError("GDI could not create a text bitmap")
            previous_bitmap = gdi32.SelectObject(dc, bitmap)
            ctypes.memset(bits, 0, width * height * 4)
            gdi32.SetTextColor(dc, 0x00FFFFFF)
            gdi32.SetBkMode(dc, 1)
            gdi32.TextOutW(dc, 1, 0, text, len(text))
            gdi32.GdiFlush()
            data = np.frombuffer(ctypes.string_at(bits, width * height * 4), np.uint8).reshape(height, width, 4)
            coverage = data[..., 1].astype(np.float32) / 255
            gdi32.SelectObject(dc, previous_bitmap)
            gdi32.DeleteObject(bitmap)
            gdi32.SelectObject(dc, previous_font)
            return coverage
        finally:
            if dc:
                gdi32.DeleteDC(dc)  # first: a font still selected into a DC cannot be deleted
            if font:
                gdi32.DeleteObject(font)


_gdi: Gdi | None = None


def gdi() -> Gdi:
    global _gdi
    if _gdi is None:
        _gdi = Gdi()
    return _gdi


# ---- compositing helpers -------------------------------------------------------------------------------


def blend(canvas: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
    """Draw a premultiplied BGRA sprite over the canvas at (x, y), clipped to the canvas."""
    height, width = sprite.shape[:2]
    canvas_height, canvas_width = canvas.shape[:2]
    left, top = max(0, x), max(0, y)
    right, bottom = min(canvas_width, x + width), min(canvas_height, y + height)
    if right <= left or bottom <= top:
        return
    source = sprite[top - y:bottom - y, left - x:right - x]
    target = canvas[top:bottom, left:right, :3]
    inverse = 255 - source[..., 3:4].astype(np.uint16)
    target[...] = (source[..., :3] + (target.astype(np.uint16) * inverse + 127) // 255).astype(np.uint8)


def blend_outline(canvas: np.ndarray, piece: np.ndarray, x: int, y: int, width: int, height: int) -> None:
    """Draw a rounded rectangle outline from `piece`, a premultiplied BGRA (2k+1)-pixel square outline: its
    corners as they are, its middle row and column repeated along the edges; the inside is left untouched."""
    corner = piece.shape[0] // 2
    if width <= 2 * corner or height <= 2 * corner:
        return
    right, bottom = x + width - corner, y + height - corner
    blend(canvas, piece[:corner, :corner], x, y)
    blend(canvas, piece[:corner, corner + 1:], right, y)
    blend(canvas, piece[corner + 1:, :corner], x, bottom)
    blend(canvas, piece[corner + 1:, corner + 1:], right, bottom)
    across, down = width - 2 * corner, height - 2 * corner
    blend(canvas, np.broadcast_to(piece[:corner, corner:corner + 1], (corner, across, 4)), x + corner, y)
    blend(canvas, np.broadcast_to(piece[corner + 1:, corner:corner + 1], (corner, across, 4)), x + corner, bottom)
    blend(canvas, np.broadcast_to(piece[corner:corner + 1, :corner], (down, corner, 4)), x, y + corner)
    blend(canvas, np.broadcast_to(piece[corner:corner + 1, corner + 1:], (down, corner, 4)), right, y + corner)


def over(target: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
    """Composite a premultiplied BGRA sprite over a premultiplied BGRA sprite (alpha included)."""
    height, width = sprite.shape[:2]
    region = target[y:y + height, x:x + width]
    source = sprite[:region.shape[0], :region.shape[1]]
    inverse = 255 - source[..., 3:4].astype(np.uint16)
    region[...] = (source + (region.astype(np.uint16) * inverse + 127) // 255).astype(np.uint8)


def fill(canvas: np.ndarray, left: int, top: int, right: int, bottom: int, color: tuple[int, int, int]) -> None:
    height, width = canvas.shape[:2]
    left, top, right, bottom = max(0, left), max(0, top), min(width, right), min(height, bottom)
    if right > left and bottom > top:
        canvas[top:bottom, left:right, :3] = color


def tinted(coverage: np.ndarray, color: tuple[int, int, int], opacity: float = 1.0) -> np.ndarray:
    """A premultiplied BGRA sprite of `color` through `coverage`."""
    alpha = coverage * opacity
    sprite = np.empty(coverage.shape + (4,), np.uint8)
    for channel in range(3):
        sprite[..., channel] = (alpha * color[channel] + 0.5).astype(np.uint8)
    sprite[..., 3] = (alpha * 255 + 0.5).astype(np.uint8)
    return sprite


# ---- geometry ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """Where the output picture sits in the view. `region` is the part of the output picture shown, in output
    pixels; `shown` is the view rectangle it covers (left, top, width, height); `scale` is view pixels per
    output pixel."""
    view_size: tuple[int, int]
    output_size: tuple[int, int]
    fit: bool
    scale: float
    region: tuple[int, int, int, int]
    shown: tuple[int, int, int, int]

    @property
    def key(self) -> tuple:
        return self.view_size, self.output_size, self.fit, self.scale, self.region, self.shown


def minimap_box(view_size: tuple[int, int], output_size: tuple[int, int],
                px: Callable[[float], int]) -> tuple[int, int, int, int] | None:
    """Where the minimap goes (left, top, right, bottom): bottom right, at most MINIMAP_WIDTH wide and a third of
    the view, and within the view's height (a tall picture in a short view gets a narrower map). None when the
    view has no room for a usable one."""
    width, height = view_size
    out_width, out_height = output_size
    margin = px(12)
    right, bottom = width - margin, height - margin
    map_width = min(px(MINIMAP_WIDTH), width // 3)
    map_height = max(1, int(map_width * out_height / out_width))
    room = bottom - margin
    if map_height > room:
        map_width, map_height = int(map_width * room / map_height), room
    if map_width < MINIMAP_MIN or map_height < MINIMAP_MIN or right - map_width < 2:
        return None
    return right - map_width, bottom - map_height, right, bottom


def compute_geometry(view_size: tuple[int, int], output_size: tuple[int, int], zoom,
                     center: tuple[float, float] | None, source_size: tuple[int, int] | None = None) -> Geometry:
    """Fit the whole picture, show it at the source's own size ("source"), or show `zoom` (any positive number)
    view pixels per output pixel, around `center` (output pixels)."""
    view_width, view_height = view_size
    out_width, out_height = output_size
    if zoom == "source":
        source_width, source_height = source_size or output_size
        ratio_x, ratio_y = out_width / source_width, out_height / source_height
        shown_width, shown_height = min(view_width, source_width), min(view_height, source_height)
        region_width = min(out_width, int(round(shown_width * ratio_x)))
        region_height = min(out_height, int(round(shown_height * ratio_y)))
        center_x, center_y = center or (out_width / 2, out_height / 2)
        # On the source's pixel grid, so the original side is an exact copy of its pixels.
        x0 = int(round(round(min(max(center_x - region_width / 2, 0), out_width - region_width) / ratio_x) * ratio_x))
        y0 = int(round(round(min(max(center_y - region_height / 2, 0), out_height - region_height) / ratio_y)
                       * ratio_y))
        x0, y0 = min(x0, out_width - region_width), min(y0, out_height - region_height)
        return Geometry(view_size, output_size, False, shown_width / region_width, (x0, y0, region_width, region_height),
                        ((view_width - shown_width) // 2, (view_height - shown_height) // 2, shown_width, shown_height))
    if zoom == "fit":
        scale = min(view_width / out_width, view_height / out_height)
        shown_width = max(1, min(view_width, int(round(out_width * scale))))
        shown_height = max(1, min(view_height, int(round(out_height * scale))))
        return Geometry(view_size, output_size, True, scale, (0, 0, out_width, out_height),
                        ((view_width - shown_width) // 2, (view_height - shown_height) // 2, shown_width, shown_height))
    region_width = min(out_width, math.ceil(view_width / zoom))
    region_height = min(out_height, math.ceil(view_height / zoom))
    center_x, center_y = center or (out_width / 2, out_height / 2)
    x0 = int(round(min(max(center_x - region_width / 2, 0), out_width - region_width)))
    y0 = int(round(min(max(center_y - region_height / 2, 0), out_height - region_height)))
    shown_width = min(view_width, int(round(region_width * zoom)))
    shown_height = min(view_height, int(round(region_height * zoom)))
    return Geometry(view_size, output_size, False, float(zoom), (x0, y0, region_width, region_height),
                    ((view_width - shown_width) // 2, (view_height - shown_height) // 2, shown_width, shown_height))


def step_zoom(scale: float, steps: int, stops: list[float], low: float, high: float) -> float:
    """The scale `steps` wheel notches from `scale` (positive zooms in): quarter octaves, stopping at any of
    `stops` on the way, kept within `low`..`high`."""
    for _ in range(abs(steps)):
        if steps > 0:
            target = scale * WHEEL_STEP
            passed = [stop for stop in stops if scale * (1 + SNAP_TOLERANCE) < stop <= target]
            scale = min(passed) if passed else target
        else:
            target = scale / WHEEL_STEP
            passed = [stop for stop in stops if target <= stop < scale * (1 - SNAP_TOLERANCE)]
            scale = max(passed) if passed else target
        scale = min(high, max(low, scale))
    return scale


def reference_shrinks(geometry: Geometry, source_size: tuple[int, int]) -> bool:
    """True when the source is shown smaller than its own pixels (the filter choice does not apply)."""
    return geometry.fit and geometry.shown[2] <= source_size[0] and geometry.shown[3] <= source_size[1]


@functools.lru_cache(maxsize=6)
def block_maps(scale: float, first: int, columns: int, rows: int) -> tuple[np.ndarray, int, tuple[int, int]]:
    """A cv2.remap table (int16 pairs, 4 bytes per view pixel) that draws output pixels as blocks of view pixels
    at `scale` (1 or more): view column `first + j` shows output column floor((first + j + 0.5) / scale) of the
    region, view row i output row floor((i + 0.5) / scale). Returns (table, first output column, output
    (columns, rows) used); the table counts columns from that first one. Shared: never write to it."""
    column_index = np.floor((np.arange(first, first + columns) + 0.5) / scale).astype(np.int16)
    start = int(column_index[0])
    row_index = np.floor((np.arange(rows) + 0.5) / scale).astype(np.int16)
    table = np.empty((rows, columns, 2), np.int16)
    table[..., 0] = column_index - start
    table[..., 1] = row_index[:, None]
    return table, start, (int(column_index[-1]) - start + 1, int(row_index[-1]) + 1)


def block_view(region: np.ndarray, scale: float, first: int, columns: int, rows: int) -> np.ndarray:
    """View columns `first`..`first + columns` (and `rows` rows) of `region`, output pixels from the shown
    region's top-left corner, each drawn as a block at `scale` (1 or more)."""
    table, start, _ = block_maps(scale, first, columns, rows)
    return cv2.remap(region[:, start:], table, None, cv2.INTER_NEAREST)


def reference_view(source: np.ndarray, geometry: Geometry, interpolation: int, columns: int,
                   first: int = 0) -> np.ndarray:
    """View columns `first` to `first + columns` of the source drawn at `geometry` (BGRA), enlarged with
    `interpolation`.

    Pixel centres line up with the upscaled picture: view pixel i shows output position (i + 0.5) / scale,
    which is source position (that * source / output) - 0.5, as OpenCV's resize maps them.
    """
    source_height, source_width = source.shape[:2]
    _, _, shown_width, shown_height = geometry.shown
    first = max(0, min(first, shown_width - 1))
    columns = max(1, min(columns, shown_width - first))
    if reference_shrinks(geometry, (source_width, source_height)):
        shrunk = cv2.resize(source, (shown_width, shown_height), interpolation=cv2.INTER_AREA)
        return shrunk[:, first:first + columns]
    if geometry.fit:
        step_x, step_y = source_width / shown_width, source_height / shown_height
        matrix = np.array([[step_x, 0, (first + 0.5) * step_x - 0.5], [0, step_y, 0.5 * step_y - 0.5]], np.float32)
        return cv2.warpAffine(source, matrix, (columns, shown_height), flags=interpolation | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
    out_width, out_height = geometry.output_size
    x0, y0, region_width, region_height = geometry.region
    if geometry.scale < 1:
        # Shrunk region (Original size): the same mapping as resizing the region to the shown size.
        step_x = source_width / out_width * region_width / shown_width
        step_y = source_height / out_height * region_height / shown_height
        matrix = np.array([[step_x, 0, x0 * source_width / out_width + (first + 0.5) * step_x - 0.5],
                           [0, step_y, y0 * source_height / out_height + 0.5 * step_y - 0.5]], np.float32)
        return cv2.warpAffine(source, matrix, (columns, shown_height), flags=interpolation | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
    # 100% and more: the output pixels the view needs, through the filter, then each one as a block.
    table, start, needed = block_maps(geometry.scale, first, columns, shown_height)
    step_x, step_y = source_width / out_width, source_height / out_height
    matrix = np.array([[step_x, 0, (x0 + start + 0.5) * step_x - 0.5], [0, step_y, (y0 + 0.5) * step_y - 0.5]],
                      np.float32)
    region = cv2.warpAffine(source, matrix, needed, flags=interpolation | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_REPLICATE)
    return cv2.remap(region, table, None, cv2.INTER_NEAREST)


def upscaled_view(upscaled: np.ndarray, geometry: Geometry) -> np.ndarray:
    """A full-resolution upscaled frame (BGRA) drawn at `geometry`."""
    _, _, shown_width, shown_height = geometry.shown
    if geometry.fit:
        interpolation = cv2.INTER_AREA if geometry.scale < 1 else cv2.INTER_CUBIC
        return cv2.resize(upscaled, (shown_width, shown_height), interpolation=interpolation)
    x0, y0, region_width, region_height = geometry.region
    region = upscaled[y0:y0 + region_height, x0:x0 + region_width]
    if geometry.scale < 1:
        return cv2.resize(region, (shown_width, shown_height), interpolation=cv2.INTER_AREA)
    return block_view(region, geometry.scale, 0, shown_width, shown_height)


def reframe(image: np.ndarray, old: Geometry, new: Geometry, first: int, columns: int) -> np.ndarray:
    """View columns `first`..`first + columns` of geometry `new` drawn from `image`, a picture of `old`'s region
    at any size: the same output pixels where they now belong, resampled. A picture decoded for the previous
    zoom or position stays in place (softer) until its decoder catches up, so it moves with the other side."""
    height, width = image.shape[:2]
    old_x0, old_y0, old_width, old_height = old.region
    new_x0, new_y0, _, _ = new.region
    image_x, image_y = width / old_width, height / old_height  # image pixels per output pixel
    matrix = np.array([[image_x / new.scale, 0, ((first + 0.5) / new.scale + new_x0 - old_x0) * image_x - 0.5],
                       [0, image_y / new.scale, (0.5 / new.scale + new_y0 - old_y0) * image_y - 0.5]], np.float32)
    return cv2.warpAffine(image, matrix, (columns, new.shown[3]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def decode_request(geometry: Geometry) -> tuple[tuple[int, int, int, int] | None, tuple[int, int], str]:
    """What the playback decoder should crop and scale an upscaled video to for `geometry` (crop, size, scale
    flags). From 100% up it decodes the region at its own pixels, which the view draws as blocks exactly as
    when paused."""
    _, _, shown_width, shown_height = geometry.shown
    if geometry.fit:
        return None, (shown_width, shown_height), "area" if geometry.scale < 1 else "bicubic"
    x0, y0, region_width, region_height = geometry.region
    if geometry.scale < 1:
        return (x0, y0, region_width, region_height), (shown_width, shown_height), "area"
    return (x0, y0, region_width, region_height), (region_width, region_height), "neighbor"


# ---- the view ------------------------------------------------------------------------------------------


class VideoView(tk.Frame):
    """See the module docstring. Methods marked thread-safe may be called from the player thread."""

    def __init__(self, master: tk.Misc, theme: Theme, on_browse: Callable[[], None],
                 on_zoom_changed: Callable[[object], None], on_side_click: Callable[[str], None] | None = None):
        super().__init__(master, background="", highlightthickness=0, borderwidth=0)
        self.theme = theme
        self.on_browse = on_browse
        self.on_zoom_changed = on_zoom_changed
        self.on_side_click = on_side_click
        palette, px = theme.palette, theme.px
        self.colors = {name: hex_to_bgr(getattr(palette, name)) for name in
                       ("background", "view", "border", "border_hover", "text", "text_secondary", "text_faint",
                        "accent")}
        self._lock = threading.RLock()
        self._window = self.winfo_id()
        self._view_size = (1, 1)
        self._canvas: np.ndarray | None = None
        self.mode = "empty"
        self.message = ""
        self.zoom = "fit"
        self.center: tuple[float, float] | None = None
        self.split = 0.5
        self.handle_y: int | None = None  # where the divider was last pressed or dragged; None: the picture's middle
        self.sides = {"left": "cubic", "right": "unavailable"}
        self.side_labels: dict[str, str] = {}
        self.fullscreen = False
        self.hover = False
        self.spinner = False
        self.status = ""
        self.source_size = (1, 1)
        self.output_size = (1, 1)
        self._still: tuple[np.ndarray, dict] | None = None
        self._stream: tuple[np.ndarray, dict] | None = None
        self._pill_boxes: dict[str, tuple[int, int, int, int]] = {}
        self._cache: dict[str, tuple] = {}
        self._geometry: Geometry | None = None
        self._minimap_box: tuple[int, int, int, int] | None = None
        self._next_minimap_box: tuple[int, int, int, int] | None = None
        self._drag: str | None = None
        self._pill_pressed: str | None = None
        self._pan_origin: tuple[int, int, float, float] | None = None
        self._blit_failed = False
        self._text_cache: dict[tuple, np.ndarray] = {}
        self._fonts = {name: (font.actual("family"), self._pixel_height(font.actual("size")))
                       for name, font in theme.fonts.items()}
        self._handle = self._make_handle(px(30))
        self._spinner_frames = self._make_spinner(px(40))
        self._spinner_index = 0
        self._icon = {False: self._make_drop_icon(px(44), palette.text_faint),
                      True: self._make_drop_icon(px(44), palette.accent)}
        self._outline = {False: self._make_outline(px(12), palette.border_hover),
                         True: self._make_outline(px(12), palette.accent)}
        self.bind("<Configure>", self._configured)
        self.bind("<Expose>", lambda _: self.redraw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag_motion)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Double-Button-1>", self._double_click)
        for button in ("2", "3"):
            self.bind(f"<ButtonPress-{button}>", self._pan_press)
            self.bind(f"<B{button}-Motion>", self._pan_motion)
            self.bind(f"<ButtonRelease-{button}>", self._release)
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Motion>", self._motion)
        self.bind("<Enter>", lambda _: self._set_hover(True))
        self.bind("<Leave>", lambda _: self._set_hover(False))

    def _pixel_height(self, points: int) -> int:
        return max(6, int(round(abs(points) * 96 * self.theme.scale / 72)))

    # ---- content (thread-safe) ------------------------------------------------------------------

    def set_media(self, source_size: tuple[int, int], output_size: tuple[int, int]) -> None:
        """Sizes of the source and of the planned output; a new output size resets the zoom position."""
        with self._lock:
            if output_size != self.output_size:
                self.center = None
            self.source_size, self.output_size = source_size, output_size
            self._cache.clear()

    def show_empty(self) -> None:
        with self._lock:
            self.mode, self._still, self._stream = "empty", None, None
            self.spinner = False
        self.redraw()

    def show_message(self, message: str, spinner: bool = False) -> None:
        with self._lock:
            self.mode, self.message, self.spinner = "message", message, spinner
        self.redraw()

    def show_still(self, source: np.ndarray, layers: dict[str, np.ndarray | None], status: str = "",
                   spinner: bool = False) -> None:
        """A paused frame: the source (BGRA, its own size) and the full upscaled frames by layer (None where
        that frame is not upscaled), with a status for missing ones and an optional spinner. Thread-safe."""
        with self._lock:
            self.mode = "compare"
            self._still, self._stream = (source, layers), None
            self.status = status
            self.spinner = spinner
            self._cache.clear()
        self.redraw()

    def present(self, source: np.ndarray, layers: dict[str, tuple[np.ndarray, tuple] | None]) -> None:
        """A playing frame: the source (BGRA) and, by layer, the upscaled picture already at the geometry its
        key was computed for (see `decode_target`), or None. Thread-safe."""
        with self._lock:
            self.mode = "compare"
            self._stream, self._still = (source, layers), None
            self.status = "not upscaled here"
            self.spinner = False
        self.redraw()

    def set_spinner(self, spinning: bool, advance: bool = False) -> None:
        """Show (or hide) the busy spinner; `advance` turns it one step. Thread-safe."""
        with self._lock:
            self.spinner = spinning
            if advance:
                self._spinner_index = (self._spinner_index + 1) % SPINNER_FRAMES
        self.redraw()

    def set_status(self, status: str) -> None:
        with self._lock:
            self.status = status
        self.redraw()

    def set_sides(self, sides: dict[str, str], labels: dict[str, str]) -> None:
        """What each side ("left", "right") shows: a REFERENCE_FILTERS name, an upscaled layer's key (two sides
        showing the same version share one) or OFF, and the label of each side that shows an upscaled layer."""
        with self._lock:
            self.sides.update(sides)
            self.side_labels = dict(labels)
        self.redraw()

    def needed_layers(self) -> set[str]:
        """The upscaled layers some side shows (the player decodes only those). Thread-safe."""
        with self._lock:
            return {layer for layer in self.sides.values() if layer not in REFERENCE_FILTERS and layer != OFF}

    @property
    def single(self) -> bool:
        """Whether one side is off (the other fills the picture, without the divider)."""
        return OFF in self.sides.values()

    def set_fullscreen(self, fullscreen: bool) -> None:
        with self._lock:
            self.fullscreen = fullscreen
        self.redraw()

    def decode_target(self) -> tuple[tuple, tuple[int, int, int, int] | None, tuple[int, int], str]:
        """(geometry key, crop, size, scale flags) for decoding upscaled pieces to the current view. Thread-safe."""
        with self._lock:
            geometry = self._compute_geometry()
            return (geometry.key,) + decode_request(geometry)

    @property
    def comparing(self) -> bool:
        return self.mode == "compare" and (self._still is not None or self._stream is not None)

    @property
    def playing_stream(self) -> bool:
        return self._stream is not None

    def snapshot(self) -> np.ndarray | None:
        """A copy of the last composed canvas (for tests)."""
        with self._lock:
            return None if self._canvas is None else self._canvas.copy()

    # ---- zoom and position (Tk thread) ----------------------------------------------------------

    def set_zoom(self, zoom, anchor: tuple[int, int] | None = None) -> None:
        """Change the zoom ("fit", "source" or view pixels per output pixel), keeping the output pixel under
        `anchor` (view coordinates) in place."""
        with self._lock:
            if zoom == self.zoom:
                return
            geometry = self._geometry
            if anchor is not None and geometry is not None and zoom != "fit":
                pixel_x, pixel_y = self._output_point(geometry, *anchor)
                view_width, view_height = self._view_size
                scale = self.zoom_scale(zoom)
                self.center = (pixel_x + (view_width / 2 - anchor[0]) / scale,
                               pixel_y + (view_height / 2 - anchor[1]) / scale)
            self.zoom = zoom
        self.on_zoom_changed(zoom)
        self.redraw()

    def fit_scale(self) -> float:
        """View pixels per output pixel when the whole picture fits."""
        view_width, view_height = self._view_size
        return min(view_width / self.output_size[0], view_height / self.output_size[1])

    def zoom_scale(self, zoom) -> float:
        """View pixels per output pixel at a zoom level."""
        if zoom == "fit":
            return self.fit_scale()
        if zoom == "source":
            return self.source_size[0] / self.output_size[0]
        return float(zoom)

    def zoom_range(self) -> tuple[float, float]:
        """The smallest and largest scale the slider and the wheel go to: the smallest of Fit, Original size and
        100%, to MAX_ZOOM."""
        return min(self.fit_scale(), self.zoom_scale("source"), 1.0), MAX_ZOOM

    def zoom_named(self, scale: float):
        """The zoom level for `scale`: "fit", "source" or a whole number when it is (nearly) one of those."""
        for name in ("fit", "source"):
            if abs(scale / self.zoom_scale(name) - 1) < SNAP_TOLERANCE:
                return name
        if abs(scale - round(scale)) < SNAP_TOLERANCE * scale and round(scale) >= 1:
            return int(round(scale))
        return scale

    def zoom_step(self, steps: int):
        """The zoom `steps` wheel notches away (positive zooms in), stopping at Fit, Original size and whole
        zooms on the way."""
        stops = [self.fit_scale(), self.zoom_scale("source"), 1.0, 2.0, 4.0, 8.0]
        return self.zoom_named(step_zoom(self.zoom_scale(self.zoom), steps, stops, *self.zoom_range()))

    @staticmethod
    def _output_point(geometry: Geometry, x: float, y: float) -> tuple[float, float]:
        left, top = geometry.shown[:2]
        return (geometry.region[0] + (x - left) / geometry.scale, geometry.region[1] + (y - top) / geometry.scale)

    def _compute_geometry(self) -> Geometry:
        geometry = compute_geometry(self._view_size, self.output_size, self.zoom, self.center, self.source_size)
        if not geometry.fit:
            x0, y0, width, height = geometry.region
            self.center = (x0 + width / 2, y0 + height / 2)
        return geometry

    # ---- drawing --------------------------------------------------------------------------------

    def redraw(self) -> None:
        """Compose the current state and blit it. Thread-safe."""
        with self._lock:
            width, height = self._view_size
            if width < 8 or height < 8:
                return
            if self._canvas is None or self._canvas.shape[:2] != (height, width):
                self._canvas = np.empty((height, width, 4), np.uint8)
            canvas = self._canvas
            # The Tk thread reads the box while the player thread composes: it keeps the last one until then.
            self._next_minimap_box = None
            if self.comparing:
                self._compose_compare(canvas)
            else:
                fill(canvas, 0, 0, width, height, self.colors["view"])
                if self.mode == "message":
                    self._compose_message(canvas)
                else:
                    self._compose_empty(canvas)
            self._minimap_box = self._next_minimap_box
            if self.spinner:
                sprite = self._spinner_frames[self._spinner_index]
                blend(canvas, sprite, (width - sprite.shape[1]) // 2, (height - sprite.shape[0]) // 2)
            if not self.fullscreen:
                self._draw_frame(canvas)
            if not gdi().blit(self._window, canvas) and not self._blit_failed:
                self._blit_failed = True
                log.emit("[ui] the preview could not be drawn (GDI refused the blit)")

    def _compose_compare(self, canvas: np.ndarray) -> None:
        geometry = self._compute_geometry()
        self._geometry = geometry
        width, height = self._view_size
        left, top, shown_width, shown_height = geometry.shown
        background = self.colors["view"]
        fill(canvas, 0, 0, width, top, background)
        fill(canvas, 0, top + shown_height, width, height, background)
        fill(canvas, 0, top, left, top + shown_height, background)
        fill(canvas, left + shown_width, top, width, top + shown_height, background)
        source, layers = self._stream if self._stream is not None else self._still
        split = int(round(self.split * shown_width))
        if self.sides["left"] == OFF:
            split = 0
        elif self.sides["right"] == OFF:
            split = shown_width
        target = canvas[top:top + shown_height, left:left + shown_width]
        missing = set()
        for side, first, count in (("left", 0, split), ("right", split, shown_width - split)):
            if count <= 0:
                continue
            image = self._layer_columns(self.sides[side], geometry, source, layers, first, count)
            if image is None:
                # Not upscaled at this frame: the source, dimmed, holds the place.
                missing.add(side)
                image = self._layer_columns("cubic", geometry, source, layers, first, count)
                target[:, first:first + count, :3] = (image[:shown_height, :count, :3] * PLACEHOLDER_DIM).astype(np.uint8)
            else:
                target[:, first:first + count] = image[:shown_height, :count]
        self._draw_overlays(canvas, geometry, source, split, missing)

    def _layer_columns(self, layer: str, geometry: Geometry, source: np.ndarray, layers: dict, first: int,
                       count: int) -> np.ndarray | None:
        """View columns `first`..`first + count` of `layer` at `geometry`, or None where it has no picture."""
        _, _, shown_width, shown_height = geometry.shown
        if layer in REFERENCE_FILTERS:
            interpolation = REFERENCE_FILTERS[layer][1]
            if self._stream is None:
                full = self._cached(f"filter:{layer}", (geometry.key, id(source)),
                                    lambda: reference_view(source, geometry, interpolation, shown_width))
                return full[:, first:first + count]
            return reference_view(source, geometry, interpolation, count, first)
        entry = layers.get(layer)
        if entry is None:
            return None
        if self._stream is not None:
            image, key = entry
            if key != geometry.key:
                previous = Geometry(*key)
                if previous.output_size == geometry.output_size:
                    # The decoder still works at the previous zoom or position: its picture, put where it belongs.
                    return reframe(image, previous, geometry, first, count)
                image = cv2.resize(image, (shown_width, shown_height), interpolation=cv2.INTER_LINEAR)
            elif not geometry.fit and geometry.scale >= 1:
                return block_view(image, geometry.scale, first, count, shown_height)
            return image[:, first:first + count]

        def make() -> np.ndarray:
            frame = entry
            if (frame.shape[1], frame.shape[0]) != geometry.output_size:  # a version made at another size
                frame = cv2.resize(frame, geometry.output_size, interpolation=cv2.INTER_CUBIC)
            return upscaled_view(frame, geometry)

        return self._cached(f"layer:{layer}", (geometry.key, id(entry)), make)[:, first:first + count]

    def _cached(self, name: str, key: tuple, make: Callable[[], np.ndarray]) -> np.ndarray:
        entry = self._cache.get(name)
        if entry is None or entry[0] != key:
            entry = (key, make())
            self._cache[name] = entry
        return entry[1]

    def _side_label(self, side: str, geometry: Geometry, source: np.ndarray, missing: bool) -> str:
        layer = self.sides[side]
        if layer in REFERENCE_FILTERS:
            if reference_shrinks(geometry, (source.shape[1], source.shape[0])):
                return "Source"
            return REFERENCE_FILTERS[layer][0]
        label = self.side_labels.get(side, "Upscaled")
        return f"{label} · {self.status or 'not upscaled here'}" if missing else label

    def _draw_overlays(self, canvas: np.ndarray, geometry: Geometry, source: np.ndarray, split: int,
                       missing: set[str]) -> None:
        px = self.theme.px
        left, top, shown_width, shown_height = geometry.shown
        if not self.single:
            x = left + split
            line = max(1, px(2))
            fill(canvas, x - line // 2, top, x - line // 2 + line, top + shown_height, DIVIDER_COLOR)
            half = self._handle.shape[0] // 2
            middle = top + shown_height // 2 if self.handle_y is None else self.handle_y
            middle = max(top + half, min(top + shown_height - half, middle))  # whole, inside the picture
            blend(canvas, self._handle, x - self._handle.shape[1] // 2, middle - half)
        margin = px(10)
        boxes = {}
        arrow = "" if self.fullscreen or self.on_side_click is None else "  ▾"
        for side in ("left", "right"):
            if self.sides[side] == OFF:
                continue
            pill = self._pill(self._side_label(side, geometry, source, side in missing) + arrow)
            pill_left = left + margin if side == "left" else left + shown_width - margin - pill.shape[1]
            blend(canvas, pill, pill_left, top + margin)
            boxes[side] = (pill_left, top + margin, pill_left + pill.shape[1], top + margin + pill.shape[0])
        self._pill_boxes = boxes
        if not geometry.fit:
            pill = self._pill("Original size" if self.zoom == "source" else f"{geometry.scale * 100:.0f}%")
            blend(canvas, pill, left + margin, top + shown_height - margin - pill.shape[0])
            x0, y0, region_width, region_height = geometry.region
            out_width, out_height = geometry.output_size
            if region_width < out_width or region_height < out_height:
                self._draw_minimap(canvas, geometry, source)

    def _draw_minimap(self, canvas: np.ndarray, geometry: Geometry, source: np.ndarray) -> None:
        px = self.theme.px
        out_width, _ = geometry.output_size
        box = minimap_box(self._view_size, geometry.output_size, px)
        if box is None:
            return
        map_left, map_top, right, bottom = box
        map_width, map_height = right - map_left, bottom - map_top
        thumbnail = self._cached("minimap", (map_width, map_height, id(source)),
                                 lambda: cv2.resize(source, (map_width, map_height), interpolation=cv2.INTER_AREA))
        fill(canvas, map_left - 2, map_top - 2, right + 2, bottom + 2, self.colors["border_hover"])
        canvas[map_top:bottom, map_left:right, :3] = thumbnail[..., :3]
        scale = map_width / out_width
        x0, y0, region_width, region_height = geometry.region
        box_left, box_top = map_left + int(x0 * scale), map_top + int(y0 * scale)
        box_right = map_left + int(math.ceil((x0 + region_width) * scale))
        box_bottom = map_top + int(math.ceil((y0 + region_height) * scale))
        thickness = max(1, px(1.5))
        accent = self.colors["accent"]
        fill(canvas, box_left, box_top, box_right, box_top + thickness, accent)
        fill(canvas, box_left, box_bottom - thickness, box_right, box_bottom, accent)
        fill(canvas, box_left, box_top, box_left + thickness, box_bottom, accent)
        fill(canvas, box_right - thickness, box_top, box_right, box_bottom, accent)
        self._next_minimap_box = box

    def _compose_message(self, canvas: np.ndarray) -> None:
        width, height = self._view_size
        text = self._text(self.message, "body", self.colors["text_secondary"])
        offset = self.theme.px(40) if self.spinner else 0
        blend(canvas, text, (width - text.shape[1]) // 2, (height - text.shape[0]) // 2 + offset)

    def _compose_empty(self, canvas: np.ndarray) -> None:
        px = self.theme.px
        width, height = self._view_size
        margin = px(18)
        blend_outline(canvas, self._outline[self.hover], margin, margin, width - 2 * margin, height - 2 * margin)
        middle = height // 2
        icon = self._icon[self.hover]
        blend(canvas, icon, (width - icon.shape[1]) // 2, middle - px(58) - icon.shape[0] // 2)
        for text, font, color, offset in (
                ("Drop a video or an image here", "display", "text", 0),
                ("or click to browse", "body", "text_secondary", px(32)),
                ("MP4 · MKV · MOV · AVI · WMV · WebM · MPEG · FLV · TS · Ogg · PNG · JPG · WebP", "small",
                 "text_faint", px(64))):
            sprite = self._text(text, font, self.colors[color])
            blend(canvas, sprite, (width - sprite.shape[1]) // 2, middle + offset - sprite.shape[0] // 2)

    def _draw_frame(self, canvas: np.ndarray) -> None:
        """Rounded corners and a 1px border over everything (not in full screen)."""
        height, width = canvas.shape[:2]
        border = self.colors["border"]
        fill(canvas, 0, 0, width, 1, border)
        fill(canvas, 0, height - 1, width, height, border)
        fill(canvas, 0, 0, 1, height, border)
        fill(canvas, width - 1, 0, width, height, border)
        corners = self._cached("corners", (self.theme.px(10),), self._make_corners)
        top_left, top_right, bottom_left, bottom_right = corners
        size = top_left.shape[0]
        blend(canvas, top_left, 0, 0)
        blend(canvas, top_right, width - size, 0)
        blend(canvas, bottom_left, 0, height - size)
        blend(canvas, bottom_right, width - size, height - size)

    # ---- sprites --------------------------------------------------------------------------------

    def _text(self, text: str, font: str, color: tuple[int, int, int]) -> np.ndarray:
        key = (text, font, color)
        sprite = self._text_cache.get(key)
        if sprite is None:
            family, pixel_height = self._fonts[font]
            sprite = tinted(gdi().text_coverage(text, family, pixel_height), color)
            if len(self._text_cache) > 256:
                self._text_cache.clear()
            self._text_cache[key] = sprite
        return sprite

    def _pill(self, text: str) -> np.ndarray:
        key = ("pill", text)
        sprite = self._text_cache.get(key)
        if sprite is None:
            px = self.theme.px
            label = self._text(text, "small", (0xf5, 0xf4, 0xf4))
            width, height = label.shape[1] + 2 * px(9), px(22)
            sketch = Sketch(width, height)
            sketch.fill(sketch.rounded_rect(0, 0, width, height, height / 2), "#000000", opacity=0.6)
            sprite = sketch.premultiplied_bgra()
            over(sprite, label, px(9), (height - label.shape[0]) // 2)
            self._text_cache[key] = sprite
        return sprite

    def _make_handle(self, size: int) -> np.ndarray:
        sketch = Sketch(size, size)
        center = size / 2
        sketch.fill(sketch.circle(center, center + 1, center - 1), "#000000", opacity=0.35)
        sketch.fill(sketch.circle(center, center, center - 2), "#f4f4f5")
        arm = size * 0.12
        thickness = max(1.2, size * 0.06)
        for direction in (-1, 1):
            tip = center + direction * size * 0.24
            sketch.fill(sketch.polyline([(tip - direction * arm, center - arm), (tip, center),
                                         (tip - direction * arm, center + arm)], thickness), "#1e2024")
        return sketch.premultiplied_bgra()

    def _make_spinner(self, size: int) -> list[np.ndarray]:
        frames = []
        center = size / 2
        thickness = max(2.0, size * 0.09)
        for index in range(SPINNER_FRAMES):
            sketch = Sketch(size, size)
            sketch.fill(sketch.circle(center, center, center), "#000000", opacity=0.45)
            ring = np.clip(sketch.circle(center, center, center * 0.62)
                           - sketch.circle(center, center, center * 0.62 - thickness), 0, 1)
            angle = (np.arctan2(sketch.y - center, sketch.x - center) / (2 * np.pi) + 1 - index / SPINNER_FRAMES) % 1
            sketch.fill(ring * angle ** 1.5, "#ffffff")
            frames.append(sketch.premultiplied_bgra())
        return frames

    def _make_drop_icon(self, size: int, color: str) -> np.ndarray:
        unit = size / 32
        sketch = Sketch(size, size)
        thickness = 2.2 * unit
        tray = sketch.polyline([(5 * unit, 18 * unit), (5 * unit, 26 * unit), (27 * unit, 26 * unit),
                                (27 * unit, 18 * unit)], thickness)
        arrow = sketch.polyline([(16 * unit, 20 * unit), (16 * unit, 5 * unit)], thickness)
        head = sketch.polyline([(10 * unit, 11 * unit), (16 * unit, 5 * unit), (22 * unit, 11 * unit)], thickness)
        sketch.fill(np.maximum(np.maximum(tray, arrow), head), color)
        return sketch.premultiplied_bgra()

    @staticmethod
    def _make_outline(radius: int, color: str) -> np.ndarray:
        """The drop zone's outline as a small square for `blend_outline` (one at full size took 0.3 s to make
        at 4K)."""
        side = 2 * (radius + 2) + 1
        sketch = Sketch(side, side)
        outer = sketch.rounded_rect(0, 0, side, side, radius)
        inner = sketch.rounded_rect(1.5, 1.5, side - 1.5, side - 1.5, radius - 1.5)
        sketch.fill(np.clip(outer - inner, 0, 1), color)
        return sketch.premultiplied_bgra()

    def _make_corners(self) -> list[np.ndarray]:
        palette, radius = self.theme.palette, self.theme.px(10)
        sketch = Sketch(radius, radius)
        inside = sketch.circle(radius, radius, radius)
        sketch.fill(1 - inside, palette.background)
        sketch.fill(np.clip(inside - sketch.circle(radius, radius, radius - 1), 0, 1), palette.border)
        pixels = sketch.premultiplied_bgra()
        return [np.ascontiguousarray(pixels), np.ascontiguousarray(pixels[:, ::-1]),
                np.ascontiguousarray(pixels[::-1, :]), np.ascontiguousarray(pixels[::-1, ::-1])]

    # ---- input (Tk thread) ----------------------------------------------------------------------

    def _configured(self, event: tk.Event) -> None:
        with self._lock:
            self._view_size = (max(1, event.width), max(1, event.height))
            for name in [name for name in self._cache if name.startswith(("layer:", "filter:"))]:
                del self._cache[name]
        self.redraw()

    def _set_hover(self, hover: bool) -> None:
        self.hover = hover
        if self.mode == "empty":
            self.redraw()

    def _in_minimap(self, x: int, y: int) -> bool:
        box = self._minimap_box
        return box is not None and box[0] <= x <= box[2] and box[1] <= y <= box[3]

    def _pill_at(self, x: int, y: int) -> str | None:
        for side, (left, top, right, bottom) in self._pill_boxes.items():
            if left <= x <= right and top <= y <= bottom:
                return side
        return None

    def _press(self, event: tk.Event) -> None:
        if self.mode == "empty":
            self.on_browse()
            return
        if not self.comparing:
            return
        side = self._pill_at(event.x, event.y)
        if side is not None and self.on_side_click is not None and not self.fullscreen:
            self._pill_pressed = side  # opened on release, or the release would land on the list and pick
            return
        if self._in_minimap(event.x, event.y):
            self._drag = "minimap"
            self._move_to_minimap(event.x, event.y)
        elif not self.single:
            self._drag = "divider"
            self._move_divider(event.x, event.y)

    def _drag_motion(self, event: tk.Event) -> None:
        if self._drag == "minimap":
            self._move_to_minimap(event.x, event.y)
        elif self._drag == "divider":
            self._move_divider(event.x, event.y)

    def _release(self, event: tk.Event) -> None:
        side, self._pill_pressed = self._pill_pressed, None
        self._drag = None
        self._pan_origin = None
        self._motion(event)
        if side is not None and self._pill_at(event.x, event.y) == side:
            self.on_side_click(side)

    def _move_divider(self, x: int, y: int) -> None:
        geometry = self._geometry
        if geometry is None:
            return
        left, _, shown_width, _ = geometry.shown
        with self._lock:
            self.split = min(1.0, max(0.0, (x - left) / max(shown_width, 1)))
            self.handle_y = y
        self.redraw()

    def _move_to_minimap(self, x: int, y: int) -> None:
        box = self._minimap_box
        if box is None:
            return  # zoomed out meanwhile, or no room left for the map
        left, top, right, _ = box
        scale = (right - left) / self.output_size[0]
        with self._lock:
            self.center = ((x - left) / scale, (y - top) / scale)
        self.redraw()

    def _pan_press(self, event: tk.Event) -> None:
        if self.comparing and self.zoom != "fit" and self.center is not None:
            self._pan_origin = (event.x, event.y, self.center[0], self.center[1])
            self.configure(cursor="fleur")

    def _pan_motion(self, event: tk.Event) -> None:
        if self._pan_origin is None:
            return
        start_x, start_y, center_x, center_y = self._pan_origin
        scale = self.zoom_scale(self.zoom)
        with self._lock:
            self.center = (center_x - (event.x - start_x) / scale, center_y - (event.y - start_y) / scale)
        self.redraw()

    def _wheel(self, event: tk.Event) -> None:
        if self.comparing:
            self.set_zoom(self.zoom_step(1 if event.delta > 0 else -1), anchor=(event.x, event.y))

    def _double_click(self, event: tk.Event) -> None:
        if self.comparing:
            self.set_zoom(1 if self.zoom == "fit" else "fit", anchor=(event.x, event.y))

    def _motion(self, event: tk.Event) -> None:
        if self.mode == "empty":
            cursor = "hand2"
        elif not self.comparing:
            cursor = ""
        elif self._in_minimap(event.x, event.y) or (self._pill_at(event.x, event.y) is not None
                                                     and self.on_side_click is not None and not self.fullscreen):
            cursor = "hand2"
        else:
            cursor = "" if self.single else "sb_h_double_arrow"
        if self._pan_origin is None and str(self.cget("cursor")) != cursor:
            self.configure(cursor=cursor)
