"""Anti-aliased shapes drawn with numpy, exported as PNG or PPM data for Tk images (no imaging package).

Shapes are signed distance fields: a pixel's coverage is `clip(0.5 - distance, 0, 1)`, which gives smooth
one-pixel edges without supersampling. `Sketch` composites shapes in premultiplied RGBA.
"""
from __future__ import annotations

import base64
import struct
import zlib

import numpy as np


def rgb(color: str) -> np.ndarray:
    """'#rrggbb' to a float array in 0..1."""
    color = color.lstrip("#")
    return np.array([int(color[index:index + 2], 16) for index in (0, 2, 4)], dtype=np.float32) / 255


def png_bytes(rgba: np.ndarray) -> bytes:
    """Encode an (height, width, 4) uint8 array as a PNG file."""
    height, width = rgba.shape[:2]
    rows = np.concatenate([np.zeros((height, 1), np.uint8), rgba.reshape(height, width * 4)], axis=1)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
            + chunk(b"IEND", b""))


def png_base64(rgba: np.ndarray) -> str:
    return base64.b64encode(png_bytes(rgba)).decode("ascii")


def ppm_bytes(rgb_pixels: np.ndarray) -> bytes:
    """Encode an (height, width, 3) uint8 array as binary PPM, the fastest format Tk photo images load."""
    height, width = rgb_pixels.shape[:2]
    return b"P6 %d %d 255\n" % (width, height) + np.ascontiguousarray(rgb_pixels).tobytes()


class Sketch:
    """An RGBA image to paint anti-aliased shapes on."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.premultiplied = np.zeros((height, width, 4), np.float32)
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        self.x = xs + 0.5
        self.y = ys + 0.5

    # ---- coverage masks -------------------------------------------------------------------------

    def rounded_rect(self, x0: float, y0: float, x1: float, y1: float, radius: float) -> np.ndarray:
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        half_x, half_y = (x1 - x0) / 2, (y1 - y0) / 2
        radius = min(radius, half_x, half_y)
        qx = np.abs(self.x - center_x) - (half_x - radius)
        qy = np.abs(self.y - center_y) - (half_y - radius)
        outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
        inside = np.minimum(np.maximum(qx, qy), 0)
        return np.clip(0.5 - (outside + inside - radius), 0, 1)

    def circle(self, center_x: float, center_y: float, radius: float) -> np.ndarray:
        return np.clip(0.5 - (np.hypot(self.x - center_x, self.y - center_y) - radius), 0, 1)

    def segment(self, ax: float, ay: float, bx: float, by: float, thickness: float) -> np.ndarray:
        """A capsule (line with round caps) from a to b."""
        px, py = self.x - ax, self.y - ay
        dx, dy = bx - ax, by - ay
        length_squared = max(dx * dx + dy * dy, 1e-6)
        t = np.clip((px * dx + py * dy) / length_squared, 0, 1)
        distance = np.hypot(px - t * dx, py - t * dy) - thickness / 2
        return np.clip(0.5 - distance, 0, 1)

    def polyline(self, points: list[tuple[float, float]], thickness: float) -> np.ndarray:
        mask = np.zeros((self.height, self.width), np.float32)
        for (ax, ay), (bx, by) in zip(points, points[1:]):
            mask = np.maximum(mask, self.segment(ax, ay, bx, by, thickness))
        return mask

    # ---- painting -------------------------------------------------------------------------------

    def fill(self, mask: np.ndarray, color: str | np.ndarray, opacity: float = 1.0) -> "Sketch":
        """Paint `color` (a hex string or an (h, w, 3) array) through `mask` over what is there."""
        alpha = (mask * opacity)[..., None]
        color_values = rgb(color) if isinstance(color, str) else color
        self.premultiplied[..., :3] = color_values * alpha + self.premultiplied[..., :3] * (1 - alpha)
        self.premultiplied[..., 3:] = alpha + self.premultiplied[..., 3:] * (1 - alpha)
        return self

    def vertical_gradient(self, top: str, bottom: str) -> np.ndarray:
        t = (self.y / self.height)[..., None]
        return rgb(top) * (1 - t) + rgb(bottom) * t

    def rgba(self) -> np.ndarray:
        alpha = self.premultiplied[..., 3:]
        color = np.where(alpha > 0, self.premultiplied[..., :3] / np.maximum(alpha, 1e-6), 0)
        return (np.concatenate([color, alpha], axis=2) * 255 + 0.5).clip(0, 255).astype(np.uint8)

    def png_base64(self) -> str:
        return png_base64(self.rgba())


# ---- shared artwork ---------------------------------------------------------------------------------


def app_icon(size: int) -> np.ndarray:
    """The app icon: on an orange tile, a small block in the corner of a large frame, and an arrow growing it."""
    sketch = Sketch(size, size)
    inset = size * 0.04
    sketch.fill(sketch.rounded_rect(inset, inset, size - inset, size - inset, size * 0.22),
                sketch.vertical_gradient("#ff8a4c", "#e8452f"))
    unit = size / 16
    stroke = max(1.0, 1.15 * unit)
    outline = sketch.rounded_rect(3.3 * unit, 3.3 * unit, 12.7 * unit, 12.7 * unit, 1.6 * unit)
    hole = sketch.rounded_rect(3.3 * unit + stroke, 3.3 * unit + stroke, 12.7 * unit - stroke, 12.7 * unit - stroke,
                               max(0.0, 1.6 * unit - stroke))
    sketch.fill(np.clip(outline - hole, 0, 1), "#ffffff", opacity=0.55)
    sketch.fill(sketch.rounded_rect(3.3 * unit, 7.9 * unit, 8.1 * unit, 12.7 * unit, 1.1 * unit), "#ffffff")
    tip_x, tip_y = 10.9 * unit, 5.1 * unit
    arrow = sketch.polyline([(8.4 * unit, 7.6 * unit), (tip_x, tip_y)], stroke * 1.1)
    head = sketch.polyline([(tip_x - 2.4 * unit, tip_y), (tip_x, tip_y), (tip_x, tip_y + 2.4 * unit)], stroke * 1.1)
    sketch.fill(np.maximum(arrow, head), "#ffffff")
    return sketch.rgba()


def ico_bytes(sizes: tuple[int, ...] = (16, 24, 32, 48, 64, 128, 256)) -> bytes:
    """A Windows .ico file holding the app icon at several sizes (PNG-compressed entries)."""
    images = [png_bytes(app_icon(size)) for size in sizes]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries = b""
    for size, data in zip(sizes, images):
        dimension = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dimension, dimension, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    return header + entries + b"".join(images)
