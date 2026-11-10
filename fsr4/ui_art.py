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


def stretch_middle(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    """A 9-slice image grown to at least `width` x `height` by repeating its middle column and row: the same
    picture at any widget size, but ttk covers a widget with a few long middle pieces instead of many short ones."""
    rows, columns = rgba.shape[:2]
    if width > columns:
        middle = columns // 2
        rgba = np.concatenate([rgba[:, :middle], np.repeat(rgba[:, middle:middle + 1], width - columns + 1, axis=1),
                               rgba[:, middle + 1:]], axis=1)
    if height > rows:
        middle = rows // 2
        rgba = np.concatenate([rgba[:middle], np.repeat(rgba[middle:middle + 1], height - rows + 1, axis=0),
                               rgba[middle + 1:]], axis=0)
    return rgba


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

    def ellipse(self, center_x: float, center_y: float, radius_x: float, radius_y: float) -> np.ndarray:
        """An axis-aligned ellipse; its distance is approximate, close enough for one-pixel edges."""
        stretched = np.hypot((self.x - center_x) / radius_x, (self.y - center_y) / radius_y)
        return np.clip(0.5 - (stretched - 1) * min(radius_x, radius_y), 0, 1)

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

    def polygon(self, points: list[tuple[float, float]]) -> np.ndarray:
        """A filled convex polygon (points in either winding order)."""
        distance = np.full((self.height, self.width), -np.inf, np.float32)
        area = sum(ax * by - bx * ay for (ax, ay), (bx, by) in zip(points, points[1:] + points[:1]))
        sign = 1.0 if area > 0 else -1.0
        for (ax, ay), (bx, by) in zip(points, points[1:] + points[:1]):
            length = max(np.hypot(bx - ax, by - ay), 1e-6)
            # signed distance to the edge line, negative on the inner side
            edge = sign * ((self.x - ax) * (by - ay) - (self.y - ay) * (bx - ax)) / length
            distance = np.maximum(distance, edge)
        return np.clip(0.5 - distance, 0, 1)

    def rounded_triangle(self, points: list[tuple[float, float]], radius: float) -> np.ndarray:
        """A filled triangle whose corners are rounded with `radius`, filling the same outline as the sharp one."""
        inner = inset_triangle(points, radius)
        return np.maximum(self.polygon(inner), self.polyline(inner + inner[:1], radius * 2))

    def ring(self, center_x: float, center_y: float, radius: float, thickness: float, start: float,
             end: float) -> np.ndarray:
        """An arc of a circle outline between angles `start` and `end` (radians, 0 = right, clockwise in y-down)."""
        distance = np.abs(np.hypot(self.x - center_x, self.y - center_y) - radius) - thickness / 2
        angle = np.arctan2(self.y - center_y, self.x - center_x)
        inside = (angle >= start) & (angle <= end)
        return np.clip(0.5 - distance, 0, 1) * inside

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

    def opaque(self, color: str) -> np.ndarray:
        """RGBA uint8 of the sketch over a solid `color`: every pixel fully opaque."""
        alpha = self.premultiplied[..., 3:]
        rgba = np.full((self.height, self.width, 4), 255, np.uint8)
        rgba[..., :3] = ((self.premultiplied[..., :3] + rgb(color) * (1 - alpha)) * 255 + 0.5).clip(0, 255)
        return rgba

    def png_base64(self) -> str:
        return png_base64(self.rgba())

    def premultiplied_bgra(self) -> np.ndarray:
        """Premultiplied BGRA uint8, the layout the video view composites and blits."""
        data = (self.premultiplied * 255 + 0.5).clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(data[..., [2, 1, 0, 3]])


# ---- shared artwork ---------------------------------------------------------------------------------


def inset_triangle(points: list[tuple[float, float]], inset: float) -> list[tuple[float, float]]:
    """The triangle whose sides lie `inset` inside those of `points`: a scale about the incircle's centre."""
    (ax, ay), (bx, by), (cx, cy) = points
    side_a, side_b, side_c = np.hypot(bx - cx, by - cy), np.hypot(cx - ax, cy - ay), np.hypot(ax - bx, ay - by)
    perimeter = side_a + side_b + side_c
    center_x = (side_a * ax + side_b * bx + side_c * cx) / perimeter
    center_y = (side_a * ay + side_b * by + side_c * cy) / perimeter
    inradius = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / perimeter
    scale = (inradius - inset) / inradius
    return [(center_x + (x - center_x) * scale, center_y + (y - center_y) * scale) for x, y in points]


LADYBUG_RED = "#e8402f"
LADYBUG_BLACK = "#17171a"


def app_icon(size: int) -> np.ndarray:
    """The app icon: a ladybug seen from above, facing right, its head a play triangle tucked under the red
    shell. Eyes and gloss appear only where the size leaves room for them."""
    sketch = Sketch(size, size)
    unit = size / 16

    def scaled(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return [(x * unit, y * unit) for x, y in points]

    head_corners = [(8.2, 3.4), (15.7, 8.0), (8.2, 12.6)]
    # a red rim keeps the black head visible on a dark title bar
    head_rim = sketch.rounded_triangle(scaled(head_corners), 1.1 * unit)
    sketch.fill(head_rim, LADYBUG_RED)
    sketch.fill(sketch.rounded_triangle(scaled(inset_triangle(head_corners, 0.6)), 0.55 * unit), LADYBUG_BLACK)
    shell_x, shell_radius_x, shell_radius_y = 5.0, 4.6, 5.4
    shell_front = shell_x + shell_radius_x
    if size >= 32:
        eyes = np.maximum(sketch.circle((shell_front + 1.15) * unit, 6.65 * unit, 0.5 * unit),
                          sketch.circle((shell_front + 1.15) * unit, 9.35 * unit, 0.5 * unit))
        sketch.fill(eyes, "#ffffff")
    # a dark gap where the shell overlaps the head, so their two reds do not merge
    gap = sketch.ellipse(shell_x * unit, 8 * unit, (shell_radius_x + 0.4) * unit, (shell_radius_y + 0.4) * unit)
    sketch.fill(gap * head_rim, LADYBUG_BLACK)
    shell = sketch.ellipse(shell_x * unit, 8 * unit, shell_radius_x * unit, shell_radius_y * unit)
    sketch.fill(shell, sketch.vertical_gradient("#ff6a52", "#c3170f"))
    spots = sketch.segment(0.4 * unit, 8 * unit, (shell_front - 0.3) * unit, 8 * unit, max(0.7, 0.5 * unit))
    for x, y, radius in ((2.1, 5.6, 1.1), (4.8, 4.1, 1.1)):
        spots = np.maximum(spots, np.maximum(sketch.circle(x * unit, y * unit, radius * unit),
                                             sketch.circle(x * unit, (16 - y) * unit, radius * unit)))
    sketch.fill(spots * shell, LADYBUG_BLACK)
    if size >= 24:
        sketch.fill(sketch.ellipse(3.1 * unit, 4.4 * unit, 1.3 * unit, 0.7 * unit) * shell, "#ffffff", 0.35)
    return sketch.rgba()


ICON_NAMES = ("play", "reverse", "pause", "volume", "volume_low", "volume_none", "mute", "fullscreen", "windowed",
              "sidebar", "add", "delete", "frame_back", "frame_forward")


def icon(name: str, size: int, color: str) -> np.ndarray:
    """A small control icon (straight RGBA) drawn on a 24-unit grid: play, reverse (play, mirrored), pause, volume
    (a speaker with two waves; volume_low one, volume_none none), mute (crossed), fullscreen, windowed (leave full
    screen), sidebar, add (a plus), delete (a bin), frame_back or frame_forward (a step of one frame: a triangle
    against a bar)."""
    sketch = Sketch(size, size)
    unit = size / 24
    stroke = max(1.2, 1.9 * unit)
    if name == "play":
        sketch.fill(sketch.polygon([(8 * unit, 5.5 * unit), (19 * unit, 12 * unit), (8 * unit, 18.5 * unit)]), color)
    elif name == "reverse":
        sketch.fill(sketch.polygon([(16 * unit, 5.5 * unit), (5 * unit, 12 * unit), (16 * unit, 18.5 * unit)]), color)
    elif name == "pause":
        for left in (7, 13.5):
            sketch.fill(sketch.rounded_rect(left * unit, 5.5 * unit, (left + 3.5) * unit, 18.5 * unit, 1 * unit), color)
    elif name in ("volume", "volume_low", "volume_none", "mute"):
        speaker = sketch.polygon([(4 * unit, 9.5 * unit), (8 * unit, 9.5 * unit), (13 * unit, 5 * unit),
                                  (13 * unit, 19 * unit), (8 * unit, 14.5 * unit), (4 * unit, 14.5 * unit)])
        sketch.fill(speaker, color)
        if name == "volume":
            waves = np.maximum(sketch.ring(12 * unit, 12 * unit, 4.5 * unit, stroke, -0.9, 0.9),
                               sketch.ring(12 * unit, 12 * unit, 8 * unit, stroke, -0.9, 0.9))
            sketch.fill(waves, color)
        elif name == "volume_low":
            sketch.fill(sketch.ring(12 * unit, 12 * unit, 4.5 * unit, stroke, -0.9, 0.9), color)
        elif name == "mute":
            cross = np.maximum(sketch.segment(15.5 * unit, 9 * unit, 21 * unit, 15 * unit, stroke),
                               sketch.segment(21 * unit, 9 * unit, 15.5 * unit, 15 * unit, stroke))
            sketch.fill(cross, color)
    elif name in ("fullscreen", "windowed"):
        inward = name == "windowed"
        mask = np.zeros((size, size), np.float32)
        for corner_x, corner_y in ((4, 4), (20, 4), (4, 20), (20, 20)):
            towards_x = 1 if corner_x < 12 else -1
            towards_y = 1 if corner_y < 12 else -1
            if inward:
                tip_x, tip_y = corner_x + towards_x * 5, corner_y + towards_y * 5
                points = [(tip_x - towards_x * 5, tip_y), (tip_x, tip_y), (tip_x, tip_y - towards_y * 5)]
            else:
                points = [(corner_x + towards_x * 5, corner_y), (corner_x, corner_y), (corner_x, corner_y + towards_y * 5)]
            mask = np.maximum(mask, sketch.polyline([(x * unit, y * unit) for x, y in points], stroke))
        sketch.fill(mask, color)
    elif name == "sidebar":
        outer = sketch.rounded_rect(3 * unit, 4.5 * unit, 21 * unit, 19.5 * unit, 2.5 * unit)
        inner = sketch.rounded_rect(3 * unit + stroke, 4.5 * unit + stroke, 21 * unit - stroke, 19.5 * unit - stroke,
                                    max(0.0, 2.5 * unit - stroke))
        sketch.fill(np.clip(outer - inner, 0, 1), color)
        sketch.fill(sketch.rounded_rect(3 * unit, 4.5 * unit, 9.5 * unit, 19.5 * unit, 2.5 * unit), color, opacity=0.9)
    elif name in ("frame_back", "frame_forward"):
        forward = name == "frame_forward"
        bar_left = 15.5 if forward else 6
        sketch.fill(sketch.rounded_rect(bar_left * unit, 6 * unit, (bar_left + 2.5) * unit, 18 * unit, 0.8 * unit),
                    color)
        points = [(7, 6), (14.5, 12), (7, 18)] if forward else [(17, 6), (9.5, 12), (17, 18)]
        sketch.fill(sketch.polygon([(x * unit, y * unit) for x, y in points]), color)
    elif name == "add":
        sketch.fill(np.maximum(sketch.segment(12 * unit, 5.5 * unit, 12 * unit, 18.5 * unit, stroke),
                               sketch.segment(5.5 * unit, 12 * unit, 18.5 * unit, 12 * unit, stroke)), color)
    elif name == "delete":
        lines = [[(5, 7), (19, 7)], [(9.5, 7), (9.5, 4.5), (14.5, 4.5), (14.5, 7)],
                 [(6.8, 7), (7.8, 19.5), (16.2, 19.5), (17.2, 7)], [(10.5, 10.5), (10.5, 16.5)],
                 [(13.5, 10.5), (13.5, 16.5)]]
        mask = np.zeros((size, size), np.float32)
        for points in lines:
            mask = np.maximum(mask, sketch.polyline([(x * unit, y * unit) for x, y in points], stroke * 0.85))
        sketch.fill(mask, color)
    else:
        raise ValueError(f"unknown icon {name}; choose from {', '.join(ICON_NAMES)}")
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
