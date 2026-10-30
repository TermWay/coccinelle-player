"""The preview area: a drop target when empty, the source frame once a file is loaded, then a before/after
comparison of the upscaled frame against a plain bicubic upscale of the same frame.

The comparison shows output pixels 1:1 (or enlarged 2x/4x/8x without smoothing), split by a divider that
follows the left mouse button. The right or middle button pans, the wheel zooms around the cursor, and the
minimap in the corner shows and moves the visible area.
"""
from __future__ import annotations

import math
import tkinter as tk
from typing import Callable

import numpy as np

from .ui_art import Sketch, png_base64, ppm_bytes
from .ui_theme import Theme

try:
    import cv2
except ImportError:  # the preview falls back to nearest-neighbour scaling without OpenCV
    cv2 = None

ZOOM_LEVELS = (1, 2, 4, 8)
MINIMAP_WIDTH = 180


def resize(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Area-averaged downscale (or bicubic upscale) of an RGB image; nearest neighbour without OpenCV."""
    if cv2 is not None:
        shrinking = width < rgb.shape[1]
        return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC)
    rows = (np.arange(height) * rgb.shape[0] / height).astype(int)
    columns = (np.arange(width) * rgb.shape[1] / width).astype(int)
    return rgb[rows][:, columns]


def bicubic_region(source: np.ndarray, output_shape: tuple[int, int], x0: int, y0: int, width: int,
                   height: int) -> np.ndarray:
    """The pixels (x0, y0, width, height) of `source` upscaled to `output_shape`, computed for that region only."""
    scale_x = source.shape[1] / output_shape[1]
    scale_y = source.shape[0] / output_shape[0]
    if cv2 is not None:
        matrix = np.array([[scale_x, 0, (x0 + 0.5) * scale_x - 0.5],
                           [0, scale_y, (y0 + 0.5) * scale_y - 0.5]], np.float32)
        return cv2.warpAffine(source, matrix, (width, height), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
    rows = np.clip(((np.arange(y0, y0 + height) + 0.5) * scale_y).astype(int), 0, source.shape[0] - 1)
    columns = np.clip(((np.arange(x0, x0 + width) + 0.5) * scale_x).astype(int), 0, source.shape[1] - 1)
    return source[rows][:, columns]


class CompareView(tk.Canvas):
    """See the module docstring. `on_browse` runs when the empty view is clicked."""

    def __init__(self, master: tk.Misc, theme: Theme, on_browse: Callable[[], None],
                 on_zoom_changed: Callable[[int], None]):
        palette = theme.palette
        super().__init__(master, background=palette.view, highlightthickness=0, borderwidth=0)
        self.theme = theme
        self.on_browse = on_browse
        self.on_zoom_changed = on_zoom_changed
        self.mode = "empty"
        self.message = ""
        self.caption = ""
        self.source: np.ndarray | None = None
        self.result: np.ndarray | None = None
        self.zoom = 1
        self.center: list[float] | None = None
        self.split = 0.5
        self.hover = False
        self._render_pending = False
        self._photos: dict[str, tk.PhotoImage] = {}
        self._pills: dict[str, tk.PhotoImage] = {}
        self._cache: dict[str, tuple] = {}
        self._geometry: tuple[int, int, int, int, int, int, int, int] | None = None
        self._minimap_box: tuple[int, int, int, int] | None = None
        self._drag: str | None = None
        self._pan_origin: tuple[int, int, float, float] | None = None
        self._handle = self._make_handle()
        self._icon = self._make_drop_icon(palette.text_faint)
        self._icon_hover = self._make_drop_icon(palette.accent)
        self.bind("<Configure>", lambda _: self.request_render())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag_motion)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Double-Button-1>", lambda _: self.set_zoom(1))
        for button in ("2", "3"):
            self.bind(f"<ButtonPress-{button}>", self._pan_press)
            self.bind(f"<B{button}-Motion>", self._pan_motion)
            self.bind(f"<ButtonRelease-{button}>", self._release)
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Motion>", self._motion)
        self.bind("<Enter>", lambda _: self._set_hover(True))
        self.bind("<Leave>", lambda _: self._set_hover(False))

    # ---- public ---------------------------------------------------------------------------------

    def show_empty(self) -> None:
        self.mode, self.source, self.result = "empty", None, None
        self.request_render()

    def show_message(self, message: str) -> None:
        self.mode, self.message = "message", message
        self.request_render()

    def show_source(self, rgba: np.ndarray, caption: str) -> None:
        self.mode, self.caption = "source", caption
        self.source = np.ascontiguousarray(rgba[..., :3])
        self.result = None
        self.center = None
        self._cache.clear()
        self.request_render()

    def show_result(self, source_rgba: np.ndarray, result_rgba: np.ndarray) -> None:
        """Show (or update, keeping zoom and position) the comparison for one frame."""
        if self.mode != "compare" or self.result is None or self.result.shape[:2] != result_rgba.shape[:2]:
            self.center = None
            self.split = 0.5
        self.mode = "compare"
        self.source = np.ascontiguousarray(source_rgba[..., :3])
        self.result = np.ascontiguousarray(result_rgba[..., :3])
        self._cache.pop("minimap", None)
        self.request_render()

    @property
    def comparing(self) -> bool:
        return self.mode == "compare" and self.result is not None

    def set_zoom(self, zoom: int, anchor: tuple[int, int] | None = None) -> None:
        """Change the magnification, keeping the output pixel under `anchor` (view coordinates) in place."""
        zoom = min(ZOOM_LEVELS, key=lambda level: abs(level - zoom))
        if not self.comparing or zoom == self.zoom:
            self.zoom = zoom if self.comparing else 1
            self.on_zoom_changed(self.zoom)
            return
        if anchor is not None and self._geometry is not None:
            offset_x, offset_y, _, _, x0, y0, _, _ = self._geometry
            pixel_x = x0 + (anchor[0] - offset_x) / self.zoom
            pixel_y = y0 + (anchor[1] - offset_y) / self.zoom
            self.center = [pixel_x + (self.winfo_width() / 2 - anchor[0]) / zoom,
                           pixel_y + (self.winfo_height() / 2 - anchor[1]) / zoom]
        self.zoom = zoom
        self.on_zoom_changed(zoom)
        self.request_render()

    def request_render(self) -> None:
        if not self._render_pending:
            self._render_pending = True
            self.after_idle(self._render)

    # ---- drawing --------------------------------------------------------------------------------

    def _render(self) -> None:
        self._render_pending = False
        self.delete("all")
        self._geometry = None
        self._minimap_box = None
        width, height = self.winfo_width(), self.winfo_height()
        if width < 8 or height < 8:
            return
        if self.comparing:
            self._render_compare(width, height)
        elif self.mode == "source" and self.source is not None:
            self._render_source(width, height)
        elif self.mode == "message":
            self._centered_text(width, height, self.message)
        else:
            self._render_empty(width, height)
        self._draw_frame(width, height)

    def _render_empty(self, width: int, height: int) -> None:
        px, palette, fonts = self.theme.px, self.theme.palette, self.theme.fonts
        margin = px(18)
        outline = palette.accent if self.hover else palette.border_hover
        self._rounded_outline(margin, margin, width - margin, height - margin, px(12), outline)
        middle = height // 2
        self.create_image(width // 2, middle - px(58), image=self._icon_hover if self.hover else self._icon)
        self.create_text(width // 2, middle, text="Drop a video or an image here", fill=palette.text,
                         font=fonts["display"])
        self.create_text(width // 2, middle + px(32), text="or click to browse", fill=palette.text_secondary,
                         font=fonts["body"])
        self.create_text(width // 2, middle + px(64),
                         text="MP4 · MKV · MOV · AVI · WMV · WebM · MPEG · FLV · TS · PNG · JPG · WebP",
                         fill=palette.text_faint, font=fonts["small"])
        self.configure(cursor="hand2")

    def _centered_text(self, width: int, height: int, text: str) -> None:
        self.create_text(width // 2, height // 2, text=text, fill=self.theme.palette.text_secondary,
                         font=self.theme.fonts["body"])
        self.configure(cursor="")

    def _render_source(self, width: int, height: int) -> None:
        px = self.theme.px
        source_height, source_width = self.source.shape[:2]
        fit = min((width - 2 * px(18)) / source_width, (height - 2 * px(18)) / source_height)
        shown_width, shown_height = max(1, int(source_width * fit)), max(1, int(source_height * fit))
        key = (shown_width, shown_height, id(self.source))
        if self._cache.get("fit", (None,))[0] != key:
            self._cache["fit"] = (key, self._photo(resize(self.source, shown_width, shown_height)))
        self.create_image(width // 2, height // 2, image=self._cache["fit"][1])
        left, top = (width - shown_width) // 2, (height - shown_height) // 2
        self._pill(left + px(10), top + px(10), self.caption, "nw")
        self.configure(cursor="")

    def _render_compare(self, width: int, height: int) -> None:
        px = self.theme.px
        result_height, result_width = self.result.shape[:2]
        zoom = self.zoom
        region_width = min(result_width, math.ceil(width / zoom))
        region_height = min(result_height, math.ceil(height / zoom))
        center_x, center_y = self.center or (result_width / 2, result_height / 2)
        x0 = int(round(min(max(center_x - region_width / 2, 0), result_width - region_width)))
        y0 = int(round(min(max(center_y - region_height / 2, 0), result_height - region_height)))
        self.center = [x0 + region_width / 2, y0 + region_height / 2]
        after = self.result[y0:y0 + region_height, x0:x0 + region_width]
        before = bicubic_region(self.source, (result_height, result_width), x0, y0, region_width, region_height)
        if zoom > 1:
            after = np.repeat(np.repeat(after, zoom, axis=0), zoom, axis=1)[:height, :width]
            before = np.repeat(np.repeat(before, zoom, axis=0), zoom, axis=1)[:height, :width]
        shown_height, shown_width = after.shape[:2]
        offset_x, offset_y = (width - shown_width) // 2, (height - shown_height) // 2
        self._geometry = (offset_x, offset_y, shown_width, shown_height, x0, y0, region_width, region_height)
        self._photos["before"] = self._photo(before)
        self._photos["after"] = self._photo(after)
        self._photos["clip"] = tk.PhotoImage(master=self, width=1, height=1)
        self.create_image(offset_x, offset_y, image=self._photos["after"], anchor="nw")
        self.create_image(offset_x, offset_y, image=self._photos["clip"], anchor="nw", tags="clip")
        self.create_line(0, 0, 0, 0, fill="#f4f4f5", width=max(1, px(2)), tags="divider")
        self.create_image(0, 0, image=self._handle, tags="handle")
        self._pill(offset_x + px(10), offset_y + px(10), "Bicubic", "nw")
        self._pill(offset_x + shown_width - px(10), offset_y + px(10), "FSR 4", "ne")
        if zoom > 1:
            self._pill(offset_x + px(10), offset_y + shown_height - px(10), f"{zoom * 100}%", "sw")
        if region_width < result_width or region_height < result_height:
            self._draw_minimap(width, height, x0, y0, region_width, region_height)
        self._place_divider()

    def _place_divider(self) -> None:
        if self._geometry is None:
            return
        offset_x, offset_y, shown_width, shown_height = self._geometry[:4]
        split = int(round(self.split * shown_width))
        clip = self._photos["clip"]
        if split <= 0:
            self.itemconfigure("clip", state="hidden")
        else:
            clip.blank()
            clip.configure(width=split, height=shown_height)
            self.tk.call(clip, "copy", self._photos["before"], "-from", 0, 0, split, shown_height, "-to", 0, 0)
            self.itemconfigure("clip", state="normal")
        x = offset_x + split
        self.coords("divider", x, offset_y, x, offset_y + shown_height)
        self.coords("handle", x, offset_y + shown_height // 2)
        self.tag_raise("divider")
        self.tag_raise("handle")
        self.tag_raise("overlay")

    def _draw_minimap(self, width: int, height: int, x0: int, y0: int, region_width: int, region_height: int) -> None:
        px, palette = self.theme.px, self.theme.palette
        result_height, result_width = self.result.shape[:2]
        map_width = min(px(MINIMAP_WIDTH), width // 3)
        map_height = max(1, int(map_width * result_height / result_width))
        key = (map_width, map_height)
        if self._cache.get("minimap", (None,))[0] != key:
            self._cache["minimap"] = (key, self._photo(resize(self.result, map_width, map_height)))
        right, bottom = width - px(12), height - px(12)
        left, top = right - map_width, bottom - map_height
        self.create_rectangle(left - 2, top - 2, right + 1, bottom + 1, outline=palette.border_hover,
                              fill=palette.view, tags="overlay")
        self.create_image(left, top, image=self._cache["minimap"][1], anchor="nw", tags="overlay")
        scale = map_width / result_width
        self.create_rectangle(left + x0 * scale, top + y0 * scale, left + (x0 + region_width) * scale,
                              top + (y0 + region_height) * scale, outline=palette.accent, width=max(1, px(1.5)),
                              tags="overlay")
        self._minimap_box = (left, top, right, bottom)

    def _draw_frame(self, width: int, height: int) -> None:
        """Rounded corners and a 1px border, drawn over everything."""
        palette, radius = self.theme.palette, self.theme.px(10)
        key = (radius, palette.background, palette.border)
        if self._cache.get("corners", (None,))[0] != key:
            sketch = Sketch(radius, radius)
            inside = sketch.circle(radius, radius, radius)
            sketch.fill(1 - inside, palette.background)
            sketch.fill(np.clip(inside - sketch.circle(radius, radius, radius - 1), 0, 1), palette.border)
            pixels = sketch.rgba()
            corners = [self._photo_rgba(pixels), self._photo_rgba(pixels[:, ::-1]),
                       self._photo_rgba(pixels[::-1, :]), self._photo_rgba(pixels[::-1, ::-1])]
            self._cache["corners"] = (key, corners)
        top_left, top_right, bottom_left, bottom_right = self._cache["corners"][1]
        self.create_rectangle(0, 0, width - 1, height - 1, outline=palette.border)
        self.create_image(0, 0, image=top_left, anchor="nw")
        self.create_image(width, 0, image=top_right, anchor="ne")
        self.create_image(0, height, image=bottom_left, anchor="sw")
        self.create_image(width, height, image=bottom_right, anchor="se")

    def _rounded_outline(self, left: int, top: int, right: int, bottom: int, radius: int, color: str) -> None:
        key = (right - left, bottom - top, radius, color)
        if self._cache.get("outline", (None,))[0] != key:
            sketch = Sketch(right - left, bottom - top)
            outer = sketch.rounded_rect(0, 0, right - left, bottom - top, radius)
            inner = sketch.rounded_rect(1.5, 1.5, right - left - 1.5, bottom - top - 1.5, radius - 1.5)
            sketch.fill(np.clip(outer - inner, 0, 1), color)
            self._cache["outline"] = (key, self._photo_rgba(sketch.rgba()))
        self.create_image(left, top, image=self._cache["outline"][1], anchor="nw")

    def _pill(self, x: int, y: int, text: str, anchor: str) -> None:
        px, fonts = self.theme.px, self.theme.fonts
        if text not in self._pills:
            width = fonts["small"].measure(text) + 2 * px(9)
            height = px(22)
            sketch = Sketch(width, height)
            sketch.fill(sketch.rounded_rect(0, 0, width, height, height / 2), "#000000", opacity=0.6)
            self._pills[text] = self._photo_rgba(sketch.rgba())
        image = self._pills[text]
        self.create_image(x, y, image=image, anchor=anchor, tags="overlay")
        left = {"nw": x, "sw": x, "ne": x - image.width(), "se": x - image.width()}[anchor]
        top = {"nw": y, "ne": y, "sw": y - image.height(), "se": y - image.height()}[anchor]
        self.create_text(left + image.width() // 2, top + image.height() // 2, text=text, fill="#f4f4f5",
                         font=fonts["small"], tags="overlay")

    def _photo(self, rgb: np.ndarray) -> tk.PhotoImage:
        return tk.PhotoImage(master=self, data=ppm_bytes(rgb), format="PPM")

    def _photo_rgba(self, rgba: np.ndarray) -> tk.PhotoImage:
        return tk.PhotoImage(master=self, data=png_base64(np.ascontiguousarray(rgba)))

    def _make_handle(self) -> tk.PhotoImage:
        size = self.theme.px(30)
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
        return self._photo_rgba(sketch.rgba())

    def _make_drop_icon(self, color: str) -> tk.PhotoImage:
        size = self.theme.px(44)
        unit = size / 32
        sketch = Sketch(size, size)
        thickness = 2.2 * unit
        tray = sketch.polyline([(5 * unit, 18 * unit), (5 * unit, 26 * unit), (27 * unit, 26 * unit),
                                (27 * unit, 18 * unit)], thickness)
        arrow = sketch.polyline([(16 * unit, 20 * unit), (16 * unit, 5 * unit)], thickness)
        head = sketch.polyline([(10 * unit, 11 * unit), (16 * unit, 5 * unit), (22 * unit, 11 * unit)], thickness)
        sketch.fill(np.maximum(np.maximum(tray, arrow), head), color)
        return self._photo_rgba(sketch.rgba())

    # ---- input ----------------------------------------------------------------------------------

    def _set_hover(self, hover: bool) -> None:
        self.hover = hover
        if self.mode == "empty":
            self.request_render()

    def _in_minimap(self, x: int, y: int) -> bool:
        box = self._minimap_box
        return box is not None and box[0] <= x <= box[2] and box[1] <= y <= box[3]

    def _press(self, event: tk.Event) -> None:
        if self.mode == "empty":
            self.on_browse()
            return
        if not self.comparing:
            return
        if self._in_minimap(event.x, event.y):
            self._drag = "minimap"
            self._move_to_minimap(event.x, event.y)
        else:
            self._drag = "divider"
            self._move_divider(event.x)

    def _drag_motion(self, event: tk.Event) -> None:
        if self._drag == "minimap":
            self._move_to_minimap(event.x, event.y)
        elif self._drag == "divider":
            self._move_divider(event.x)

    def _release(self, _event: tk.Event) -> None:
        self._drag = None
        self._pan_origin = None

    def _move_divider(self, x: int) -> None:
        if self._geometry is None:
            return
        offset_x, _, shown_width = self._geometry[:3]
        self.split = min(1.0, max(0.0, (x - offset_x) / max(shown_width, 1)))
        self._place_divider()

    def _move_to_minimap(self, x: int, y: int) -> None:
        left, top, right, _ = self._minimap_box
        scale = (right - left) / self.result.shape[1]
        self.center = [(x - left) / scale, (y - top) / scale]
        self.request_render()

    def _pan_press(self, event: tk.Event) -> None:
        if self.comparing and self.center is not None:
            self._pan_origin = (event.x, event.y, self.center[0], self.center[1])
            self.configure(cursor="fleur")

    def _pan_motion(self, event: tk.Event) -> None:
        if self._pan_origin is None:
            return
        start_x, start_y, center_x, center_y = self._pan_origin
        self.center = [center_x - (event.x - start_x) / self.zoom, center_y - (event.y - start_y) / self.zoom]
        self.request_render()

    def _wheel(self, event: tk.Event) -> None:
        if not self.comparing:
            return
        index = ZOOM_LEVELS.index(self.zoom)
        index = min(len(ZOOM_LEVELS) - 1, index + 1) if event.delta > 0 else max(0, index - 1)
        self.set_zoom(ZOOM_LEVELS[index], anchor=(event.x, event.y))

    def _motion(self, event: tk.Event) -> None:
        if self.mode == "empty":
            cursor = "hand2"
        elif not self.comparing:
            cursor = ""
        elif self._in_minimap(event.x, event.y):
            cursor = "hand2"
        else:
            cursor = "sb_h_double_arrow"
        if self._pan_origin is None and str(self.cget("cursor")) != cursor:
            self.configure(cursor=cursor)
