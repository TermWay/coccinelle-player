"""The dark theme: palette, fonts, ttk styles drawn from anti-aliased images, and the Windows title bar.

ttk's `clam` theme is the base; buttons, fields, chips, switches, sliders, scrollbars and the progress bar are
replaced by image elements (9-slice scaled) rendered by `ui_art`, so corners are rounded and edges smooth.
Every size goes through `Theme.px`, which follows the display's DPI.

Tk redraws a widget's image element as many tiles, one GDI call each, and an image with partly transparent
pixels costs a read back and a blend per tile: a field took 30 ms, the whole window half a second. So the images
are opaque over the colour of the element's container wherever nothing else shows through, and 9-slice images
have a middle `SLICE_LENGTH` long (the elements keep their small requested size): a field now takes 1 ms.
"""
from __future__ import annotations

import ctypes
import os
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from tkinter import ttk

from . import log
from .ui_art import Sketch, app_icon, png_base64, stretch_middle


@dataclass(frozen=True)
class Palette:
    background: str = "#141518"
    view: str = "#0c0d0f"
    field: str = "#1e2024"
    field_hover: str = "#25282d"
    field_pressed: str = "#1a1c20"
    border: str = "#2d3036"
    border_hover: str = "#3c4047"
    text: str = "#ececee"
    text_secondary: str = "#a3a7ae"
    text_faint: str = "#6c7079"
    accent: str = "#ff6a3d"
    accent_hover: str = "#ff8157"
    accent_pressed: str = "#e65b31"
    on_accent: str = "#1c0e08"
    accent_soft: str = "#33211b"
    accent_text: str = "#ffb39c"
    success: str = "#3fb950"
    error: str = "#f2555a"
    error_soft: str = "#3b1a1c"
    error_text: str = "#ffb4b6"
    scroll_thumb: str = "#3a3d44"
    scroll_thumb_hover: str = "#4b4f57"


PALETTE = Palette()
BOX_SIZE = 24  # 96 DPI pixels: the rounded boxes as drawn, and the smallest size their elements ask for
SLICE_LENGTH = 320  # 96 DPI pixels: the middle of 9-slice images is repeated to this length (see above)
SLICE_HEIGHT = 64  # the same across, for boxes: buttons and fields are less tall than this

FONT_CHOICES = {
    "body": (("Segoe UI Variable Text", "Segoe UI"), 10),
    "strong": (("Segoe UI Variable Text Semibold", "Segoe UI Semibold"), 10),
    "small": (("Segoe UI Variable Text", "Segoe UI"), 9),
    "section": (("Segoe UI Variable Text Semibold", "Segoe UI Semibold"), 8),
    "title": (("Segoe UI Variable Display Semib", "Segoe UI Semibold"), 13),
    "display": (("Segoe UI Variable Display Semib", "Segoe UI Semibold"), 17),
    "mono": (("Cascadia Mono", "Consolas"), 9),
}

DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_BORDER_COLOR = 34
DWMWA_CAPTION_COLOR = 35
DWMWA_TEXT_COLOR = 36
PREFERRED_APP_MODE_FORCE_DARK = 2


def enable_dark_dialogs() -> None:
    """Ask Windows to draw this process's common dialogs (open/save file) dark. Call before creating windows.

    Uses uxtheme's SetPreferredAppMode (ordinal 135), the call Windows' own dark apps use; it has no public
    name, so a missing ordinal on an older Windows is reported and ignored.
    """
    if os.name != "nt":
        return
    try:
        uxtheme = ctypes.WinDLL("uxtheme")
        set_preferred_app_mode = uxtheme[135]
        set_preferred_app_mode.argtypes = [ctypes.c_int]
        set_preferred_app_mode.restype = ctypes.c_int
        set_preferred_app_mode(PREFERRED_APP_MODE_FORCE_DARK)
        uxtheme[136]()  # FlushMenuThemes
    except (AttributeError, OSError) as error:
        log.emit(f"[ui] dark dialogs unavailable on this Windows: {error}")


def _colorref(color: str) -> ctypes.c_uint32:
    red, green, blue = (int(color[index:index + 2], 16) for index in (1, 3, 5))
    return ctypes.c_uint32(red | green << 8 | blue << 16)


def style_title_bar(window: tk.Misc, palette: Palette = PALETTE) -> None:
    """Dark title bar in the window's own colours (Windows 11; Windows 10 gets the plain dark bar)."""
    if os.name != "nt":
        return
    window.update_idletasks()
    handle = int(window.wm_frame(), 16)
    dwmapi = ctypes.windll.dwmapi
    enabled = ctypes.c_int(1)
    result = dwmapi.DwmSetWindowAttribute(handle, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(enabled), 4)
    if result != 0:
        log.emit(f"[ui] dark title bar not supported (DWM error {result & 0xFFFFFFFF:#x})")
        return
    for attribute, color in ((DWMWA_CAPTION_COLOR, palette.background), (DWMWA_BORDER_COLOR, palette.border),
                             (DWMWA_TEXT_COLOR, palette.text)):
        value = _colorref(color)
        # Caption, border and text colours need Windows 11; earlier versions keep the dark default.
        dwmapi.DwmSetWindowAttribute(handle, attribute, ctypes.byref(value), 4)


class Theme:
    """Fonts, images and ttk styles for one Tk root."""

    def __init__(self, root: tk.Tk, palette: Palette = PALETTE):
        self.root = root
        self.palette = palette
        self.scale = max(1.0, root.winfo_fpixels("1i") / 96)
        self._images: list[tk.PhotoImage] = []
        self.fonts = self._make_fonts()
        self.style = ttk.Style(root)
        self.style.theme_use("clam")
        self._configure_base()
        self._configure_buttons()
        self._configure_fields()
        self._configure_choices()
        self._configure_bars()
        self._configure_options()
        root.configure(background=palette.background)

    # ---- helpers --------------------------------------------------------------------------------

    def px(self, value: float) -> int:
        """Device pixels for a size given at 96 DPI."""
        return int(round(value * self.scale))

    def photo(self, sketch: Sketch, container: str | None = None,
              stretch: tuple[float, float] = (0, 0)) -> tk.PhotoImage:
        """A Tk image of `sketch`, opaque over `container` (the colour around the element) when given, its
        middle repeated to `stretch` (96 DPI pixels across and down, 0 to keep) for 9-slice elements."""
        rgba = sketch.rgba() if container is None else sketch.opaque(container)
        rgba = stretch_middle(rgba, self.px(stretch[0]), self.px(stretch[1]))
        image = tk.PhotoImage(master=self.root, data=png_base64(rgba))
        self._images.append(image)
        return image

    def box(self, fill: str, border: str | None = None, radius: float = 6, border_width: float = 1,
            opacity: float = 1.0, container: str | None = None) -> tk.PhotoImage:
        """A rounded rectangle image for 9-slice elements (give them `box_extent()`), opaque over `container`
        (the background by default)."""
        side = self.px(BOX_SIZE)
        corner = self.px(radius)
        sketch = Sketch(side, side)
        outer = sketch.rounded_rect(0, 0, side, side, corner)
        if border is None:
            sketch.fill(outer, fill, opacity)
        else:
            width = max(1, self.px(border_width))
            sketch.fill(outer, border, opacity)
            sketch.fill(sketch.rounded_rect(width, width, side - width, side - width, max(corner - width, 0)),
                        fill, opacity)
        return self.photo(sketch, container or self.palette.background, (SLICE_LENGTH, SLICE_HEIGHT))

    def box_extent(self) -> dict[str, int]:
        """Element options for a `box` image: ask for the box's drawn size, not its stretched one."""
        return {"width": self.px(BOX_SIZE), "height": self.px(BOX_SIZE)}

    def slice_border(self, radius: float = 6) -> int:
        return self.px(radius) + 2

    def icon(self, size: float) -> tk.PhotoImage:
        """The app icon at `size` (96 DPI pixels), for use inside the window."""
        image = tk.PhotoImage(master=self.root, data=png_base64(app_icon(self.px(size))))
        self._images.append(image)
        return image

    def icon_photos(self) -> list[tk.PhotoImage]:
        photos = []
        for size in (16, 24, 32, 48, 64, 256):
            image = tk.PhotoImage(master=self.root, data=png_base64(app_icon(size)))
            self._images.append(image)
            photos.append(image)
        return photos

    def _make_fonts(self) -> dict[str, tkfont.Font]:
        available = set(tkfont.families(self.root))
        fonts = {}
        for name, (families, size) in FONT_CHOICES.items():
            family = next((choice for choice in families if choice in available), "TkDefaultFont")
            fonts[name] = tkfont.Font(root=self.root, family=family, size=size)
        for named, source in (("TkDefaultFont", "body"), ("TkTextFont", "body"), ("TkMenuFont", "body"),
                              ("TkHeadingFont", "strong"), ("TkCaptionFont", "strong"),
                              ("TkSmallCaptionFont", "small"), ("TkTooltipFont", "small"), ("TkFixedFont", "mono")):
            tkfont.nametofont(named, root=self.root).configure(**fonts[source].actual())
        return fonts

    # ---- styles ---------------------------------------------------------------------------------

    def _configure_base(self) -> None:
        p = self.palette
        style = self.style
        style.configure(".", background=p.background, foreground=p.text, font=self.fonts["body"],
                        bordercolor=p.border, lightcolor=p.background, darkcolor=p.background,
                        troughcolor=p.background, focuscolor=p.accent, selectbackground=p.accent_soft,
                        selectforeground=p.text, insertcolor=p.text, fieldbackground=p.field,
                        arrowcolor=p.text_secondary, relief="flat")
        style.map(".", foreground=[("disabled", p.text_faint)],
                  background=[("disabled", p.background), ("pressed", p.background), ("active", p.background)])
        style.configure("TFrame", background=p.background)
        style.configure("TLabel", background=p.background, foreground=p.text)
        style.configure("Muted.TLabel", foreground=p.text_secondary)
        style.configure("Faint.TLabel", foreground=p.text_faint, font=self.fonts["small"])
        style.configure("Small.TLabel", foreground=p.text_secondary, font=self.fonts["small"])
        style.configure("Section.TLabel", foreground=p.text_faint, font=self.fonts["section"])
        style.configure("Strong.TLabel", font=self.fonts["strong"])
        style.configure("Title.TLabel", font=self.fonts["title"])
        style.configure("Status.TLabel", foreground=p.text_secondary)
        style.configure("Time.TLabel", foreground=p.text_secondary, font=self.fonts["mono"])
        style.configure("Banner.TFrame", background=p.error_soft)
        style.configure("Banner.TLabel", background=p.error_soft, foreground=p.error_text)
        style.configure("TSeparator", background=p.border)

    def _button_style(self, name: str, fill: str, hover: str, pressed: str, border: str | None,
                      foreground: str, font: str, disabled_fill: str, disabled_border: str | None,
                      disabled_foreground: str, padding: tuple[int, int], container: str | None = None) -> None:
        container = container or self.palette.background
        radius = 6
        normal = self.box(fill, border, radius, container=container)
        element = f"{name}.border"
        self.style.element_create(
            element, "image", normal,
            ("disabled", self.box(disabled_fill, disabled_border, radius, container=container)),
            ("pressed", self.box(pressed, border, radius, container=container)),
            ("active", self.box(hover, border and self.palette.border_hover, radius, container=container)),
            ("focus", self.box(fill, self.palette.accent, radius, container=container)),
            border=self.slice_border(radius), padding=0, sticky="nsew", **self.box_extent())
        self.style.layout(f"{name}.TButton", [(element, {"sticky": "nsew", "children": [
            ("Button.padding", {"sticky": "nsew", "children": [("Button.label", {"sticky": "nsew"})]})]})])
        self.style.configure(f"{name}.TButton", foreground=foreground, font=self.fonts[font],
                             padding=(self.px(padding[0]), self.px(padding[1])), anchor="center",
                             background=container)
        self.style.map(f"{name}.TButton", foreground=[("disabled", disabled_foreground)],
                       background=[("disabled", container), ("pressed", container), ("active", container)])

    def _configure_buttons(self) -> None:
        p = self.palette
        self._button_style("Secondary", p.field, p.field_hover, p.field_pressed, p.border, p.text, "body",
                           p.background, p.border, p.text_faint, (14, 6))
        self._button_style("Accent", p.accent, p.accent_hover, p.accent_pressed, None, p.on_accent, "strong",
                           p.accent_soft, None, "#8a5a4a", (20, 6))
        self._button_style("Ghost", p.background, p.field, p.field_pressed, None, p.text_secondary, "body",
                           p.background, None, p.text_faint, (10, 6))
        self._button_style("Banner", p.error_soft, "#4a2225", "#331618", None, p.error_text, "body",
                           p.error_soft, None, p.text_faint, (8, 2), container=p.error_soft)
        self._button_style("Icon", p.background, p.field_hover, p.field_pressed, None, p.text, "body",
                           p.background, None, p.text_faint, (5, 4))

    def _configure_fields(self) -> None:
        p = self.palette
        radius = 6
        field = self.box(p.field, p.border, radius)
        hover = self.box(p.field, p.border_hover, radius)
        focus = self.box(p.field, p.accent, radius)
        disabled = self.box(p.background, p.border, radius)
        border = self.slice_border(radius)
        self.style.element_create("Dark.Entry.field", "image", field, ("disabled", disabled), ("focus", focus),
                                  ("hover", hover), border=border, padding=0, sticky="nsew", **self.box_extent())
        self.style.layout("TEntry", [("Dark.Entry.field", {"sticky": "nsew", "children": [
            ("Entry.padding", {"sticky": "nsew", "children": [("Entry.textarea", {"sticky": "nsew"})]})]})])
        self.style.configure("TEntry", padding=(self.px(10), self.px(6)), foreground=p.text,
                             selectbackground=p.accent_soft, selectforeground=p.text, insertcolor=p.text)
        self.style.map("TEntry", foreground=[("disabled", p.text_faint)])

        combo_hover = self.box(p.field_hover, p.border_hover, radius)
        self.style.element_create("Dark.Combobox.field", "image", field, ("disabled", disabled),
                                  ("pressed", focus), ("focus", focus), ("hover", combo_hover),
                                  border=border, padding=0, sticky="nsew", **self.box_extent())
        chevron = self._chevron(p.text_secondary)
        self.style.element_create("Dark.Combobox.arrow", "image", chevron,
                                  ("disabled", self._chevron(p.text_faint)), sticky="")
        self.style.layout("TCombobox", [("Dark.Combobox.field", {"sticky": "nsew", "children": [
            ("Dark.Combobox.arrow", {"side": "right", "sticky": "ns"}),
            ("Combobox.padding", {"expand": "1", "sticky": "nsew", "children": [
                ("Combobox.textarea", {"sticky": "nsew"})]})]})])
        self.style.configure("TCombobox", padding=(self.px(10), self.px(6), self.px(4), self.px(6)),
                             foreground=p.text, arrowsize=self.px(12))
        self.style.map("TCombobox", foreground=[("disabled", p.text_faint), ("readonly", p.text)],
                       fieldbackground=[("readonly", p.field)],
                       selectbackground=[("readonly", p.field)], selectforeground=[("readonly", p.text)])
        self.style.configure("ComboboxPopdownFrame", background=p.field, bordercolor=p.border_hover,
                             lightcolor=p.field, darkcolor=p.field, relief="solid", borderwidth=1)

    def _chevron(self, color: str) -> tk.PhotoImage:
        width, height = self.px(26), self.px(16)
        sketch = Sketch(width, height)
        center_x, center_y, half = width / 2 - self.px(3), height / 2 + self.px(0.5), self.px(3.5)
        sketch.fill(sketch.polyline([(center_x - half, center_y - half / 2), (center_x, center_y + half / 2),
                                     (center_x + half, center_y - half / 2)], self.px(1.5)), color)
        return self.photo(sketch)

    def _configure_choices(self) -> None:
        p = self.palette
        radius = 6
        border = self.slice_border(radius)
        self.style.element_create(
            "Segment.border", "image", self.box(p.field, p.border, radius),
            ("disabled", self.box(p.background, "#24262b", radius)),
            ("selected", self.box(p.accent_soft, p.accent, radius)),
            ("active", self.box(p.field_hover, p.border_hover, radius)),
            border=border, padding=0, sticky="nsew", **self.box_extent())
        self.style.layout("Segment.TRadiobutton", [("Segment.border", {"sticky": "nsew", "children": [
            ("Radiobutton.padding", {"sticky": "nsew", "children": [("Radiobutton.label", {"sticky": "nsew"})]})]})])
        self.style.configure("Segment.TRadiobutton", padding=(self.px(10), self.px(5)), anchor="center",
                             foreground=p.text_secondary, font=self.fonts["body"])
        self.style.map("Segment.TRadiobutton", foreground=[("disabled", "#4a4d55"), ("selected", p.accent_text),
                                                          ("active", p.text)])

        switch_width, switch_height = self.px(34), self.px(20)

        def switch(track: str, knob: str, on: bool, track_border: str | None = None) -> tk.PhotoImage:
            sketch = Sketch(switch_width, switch_height)
            sketch.fill(sketch.rounded_rect(0, 0, switch_width, switch_height, switch_height / 2), track_border or track)
            if track_border:
                inset = max(1, self.px(1))
                sketch.fill(sketch.rounded_rect(inset, inset, switch_width - inset, switch_height - inset,
                                                switch_height / 2 - inset), track)
            radius = switch_height / 2 - self.px(3.5)
            center_x = switch_width - switch_height / 2 if on else switch_height / 2
            sketch.fill(sketch.circle(center_x, switch_height / 2, radius), knob)
            return self.photo(sketch, p.background)

        self.style.element_create(
            "Switch.indicator", "image", switch(p.field, p.text_secondary, False, p.border_hover),
            ("disabled", "selected", switch(p.accent_soft, "#6b4a40", True)),
            ("disabled", switch(p.background, p.text_faint, False, p.border)),
            ("selected", "active", switch(p.accent_hover, p.on_accent, True)),
            ("selected", switch(p.accent, p.on_accent, True)),
            ("active", switch(p.field_hover, p.text, False, p.border_hover)),
            sticky="")
        self.style.layout("Switch.TCheckbutton", [("Switch.indicator", {"side": "left", "sticky": ""}),
                                                  ("Checkbutton.padding", {"sticky": "nsew", "children": [
                                                      ("Checkbutton.label", {"sticky": "nsew"})]})])
        self.style.configure("Switch.TCheckbutton", padding=(self.px(10), 0, 0, 0), foreground=p.text)
        self.style.map("Switch.TCheckbutton", foreground=[("disabled", p.text_faint)],
                       background=[("active", p.background)])

        knob_size = self.px(16)

        def knob(fill: str, ring: str) -> tk.PhotoImage:
            sketch = Sketch(knob_size, knob_size)
            sketch.fill(sketch.circle(knob_size / 2, knob_size / 2, knob_size / 2 - 0.5), ring)
            sketch.fill(sketch.circle(knob_size / 2, knob_size / 2, knob_size / 2 - self.px(3.5)), fill)
            return self.photo(sketch)

        track_width = self.px(24)
        track = Sketch(track_width, knob_size)
        track_height = self.px(4)
        top = (knob_size - track_height) / 2
        track.fill(track.rounded_rect(0, top, track_width, top + track_height, track_height / 2), p.border_hover)
        # The knob keeps its transparent corners: it is drawn over the track.
        self.style.element_create("Dark.Scale.trough", "image", self.photo(track, p.background, (SLICE_LENGTH, 0)),
                                  border=(self.px(4), 0, self.px(4), 0), sticky="ew", width=track_width)
        self.style.element_create("Dark.Scale.slider", "image", knob(p.on_accent, p.accent),
                                  ("disabled", knob(p.background, p.text_faint)),
                                  ("pressed", knob(p.on_accent, p.accent_pressed)),
                                  ("active", knob(p.on_accent, p.accent_hover)), sticky="")
        self.style.layout("Horizontal.TScale", [("Dark.Scale.trough", {"sticky": "nsew", "children": [
            ("Dark.Scale.slider", {"side": "left", "sticky": ""})]})])

    def _configure_bars(self) -> None:
        p = self.palette
        height = self.px(6)
        width = self.px(16)

        def bar(color: str, container: str | None) -> tk.PhotoImage:
            sketch = Sketch(width, height)
            sketch.fill(sketch.rounded_rect(0, 0, width, height, height / 2), color)
            return self.photo(sketch, container, (SLICE_LENGTH, 0))

        border = (height // 2 + 1, 0, height // 2 + 1, 0)
        self.style.element_create("Thin.Progressbar.trough", "image", bar(p.field, p.background), border=border,
                                  padding=0, sticky="nsew", width=width)
        # The bar keeps its transparent ends: it is drawn over the trough.
        self.style.element_create("Thin.Progressbar.pbar", "image", bar(p.accent, None), border=border, padding=0,
                                  sticky="nsew", width=width)
        self.style.layout("Thin.Horizontal.TProgressbar", [("Thin.Progressbar.trough", {"sticky": "nsew", "children": [
            ("Thin.Progressbar.pbar", {"side": "left", "sticky": "ns"})]})])
        self.style.configure("Thin.Horizontal.TProgressbar", thickness=height, background=p.background,
                             troughcolor=p.background, bordercolor=p.background)

        thumb_width = self.px(10)
        thumb_height = self.px(28)

        def thumb(color: str) -> tk.PhotoImage:
            sketch = Sketch(thumb_width, thumb_height)
            inset = self.px(2)
            sketch.fill(sketch.rounded_rect(inset, inset, thumb_width - inset, thumb_height - inset,
                                            (thumb_width - 2 * inset) / 2), color)
            return self.photo(sketch, p.field, (0, SLICE_LENGTH))  # over the trough, which is field coloured

        slice_ = thumb_width // 2
        self.style.element_create("Dark.Scrollbar.thumb", "image", thumb(p.scroll_thumb),
                                  ("pressed", thumb(p.text_faint)), ("active", thumb(p.scroll_thumb_hover)),
                                  border=(slice_, slice_ + 1, slice_, slice_ + 1), sticky="nsew", height=thumb_height)
        self.style.layout("Vertical.TScrollbar", [("Vertical.Scrollbar.trough", {"sticky": "ns", "children": [
            ("Dark.Scrollbar.thumb", {"expand": "1", "sticky": "nsew"})]})])
        self.style.configure("Vertical.TScrollbar", troughcolor=p.field, bordercolor=p.field,
                             lightcolor=p.field, darkcolor=p.field, background=p.field, width=thumb_width,
                             arrowsize=thumb_width)

    def _configure_options(self) -> None:
        """Colours for the classic Tk widgets ttk does not cover (the combobox list, menus, text)."""
        p = self.palette
        root = self.root
        for pattern, value in (("*TCombobox*Listbox.background", p.field),
                               ("*TCombobox*Listbox.foreground", p.text),
                               ("*TCombobox*Listbox.selectBackground", p.accent_soft),
                               ("*TCombobox*Listbox.selectForeground", p.text),
                               ("*TCombobox*Listbox.borderWidth", self.px(5)),
                               ("*TCombobox*Listbox.relief", "flat"),
                               ("*TCombobox*Listbox.highlightThickness", 0),
                               ("*TCombobox*Listbox.activeStyle", "none"),
                               ("*TCombobox*Listbox.font", self.fonts["body"]),
                               ("*Menu.background", p.field), ("*Menu.foreground", p.text),
                               ("*Menu.activeBackground", p.accent_soft), ("*Menu.activeForeground", p.text),
                               ("*Menu.relief", "flat"), ("*Menu.borderWidth", 0)):
            root.option_add(pattern, value)
