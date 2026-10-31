"""Hover tooltips in the window's dark style: a short delay, then a small borderless window near the pointer."""
from __future__ import annotations

import tkinter as tk
from typing import Callable

from .ui_theme import Theme

DELAY_MS = 450
WRAP = 320


class Tooltip:
    """Explains `widget` when the pointer rests on it. `text` may be a function, read each time it shows."""

    def __init__(self, widget: tk.Misc, theme: Theme, text: str | Callable[[], str]):
        self.widget = widget
        self.theme = theme
        self.text = text
        self._window: tk.Toplevel | None = None
        self._pending: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    def _schedule(self, _event: tk.Event) -> None:
        self._cancel()
        self._pending = self.widget.after(DELAY_MS, self._show)

    def _cancel(self) -> None:
        if self._pending is not None:
            self.widget.after_cancel(self._pending)
            self._pending = None

    def _show(self) -> None:
        self._pending = None
        text = self.text() if callable(self.text) else self.text
        if not text or self._window is not None:
            return
        palette, px = self.theme.palette, self.theme.px
        window = tk.Toplevel(self.widget)
        window.wm_overrideredirect(True)
        window.wm_attributes("-topmost", True)
        window.configure(background=palette.border_hover)
        label = tk.Label(window, text=text, justify="left", wraplength=px(WRAP), background=palette.field,
                         foreground=palette.text, font=self.theme.fonts["small"], padx=px(10), pady=px(7))
        label.pack(padx=1, pady=1)
        window.update_idletasks()
        width, height = window.winfo_reqwidth(), window.winfo_reqheight()
        x = self.widget.winfo_pointerx() + px(12)
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + px(6)
        screen_width, screen_height = window.winfo_screenwidth(), window.winfo_screenheight()
        x = max(0, min(x, screen_width - width - px(4)))
        if y + height > screen_height - px(4):
            y = self.widget.winfo_rooty() - height - px(6)
        window.wm_geometry(f"+{x}+{y}")
        self._window = window

    def _hide(self, _event: tk.Event | None = None) -> None:
        self._cancel()
        if self._window is not None:
            self._window.destroy()
            self._window = None
