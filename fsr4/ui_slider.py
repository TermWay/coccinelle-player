"""A slider drawn on a canvas like the seek bar: a track filled up to a round knob, with labelled marks under it.

The speed, the volume and the zoom use it. A click or a drag goes anywhere along the track, landing on a mark
within SNAP pixels of it; a click on a mark's label lands on that mark. The wheel is the owner's. Where two labels
would touch, the one of higher rank gives way, and so do all the labels ranked after it, so the scale stays even
(the label of the mark the knob is on always shows): a narrow slider keeps its most useful labels. The knob grows while the pointer is over the slider or drags it; a disabled
slider shows its marks faintly and no knob.
"""
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from typing import Callable, Hashable, Sequence

from .ui_theme import Theme

KNOB_INSET = 8  # 96 DPI pixels between the slider's ends and its track, for the knob
SNAP = 4  # 96 DPI pixels on either side of a mark where a click or a drag lands on it
LABEL_GAP = 4  # 96 DPI pixels at least between two labels
LABEL_REACH = 3  # 96 DPI pixels around a label that still count as a click on it
LABEL_MARGIN = 2  # 96 DPI pixels between a label and the slider's ends (glyphs overhang what the font measures)
SLIDER_HEIGHT = 40  # every slider's (tracks of sliders side by side share a line); the labels' descenders fit
BUTTON1_MASK = 0x100


@dataclass(frozen=True)
class Mark:
    """A tick under the track at `fraction` (0 to 1) with `label` under it, standing for `value` (what picking it
    means to the owner). Labels of lower `rank` are placed first; `emphasis` draws one a little brighter."""
    fraction: float
    label: str
    value: Hashable
    rank: int = 1
    emphasis: bool = False


def place_labels(centres: Sequence[float], widths: Sequence[float], ranks: Sequence[int], order: Sequence[int],
                 gap: float) -> list[int]:
    """The labels to show, as indices: taken in `order`, each one placed unless it would come closer than `gap`
    to one already placed, or a label of a lower rank than its own was left out (label `i` is `widths[i]` wide,
    centred on `centres[i]`, of rank `ranks[i]`)."""
    placed: list[tuple[float, float]] = []
    shown = []
    cut = None  # the lowest rank left out
    for index in order:
        if cut is not None and ranks[index] > cut:
            continue
        left = centres[index] - widths[index] / 2
        right = left + widths[index]
        if all(right + gap <= other_left or left >= other_right + gap for other_left, other_right in placed):
            placed.append((left, right))
            shown.append(index)
        elif cut is None or ranks[index] < cut:
            cut = ranks[index]
    return shown


def nearest_mark(marks: Sequence[Mark], fraction: float, reach: float) -> Mark | None:
    """The mark closest to `fraction` if it lies within `reach` of it (both as fractions of the track)."""
    if not marks:
        return None
    closest = min(marks, key=lambda mark: abs(mark.fraction - fraction))
    return closest if abs(closest.fraction - fraction) <= reach else None


class MarkedSlider(tk.Canvas):
    """See the module docstring. `on_pick(fraction, mark)` on a click or a drag: `mark` is the mark it landed on,
    whose fraction it is, or None between marks. `on_wheel(up)` for the wheel. The owner shows the result with
    `show`."""

    def __init__(self, master: tk.Misc, theme: Theme, width: int, on_pick: Callable[[float, Mark | None], None],
                 on_wheel: Callable[[bool], None], marks: Sequence[Mark] = ()):
        super().__init__(master, width=theme.px(width), height=theme.px(SLIDER_HEIGHT), highlightthickness=0,
                         borderwidth=0, background=theme.palette.background, cursor="hand2")
        self.theme = theme
        self.on_pick = on_pick
        self.on_wheel = on_wheel
        self.marks: list[Mark] = list(marks)
        self.fraction = 0.0
        self.current: Hashable = None
        self.dimmed = False  # the fill and the knob grey (the volume while muted)
        self.enabled = True
        self.active = False  # the pointer over the slider or dragging it: a bigger knob
        self._labels: list[tuple[Mark, float, float]] = []  # (mark, left, right) of the labels drawn
        self._redraw_pending = False
        self.bind("<Configure>", lambda _: self.request_redraw())
        self.bind("<ButtonPress-1>", lambda event: self._pressed(event, on_label=True))
        self.bind("<B1-Motion>", lambda event: self._pressed(event, on_label=False))
        self.bind("<ButtonRelease-1>", lambda event: self._set_active(0 <= event.x < self.winfo_width()
                                                                      and 0 <= event.y < self.winfo_height()))
        self.bind("<Enter>", lambda _: self._set_active(True))
        self.bind("<Leave>", lambda event: self._set_active(bool(event.state & BUTTON1_MASK)))
        self.bind("<MouseWheel>", self._wheel)

    # ---- state from the owner -------------------------------------------------------------------

    def show(self, fraction: float, current: Hashable = None) -> None:
        """The knob at `fraction`; the marks whose value is `current` stand out."""
        self.fraction, self.current = min(1.0, max(0.0, fraction)), current
        self.request_redraw()

    def set_marks(self, marks: Sequence[Mark]) -> None:
        if list(marks) != self.marks:
            self.marks = list(marks)
            self.request_redraw()

    def set_dimmed(self, dimmed: bool) -> None:
        if dimmed != self.dimmed:
            self.dimmed = dimmed
            self.request_redraw()

    def set_enabled(self, enabled: bool) -> None:
        if enabled != self.enabled:
            self.enabled = enabled
            self.configure(cursor="hand2" if enabled else "")
            self.request_redraw()

    # ---- geometry -------------------------------------------------------------------------------

    def _track(self) -> tuple[int, int]:
        """The track's left end and length in pixels."""
        inset = self.theme.px(KNOB_INSET)
        return inset, max(1, self.winfo_width() - 2 * inset)

    def x_of(self, fraction: float) -> float:
        left, length = self._track()
        return left + length * fraction

    def _label_top(self) -> int:
        return self.winfo_height() // 2 + self.theme.px(4)

    # ---- drawing --------------------------------------------------------------------------------

    def request_redraw(self) -> None:
        if not self._redraw_pending:
            self._redraw_pending = True
            self.after_idle(self.redraw)

    def redraw(self) -> None:
        self._redraw_pending = False
        palette, px = self.theme.palette, self.theme.px
        self.delete("all")
        self._labels = []
        width = self.winfo_width()
        if width <= 2 * px(KNOB_INSET):
            return
        middle, thickness = self.winfo_height() // 2, px(4)
        font = self.theme.fonts["small"]
        centres, widths = [], []
        for mark in self.marks:
            label_width = font.measure(mark.label)
            half = label_width / 2 + px(LABEL_MARGIN)
            centres.append(min(width - half, max(half, self.x_of(mark.fraction))))
            widths.append(label_width)
        order = sorted(range(len(self.marks)), key=lambda index: (self.marks[index].value != self.current,
                                                                 self.marks[index].rank, index))
        shown = set(place_labels(centres, widths, [mark.rank for mark in self.marks], order, px(LABEL_GAP)))
        for index, mark in enumerate(self.marks):
            lit = self.enabled and mark.value == self.current
            color = (palette.text if lit else palette.text_secondary if mark.emphasis and self.enabled
                     else palette.text_faint)
            x = self.x_of(mark.fraction)
            self.create_line(x, middle + px(4), x, middle + px(7), fill=color)
            if index in shown:
                self.create_text(centres[index], middle + px(4), text=mark.label, fill=color, font=font, anchor="n")
                self._labels.append((mark, centres[index] - widths[index] / 2, centres[index] + widths[index] / 2))
        left, length = self._track()
        self.create_line(left, middle, left + length, middle, fill=palette.field_hover, width=thickness,
                         capstyle="round")
        if not self.enabled:
            return
        color = palette.text_faint if self.dimmed else palette.accent
        knob = self.x_of(self.fraction)
        self.create_line(left, middle, knob, middle, fill=color, width=thickness, capstyle="round")
        radius = px(7) if self.active else px(5.5)
        self.create_oval(knob - radius, middle - radius, knob + radius, middle + radius, fill=color, outline="")

    # ---- input ----------------------------------------------------------------------------------

    def pick_at(self, x: float, y: float, on_label: bool) -> tuple[float, Mark | None]:
        """Where a press (`on_label`: its label counts) or a drag at (x, y) lands: (fraction, mark or None)."""
        if on_label and y >= self._label_top():
            reach = self.theme.px(LABEL_REACH)
            for mark, left, right in self._labels:
                if left - reach <= x <= right + reach:
                    return mark.fraction, mark
        left, length = self._track()
        fraction = min(1.0, max(0.0, (x - left) / length))
        mark = nearest_mark(self.marks, fraction, self.theme.px(SNAP) / length)
        return (mark.fraction, mark) if mark else (fraction, None)

    def _pressed(self, event: tk.Event, on_label: bool) -> None:
        if not self.enabled:
            return
        self._set_active(True)
        self.on_pick(*self.pick_at(event.x, event.y, on_label))

    def _set_active(self, active: bool) -> None:
        if active != self.active:
            self.active = active
            self.request_redraw()

    def _wheel(self, event: tk.Event) -> None:
        if self.enabled:
            self.on_wheel(event.delta > 0)
