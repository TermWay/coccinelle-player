"""Playback controls under the video: a filmstrip, the seek bar, frame steps, play/pause, time, volume, speed
and full screen. The time keeps a fixed width, so nothing after it moves as it plays.

The filmstrip shows the video as tiles in time order: upscaled tiles at full brightness, parts not upscaled
yet dimmed (from the source). The seek bar marks what is upscaled (lighter track), what has played (accent),
and where the renderer works (a small tick). Clicking or dragging either seeks; hovering shows the frame number
and its time to the millisecond. Holding a frame step button repeats it. The volume and the speed are marked
sliders (`ui_slider`), both log scales with 0 at their left end. The volume is the sound's amplitude in percent,
laid out in decibels from QUIETEST_VOLUME (-40 dB) to 100%, so equal distances sound like equal steps; its
speaker shows the level. The speed slider has SPEED_TICKS under it; the toggle beside it plays backwards at the
slider's speed and clicking the value goes back to 1x. In a narrow window the row gives up, in this order, the speed slider's
width, the volume slider's and the volume's percentage: the grid's column weights do that.
"""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import Callable

import cv2
import numpy as np

from .ui_art import icon, png_base64, ppm_bytes
from .ui_slider import KNOB_INSET, Mark, MarkedSlider
from .ui_theme import Theme

STRIP_HEIGHT = 50
BAR_HEIGHT = 20
TILE_GAP = 2
DIM = 0.35
# (speed, label, rank): a narrow slider keeps the labels of rank 0.
SPEED_TICKS = ((0.0, "0", 0), (0.05, ".05", 1), (0.1, "0.1", 1), (0.25, "¼", 0), (0.5, "½", 1), (1.0, "1×", 0),
               (2.0, "2", 1), (5.0, "5", 0))
SPEED_STOPS = tuple(stop for stop, _, _ in SPEED_TICKS)
SLOWEST_SPEED, FASTEST_SPEED = SPEED_STOPS[1], SPEED_STOPS[-1]
STOPPED_SHARE = 0.16  # of the speed slider, from 0 to the slowest speed: no speed lies in between
SPEED_WIDTH = 176
SPEED_MIN_WIDTH = 84
REPEAT_DELAY_MS = 400
REPEAT_INTERVAL_MS = 70
# (amplitude, label, rank), about 10 dB apart.
VOLUME_TICKS = ((0.0, "0", 0), (0.01, "1", 2), (0.03, "3", 3), (0.1, "10", 1), (0.3, "30", 2), (1.0, "100", 0))
QUIETEST_VOLUME = 0.01  # -40 dB, the volume slider's lowest step above 0
SILENT_SHARE = 0.14  # of the volume slider, from 0 to the quietest volume: no volume lies in between
VOLUME_LOW = 0.25  # below this the speaker shows one wave instead of two
VOLUME_WIDTH = 150
VOLUME_MIN_WIDTH = 48
VOLUME_WHEEL_DECIBELS = 2.0
VOLUME_KEY_DECIBELS = 4.0  # up and down
# The row's columns: the state takes all the spare room, and the others give it up in this order.
STATE_WEIGHT, SPEED_WEIGHT, VOLUME_WEIGHT, PERCENT_WEIGHT = 100000, 100, 10, 1


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def format_precise(seconds: float) -> str:
    """A time to the millisecond: 1:05.250, or 1:02:05.250."""
    milliseconds = int(round(max(0.0, seconds) * 1000))
    seconds, milliseconds = divmod(milliseconds, 1000)
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return f"{minutes}:{seconds:02d}.{milliseconds:03d}"


def format_speed(speed: float) -> str:
    """1×, 0.25×, −2× (a real minus sign)."""
    return ("\u2212" if speed < 0 else "") + f"{abs(speed):g}×"


def speed_fraction(speed: float) -> float:
    """Where `speed` (0, or SLOWEST_SPEED to FASTEST_SPEED) lies along the slider, from 0 to 1."""
    if speed <= 0:
        return 0.0
    along_the_log = math.log(speed / SLOWEST_SPEED) / math.log(FASTEST_SPEED / SLOWEST_SPEED)
    return STOPPED_SHARE + (1 - STOPPED_SHARE) * along_the_log


def speed_at(fraction: float) -> float:
    """The speed at `fraction` of the slider, rounded to what the value shows; in the gap between 0 and
    SLOWEST_SPEED, whichever of the two is nearer."""
    fraction = min(1.0, max(0.0, fraction))
    if fraction < STOPPED_SHARE:
        return 0.0 if fraction < STOPPED_SHARE / 2 else SLOWEST_SPEED
    speed = SLOWEST_SPEED * (FASTEST_SPEED / SLOWEST_SPEED) ** ((fraction - STOPPED_SHARE) / (1 - STOPPED_SHARE))
    return round(speed, 2 if speed < 1 else 1)


SPEED_MARKS = tuple(Mark(speed_fraction(speed), label, speed, rank, emphasis=speed == 1.0)
                    for speed, label, rank in SPEED_TICKS)


def volume_fraction(volume: float) -> float:
    """Where `volume` (the amplitude, 0 to 1) lies along the volume slider: 0 at its left end, then decibels from
    QUIETEST_VOLUME to 1."""
    if volume <= 0:
        return 0.0
    along_the_log = math.log(volume / QUIETEST_VOLUME) / math.log(1 / QUIETEST_VOLUME)
    return SILENT_SHARE + (1 - SILENT_SHARE) * min(1.0, max(0.0, along_the_log))


def two_figures(volume: float) -> float:
    """`volume` to two significant figures: 1.3%, 13%, 100% (what the percentage shows)."""
    return float(f"{volume:.2g}")


def volume_at(fraction: float) -> float:
    """The volume at `fraction` of the slider, to two significant figures; in the gap between 0 and
    QUIETEST_VOLUME, whichever of the two is nearer."""
    fraction = min(1.0, max(0.0, fraction))
    if fraction < SILENT_SHARE:
        return 0.0 if fraction < SILENT_SHARE / 2 else QUIETEST_VOLUME
    return two_figures(QUIETEST_VOLUME ** (1 - (fraction - SILENT_SHARE) / (1 - SILENT_SHARE)))


def step_volume(volume: float, decibels: float) -> float:
    """`volume` raised (or lowered) by `decibels`: up from 0 to QUIETEST_VOLUME, down below it to 0, at most 1."""
    if volume <= 0:
        return QUIETEST_VOLUME if decibels > 0 else 0.0
    target = volume * 10 ** (decibels / 20)
    return 0.0 if target < QUIETEST_VOLUME * 0.99 else min(1.0, two_figures(target))


def format_volume(volume: float) -> str:
    """80%, 1.3%, 0%."""
    percent = volume * 100
    return f"{percent:.0f}%" if percent >= 10 or percent == 0 else f"{percent:.2g}%"


VOLUME_MARKS = tuple(Mark(volume_fraction(volume), label, volume, rank) for volume, label, rank in VOLUME_TICKS)


def volume_icon(volume: float, muted: bool) -> str:
    """The speaker's icon: crossed when muted, then no wave at 0, one below VOLUME_LOW, two above."""
    if muted:
        return "mute"
    if volume <= 0:
        return "volume_none"
    return "volume_low" if volume < VOLUME_LOW else "volume"


def next_stop(speed: float, faster: bool) -> float:
    """The tick after `speed` (0 or more, on a tick or between two) going up or down; `speed` past the last one."""
    if faster:
        return next((stop for stop in SPEED_STOPS if stop > speed), speed)
    return next((stop for stop in reversed(SPEED_STOPS) if stop < speed), speed)


def nearest(thumbnails: dict[int, np.ndarray], frame: int) -> tuple[int, np.ndarray] | None:
    if not thumbnails:
        return None
    key = min(thumbnails, key=lambda candidate: abs(candidate - frame))
    return key, thumbnails[key]


class Transport(ttk.Frame):
    """See the module docstring. Callbacks: `on_toggle()` play/pause, `on_seek(frame, final)` (final is False
    while dragging), `on_step(frames)`, `on_volume(volume, muted)` (the amplitude), `on_speed(speed)`,
    `on_fullscreen()`."""

    def __init__(self, master: tk.Misc, theme: Theme, on_toggle: Callable[[], None],
                 on_seek: Callable[[int, bool], None], on_step: Callable[[int], None],
                 on_volume: Callable[[float, bool], None], on_speed: Callable[[float], None],
                 on_fullscreen: Callable[[], None]):
        super().__init__(master)
        self.theme = theme
        self.on_toggle = on_toggle
        self.on_seek = on_seek
        self.on_step = on_step
        self.on_volume = on_volume
        self.on_speed = on_speed
        self.on_fullscreen = on_fullscreen
        palette, px = theme.palette, theme.px
        self.total = 1
        self.rate = 30.0
        self.position = 0
        self.ready_ranges: list[tuple[int, int]] = []
        self.render_position: int | None = None
        self.playing = False
        self.buffering = False
        self.fullscreen = False
        self.volume = 0.8  # the sound's amplitude
        self.muted = False
        self.upscaled_thumbnails: dict[int, np.ndarray] = {}
        self.source_thumbnails: dict[int, np.ndarray] = {}
        self._tiles: dict[int, tuple] = {}
        self._hover_x: int | None = None
        self.dragging = False
        self._redraw_pending = False
        self._repeat_job: str | None = None
        self.speed = 1.0  # negative backwards, 0 stopped
        self.backwards = False  # the reverse toggle: at 0, the direction the next speed takes
        self.icons = {name: self._icon(name) for name in ("play", "pause", "volume", "volume_low", "volume_none",
                                                          "mute", "fullscreen", "windowed", "frame_back",
                                                          "frame_forward", "reverse")}
        self.icons["reverse_on"] = self._icon("reverse", palette.accent)
        self.columnconfigure(0, weight=1)
        self.strip = tk.Canvas(self, height=px(STRIP_HEIGHT), background=palette.background, highlightthickness=0,
                               borderwidth=0, cursor="hand2")
        self.strip.grid(row=0, column=0, sticky="ew")
        self.bar = tk.Canvas(self, height=px(BAR_HEIGHT), background=palette.background, highlightthickness=0,
                             borderwidth=0, cursor="hand2")
        self.bar.grid(row=1, column=0, sticky="ew")
        controls = self.controls = ttk.Frame(self)
        controls.grid(row=2, column=0, sticky="ew")
        self.frame_back_button = ttk.Button(controls, image=self.icons["frame_back"], style="Icon.TButton",
                                            takefocus=False)
        self.frame_back_button.grid(row=0, column=0)
        self.play_button = ttk.Button(controls, image=self.icons["play"], style="Icon.TButton", takefocus=False,
                                      command=self.on_toggle)
        self.play_button.grid(row=0, column=1)
        self.frame_forward_button = ttk.Button(controls, image=self.icons["frame_forward"], style="Icon.TButton",
                                               takefocus=False)
        self.frame_forward_button.grid(row=0, column=2)
        for button, frames in ((self.frame_back_button, -1), (self.frame_forward_button, 1)):
            button.bind("<ButtonPress-1>", lambda _, frames=frames: self._start_repeat(frames), add="+")
            button.bind("<ButtonRelease-1>", lambda _: self._stop_repeat(), add="+")
            button.bind("<Leave>", lambda _: self._stop_repeat(), add="+")
        # The time sits in a column as wide as its longest text for this video (`_fit_time_column`), in a
        # monospace font: what follows stays in place and the digits do not shift as they change.
        self.time_label = ttk.Label(controls, text="", style="Time.TLabel")
        self.time_label.grid(row=0, column=3, sticky="w", padx=(px(6), px(14)))
        self.volume_button = ttk.Button(controls, image=self.icons["volume"], style="Icon.TButton", takefocus=False,
                                        command=self.toggle_mute)
        self.volume_button.grid(row=0, column=4)
        self.volume_button.bind("<MouseWheel>", lambda event: self._volume_wheel(event.delta > 0))
        self.volume_bar = MarkedSlider(controls, theme, VOLUME_WIDTH, on_pick=self._volume_picked,
                                       on_wheel=self._volume_wheel, marks=VOLUME_MARKS)
        self.volume_bar.grid(row=0, column=5, sticky="ew", padx=(px(2), px(6)))
        controls.columnconfigure(5, weight=VOLUME_WEIGHT, minsize=px(VOLUME_MIN_WIDTH) + px(8))
        self.volume_label = ttk.Label(controls, text="", style="Small.TLabel", width=4)
        self.volume_label.grid(row=0, column=6, sticky="w", padx=(0, px(12)))
        controls.columnconfigure(6, weight=PERCENT_WEIGHT)
        self.state_label = ttk.Label(controls, text="", style="Faint.TLabel")
        self.state_label.grid(row=0, column=7, sticky="w")
        controls.columnconfigure(7, weight=STATE_WEIGHT)
        self.reverse_button = ttk.Button(controls, image=self.icons["reverse"], style="Icon.TButton", takefocus=False,
                                         command=self.toggle_reverse)
        self.reverse_button.grid(row=0, column=8, padx=(px(8), 0))
        self.speed_bar = MarkedSlider(controls, theme, SPEED_WIDTH, on_pick=self._speed_picked,
                                      on_wheel=self.change_speed, marks=SPEED_MARKS)
        self.speed_bar.grid(row=0, column=9, sticky="ew", padx=(px(2), px(4)))
        controls.columnconfigure(9, weight=SPEED_WEIGHT, minsize=px(SPEED_MIN_WIDTH) + px(6))
        self.speed_label = ttk.Label(controls, text=format_speed(1.0), style="Small.TLabel", width=6, anchor="w",
                                     cursor="hand2")
        self.speed_label.grid(row=0, column=10, padx=(0, px(8)))
        self.speed_label.bind("<ButtonRelease-1>", lambda _: self.set_speed(1.0, notify=True))
        self.fullscreen_button = ttk.Button(controls, image=self.icons["fullscreen"], style="Icon.TButton",
                                            takefocus=False, command=self.on_fullscreen)
        self.fullscreen_button.grid(row=0, column=11)
        self._show_volume()
        self._show_speed()
        for canvas in (self.strip, self.bar):
            canvas.bind("<Configure>", lambda _: self.request_redraw())
            canvas.bind("<ButtonPress-1>", self._press)
            canvas.bind("<B1-Motion>", self._drag)
            canvas.bind("<ButtonRelease-1>", self._release)
            canvas.bind("<Motion>", self._hover)
            canvas.bind("<Leave>", self._leave)

    def _icon(self, name: str, color: str | None = None) -> tk.PhotoImage:
        size, color = self.theme.px(20), color or self.theme.palette.text
        return tk.PhotoImage(master=self, data=png_base64(icon(name, size, color)))

    # ---- state from the window ------------------------------------------------------------------

    def set_media(self, total: int, rate: float) -> None:
        self.total, self.rate = max(1, total), rate
        self.position = 0
        self.set_playing(False, False)
        self.set_speed(1.0)
        self.ready_ranges, self.render_position = [], None
        self.upscaled_thumbnails, self.source_thumbnails = {}, {}
        self._tiles.clear()
        self._fit_time_column()
        self._update_time()
        self.request_redraw()

    def set_total(self, total: int) -> None:
        self.total = max(1, total)
        self._fit_time_column()
        self._update_time()
        self.request_redraw()

    def set_position(self, frame: int) -> None:
        if not self.dragging and frame != self.position:
            self.position = frame
            self._update_time()
            self.request_redraw()

    def set_progress(self, ready_ranges: list[tuple[int, int]], render_position: int | None) -> None:
        if ready_ranges != self.ready_ranges or render_position != self.render_position:
            self.ready_ranges, self.render_position = ready_ranges, render_position
            self.request_redraw()

    def set_thumbnails(self, upscaled: dict[int, np.ndarray] | None = None,
                       source: dict[int, np.ndarray] | None = None) -> None:
        if upscaled is not None:
            self.upscaled_thumbnails = upscaled
        if source is not None:
            self.source_thumbnails = source
        self.request_redraw()

    def set_playing(self, playing: bool, buffering: bool) -> None:
        self.playing, self.buffering = playing, buffering
        self.play_button.configure(image=self.icons["pause" if playing else "play"])
        self.state_label.configure(text="Buffering · upscaling ahead…" if playing and buffering else "")

    def set_fullscreen(self, fullscreen: bool) -> None:
        self.fullscreen = fullscreen
        self.fullscreen_button.configure(image=self.icons["windowed" if fullscreen else "fullscreen"])

    def set_speed(self, speed: float, notify: bool = False) -> None:
        """Show `speed` (negative: backwards, 0: stopped) on the slider and the reverse toggle; `notify` tells the
        window, as a slider move does."""
        self.speed = speed
        if speed != 0:
            self.backwards = speed < 0
        self.reverse_button.configure(image=self.icons["reverse_on" if self.backwards else "reverse"])
        self.speed_label.configure(text=format_speed(speed))
        self._show_speed()
        if notify:
            self.on_speed(speed)

    def set_pace(self, pace: float) -> None:
        """The slider's value `pace` (0 or more), in the reverse toggle's direction, told to the window."""
        if pace != abs(self.speed):
            self.set_speed(-pace if self.backwards else pace, notify=True)

    def change_speed(self, faster: bool) -> None:
        """The slider to the next tick up or down."""
        self.set_pace(next_stop(abs(self.speed), faster))

    def toggle_reverse(self) -> None:
        """Forwards or backwards at the slider's speed; stopped, the next speed goes that way."""
        self.backwards = not self.backwards
        self.set_speed(-self.speed, notify=self.speed != 0)

    def _show_speed(self) -> None:
        pace = abs(self.speed)
        self.speed_bar.show(speed_fraction(pace), pace)

    def _speed_picked(self, fraction: float, mark: Mark | None) -> None:
        self.set_pace(mark.value if mark else speed_at(fraction))

    def _start_repeat(self, frames: int) -> None:
        """A frame step now, then repeated while the button is held."""
        self._stop_repeat()
        self.on_step(frames)
        self._repeat_job = self.after(REPEAT_DELAY_MS, lambda: self._repeat(frames))

    def _repeat(self, frames: int) -> None:
        self.on_step(frames)
        self._repeat_job = self.after(REPEAT_INTERVAL_MS, lambda: self._repeat(frames))

    def _stop_repeat(self) -> None:
        if self._repeat_job is not None:
            self.after_cancel(self._repeat_job)
            self._repeat_job = None

    def toggle_mute(self) -> None:
        self.muted = not self.muted
        self._show_volume()
        self.on_volume(self.volume, self.muted)

    def change_volume(self, decibels: float) -> None:
        """Up or down by `decibels` (`step_volume`)."""
        self.set_volume(step_volume(self.volume, decibels))

    def set_volume(self, volume: float) -> None:
        """The volume (the amplitude) from 0 to 1; moving it while muted turns the sound back on, as players do."""
        self.volume = min(1.0, max(0.0, volume))
        if self.muted and self.volume > 0:
            self.muted = False
        self._show_volume()
        self.on_volume(self.volume, self.muted)

    def _show_volume(self) -> None:
        self.volume_button.configure(image=self.icons[volume_icon(self.volume, self.muted)])
        self.volume_label.configure(text=format_volume(self.volume),
                                    style="Faint.TLabel" if self.muted else "Small.TLabel")
        self.volume_bar.set_dimmed(self.muted)
        self.volume_bar.show(volume_fraction(self.volume), self.volume)

    def _volume_picked(self, fraction: float, mark: Mark | None) -> None:
        self.set_volume(mark.value if mark else volume_at(fraction))

    def _volume_wheel(self, up: bool) -> None:
        """The wheel over the speaker or the slider: VOLUME_WHEEL_DECIBELS up or down."""
        self.change_volume(VOLUME_WHEEL_DECIBELS if up else -VOLUME_WHEEL_DECIBELS)

    def _time_text(self, frame: int) -> str:
        rate = self.rate or 30.0
        return f"{format_precise(frame / rate)} / {format_time(self.total / rate)} · frame {frame}"

    def _fit_time_column(self) -> None:
        """The time's column as wide as its text at the last frame, the longest this video shows (the font is
        monospace), so the controls after it stay in place while it plays."""
        self.time_label.configure(text=self._time_text(self.total - 1))
        width = self.time_label.winfo_reqwidth() + self.theme.px(6) + self.theme.px(14)
        self.controls.columnconfigure(3, minsize=width)

    def _update_time(self) -> None:
        self.time_label.configure(text=self._time_text(self.position))

    # ---- drawing --------------------------------------------------------------------------------

    def request_redraw(self) -> None:
        if not self._redraw_pending:
            self._redraw_pending = True
            self.after_idle(self._redraw)

    def _frame_at(self, x: float, width: int) -> int:
        inset = self.theme.px(KNOB_INSET)
        fraction = (x - inset) / max(1, width - 2 * inset)
        return int(min(self.total - 1, max(0, round(fraction * (self.total - 1)))))

    def _x_of(self, frame: float, width: int) -> float:
        inset = self.theme.px(KNOB_INSET)
        return inset + (width - 2 * inset) * frame / max(1, self.total - 1)

    def _redraw(self) -> None:
        self._redraw_pending = False
        self._draw_strip()
        self._draw_bar()

    def _draw_bar(self) -> None:
        palette, px = self.theme.palette, self.theme.px
        canvas = self.bar
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 20:
            return
        middle = height // 2
        thickness = px(4)
        left, right = self._x_of(0, width), self._x_of(self.total - 1, width)
        canvas.create_line(left, middle, right, middle, fill=palette.field_hover, width=thickness, capstyle="round")
        for start, end in self.ready_ranges:
            canvas.create_line(self._x_of(start, width), middle, self._x_of(max(start, end - 1), width), middle,
                               fill="#5b5f68", width=thickness, capstyle="round")
        position = self._x_of(self.position, width)
        canvas.create_line(left, middle, position, middle, fill=palette.accent, width=thickness, capstyle="round")
        if self.render_position is not None:
            x = self._x_of(min(self.render_position, self.total - 1), width)
            canvas.create_line(x, middle - px(5), x, middle + px(5), fill=palette.accent_text, width=max(1, px(2)))
        radius = px(7) if self._hover_x is not None or self.dragging else px(5.5)
        canvas.create_oval(position - radius, middle - radius, position + radius, middle + radius,
                           fill=palette.accent, outline="")

    def _draw_strip(self) -> None:
        palette, px = self.theme.palette, self.theme.px
        canvas = self.strip
        width, height = canvas.winfo_width(), canvas.winfo_height()
        canvas.delete("overlay")
        if width < 20:
            return
        inset = px(8)
        tile_height = height - px(6)
        aspect = 16 / 9
        sample = next(iter(self.source_thumbnails.values()), None)
        if sample is None:
            sample = next(iter(self.upscaled_thumbnails.values()), None)
        if sample is not None:
            aspect = sample.shape[1] / sample.shape[0]
        count = max(1, int((width - 2 * inset + TILE_GAP) // (tile_height * aspect + TILE_GAP)))
        tile_width = (width - 2 * inset - (count - 1) * TILE_GAP) / count
        layout = (width, height, count)
        if any(key[0] != layout for key in self._tiles.values()):
            canvas.delete("tile")
            self._tiles.clear()
        for index in range(count):
            first = int(index * self.total / count)
            ready = self._is_ready(first)
            found = nearest(self.upscaled_thumbnails if ready else self.source_thumbnails, first)
            if found is None and ready:
                found = nearest(self.source_thumbnails, first)
            key = (layout, ready, found[0] if found else None)
            left = inset + index * (tile_width + TILE_GAP)
            if self._tiles.get(index, (None,))[0:3] != key:
                canvas.delete(f"tile{index}")
                size = (max(1, int(tile_width)), tile_height)
                if found is None:
                    item = canvas.create_rectangle(left, px(3), left + size[0], px(3) + tile_height,
                                                   fill=palette.field, outline="", tags=("tile", f"tile{index}"))
                    self._tiles[index] = key + (None, item)
                else:
                    picture = cv2.resize(found[1], size, interpolation=cv2.INTER_AREA)
                    if not ready:
                        picture = (picture.astype(np.float32) * DIM).astype(np.uint8)
                    photo = tk.PhotoImage(master=canvas, data=ppm_bytes(picture), format="PPM")
                    item = canvas.create_image(left, px(3), image=photo, anchor="nw", tags=("tile", f"tile{index}"))
                    self._tiles[index] = key + (photo, item)
        position = self._x_of(self.position, width)
        canvas.create_line(position, 0, position, height, fill=palette.accent, width=max(1, px(2)), tags="overlay")
        if self._hover_x is not None:
            frame = self._frame_at(self._hover_x, width)
            text = f"frame {frame} · {format_precise(frame / (self.rate or 30.0))}"
            x = min(max(self._hover_x, px(70)), width - px(70))
            label = canvas.create_text(x, height - px(12), text=text, fill=palette.text, font=self.theme.fonts["small"],
                                       tags="overlay")
            x0, y0, x1, y1 = canvas.bbox(label)
            back = canvas.create_rectangle(x0 - px(6), y0 - px(2), x1 + px(6), y1 + px(2), fill="#000000",
                                           outline=palette.border_hover, tags="overlay")
            canvas.tag_raise(label, back)
            canvas.create_line(self._hover_x, 0, self._hover_x, height, fill=palette.text_secondary, tags="overlay")
            canvas.tag_raise(back)
            canvas.tag_raise(label)

    def _is_ready(self, frame: int) -> bool:
        return any(start <= frame < end for start, end in self.ready_ranges)

    # ---- input ----------------------------------------------------------------------------------

    def _seek_to(self, x: int, canvas: tk.Canvas, final: bool) -> None:
        frame = self._frame_at(x, canvas.winfo_width())
        self.position = frame
        self._update_time()
        self.request_redraw()
        self.on_seek(frame, final)

    def _press(self, event: tk.Event) -> None:
        self.dragging = True
        self._seek_to(event.x, event.widget, False)

    def _drag(self, event: tk.Event) -> None:
        self._hover_x = event.x
        self._seek_to(event.x, event.widget, False)

    def _release(self, event: tk.Event) -> None:
        self.dragging = False
        self._seek_to(event.x, event.widget, True)

    def _hover(self, event: tk.Event) -> None:
        self._hover_x = event.x
        self.request_redraw()

    def _leave(self, _event: tk.Event) -> None:
        self._hover_x = None
        self.request_redraw()

