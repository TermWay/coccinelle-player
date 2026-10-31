"""Playback controls under the video: a filmstrip, the seek bar, frame steps, play/pause, time, volume, speed
and full screen.

The filmstrip shows the video as tiles in time order: upscaled tiles at full brightness, parts not upscaled
yet dimmed (from the source). The seek bar marks what is upscaled (lighter track), what has played (accent),
and where the renderer works (a small tick). Clicking or dragging either seeks; hovering shows the frame number
and its time to the millisecond. Holding a frame step button repeats it; the speed slider goes from 1x
backwards to 4x forwards in SPEEDS steps (clicking its value goes back to 1x).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

import cv2
import numpy as np

from .ui_art import icon, png_base64, ppm_bytes
from .ui_theme import Theme

STRIP_HEIGHT = 50
BAR_HEIGHT = 20
TILE_GAP = 2
DIM = 0.35
SPEEDS = (-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0)
REPEAT_DELAY_MS = 400
REPEAT_INTERVAL_MS = 70


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


def nearest(thumbnails: dict[int, np.ndarray], frame: int) -> tuple[int, np.ndarray] | None:
    if not thumbnails:
        return None
    key = min(thumbnails, key=lambda candidate: abs(candidate - frame))
    return key, thumbnails[key]


class Transport(ttk.Frame):
    """See the module docstring. Callbacks: `on_toggle()` play/pause, `on_seek(frame, final)` (final is False
    while dragging), `on_step(frames)`, `on_volume(volume, muted)`, `on_speed(speed)`, `on_fullscreen()`."""

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
        self.volume = tk.DoubleVar(value=0.8)
        self.muted = False
        self.upscaled_thumbnails: dict[int, np.ndarray] = {}
        self.source_thumbnails: dict[int, np.ndarray] = {}
        self._tiles: dict[int, tuple] = {}
        self._hover_x: int | None = None
        self.dragging = False
        self._redraw_pending = False
        self._repeat_job: str | None = None
        self.speed = 1.0
        self.speed_index = tk.DoubleVar(value=SPEEDS.index(1.0))
        self.icons = {name: self._icon(name) for name in ("play", "pause", "volume", "mute", "fullscreen", "windowed",
                                                          "frame_back", "frame_forward")}
        self.columnconfigure(0, weight=1)
        self.strip = tk.Canvas(self, height=px(STRIP_HEIGHT), background=palette.background, highlightthickness=0,
                               borderwidth=0, cursor="hand2")
        self.strip.grid(row=0, column=0, sticky="ew")
        self.bar = tk.Canvas(self, height=px(BAR_HEIGHT), background=palette.background, highlightthickness=0,
                             borderwidth=0, cursor="hand2")
        self.bar.grid(row=1, column=0, sticky="ew")
        controls = ttk.Frame(self)
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(6, weight=1)
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
        self.time_label = ttk.Label(controls, text="", style="Small.TLabel")
        self.time_label.grid(row=0, column=3, padx=(px(6), px(14)))
        self.volume_button = ttk.Button(controls, image=self.icons["volume"], style="Icon.TButton", takefocus=False,
                                        command=self.toggle_mute)
        self.volume_button.grid(row=0, column=4)
        self.volume_scale = ttk.Scale(controls, from_=0.0, to=1.0, variable=self.volume, length=px(84),
                                      takefocus=False, command=lambda _: self._volume_changed())
        self.volume_scale.grid(row=0, column=5, padx=(px(2), px(12)))
        self.state_label = ttk.Label(controls, text="", style="Faint.TLabel")
        self.state_label.grid(row=0, column=6, sticky="w")
        self.speed_scale = ttk.Scale(controls, from_=0, to=len(SPEEDS) - 1, variable=self.speed_index,
                                     length=px(96), takefocus=False, command=lambda _: self._speed_moved())
        self.speed_scale.grid(row=0, column=7, padx=(px(8), px(4)))
        self.speed_label = ttk.Label(controls, text=format_speed(1.0), style="Small.TLabel", width=6, anchor="w",
                                     cursor="hand2")
        self.speed_label.grid(row=0, column=8, padx=(0, px(8)))
        self.speed_label.bind("<ButtonRelease-1>", lambda _: self.set_speed(1.0, notify=True))
        self.fullscreen_button = ttk.Button(controls, image=self.icons["fullscreen"], style="Icon.TButton",
                                            takefocus=False, command=self.on_fullscreen)
        self.fullscreen_button.grid(row=0, column=9)
        for canvas in (self.strip, self.bar):
            canvas.bind("<Configure>", lambda _: self.request_redraw())
            canvas.bind("<ButtonPress-1>", self._press)
            canvas.bind("<B1-Motion>", self._drag)
            canvas.bind("<ButtonRelease-1>", self._release)
            canvas.bind("<Motion>", self._hover)
            canvas.bind("<Leave>", self._leave)

    def _icon(self, name: str) -> tk.PhotoImage:
        image = tk.PhotoImage(master=self, data=png_base64(icon(name, self.theme.px(20), self.theme.palette.text)))
        return image

    # ---- state from the window ------------------------------------------------------------------

    def set_media(self, total: int, rate: float) -> None:
        self.total, self.rate = max(1, total), rate
        self.position = 0
        self.set_playing(False, False)
        self.set_speed(1.0)
        self.ready_ranges, self.render_position = [], None
        self.upscaled_thumbnails, self.source_thumbnails = {}, {}
        self._tiles.clear()
        self._update_time()
        self.request_redraw()

    def set_total(self, total: int) -> None:
        self.total = max(1, total)
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
        """Show `speed` (one of SPEEDS) on the slider; `notify` tells the window, as a slider move does."""
        self.speed = speed
        self.speed_index.set(SPEEDS.index(speed))
        self.speed_label.configure(text=format_speed(speed))
        if notify:
            self.on_speed(speed)

    def change_speed(self, steps: int) -> None:
        """The next speed up (`steps` > 0) or down along SPEEDS."""
        index = min(len(SPEEDS) - 1, max(0, SPEEDS.index(self.speed) + steps))
        if SPEEDS[index] != self.speed:
            self.set_speed(SPEEDS[index], notify=True)

    def _speed_moved(self) -> None:
        speed = SPEEDS[int(round(float(self.speed_index.get())))]
        self.speed_index.set(SPEEDS.index(speed))  # the knob snaps to the step
        if speed != self.speed:
            self.set_speed(speed, notify=True)

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
        self.volume_button.configure(image=self.icons["mute" if self.muted else "volume"])
        self.on_volume(float(self.volume.get()), self.muted)

    def change_volume(self, step: float) -> None:
        self.volume.set(min(1.0, max(0.0, float(self.volume.get()) + step)))
        self._volume_changed()

    def _volume_changed(self) -> None:
        if self.muted and self.volume.get() > 0:
            self.muted = False
            self.volume_button.configure(image=self.icons["volume"])
        self.on_volume(float(self.volume.get()), self.muted)

    def _update_time(self) -> None:
        rate = self.rate or 30.0
        self.time_label.configure(text=f"{format_precise(self.position / rate)} / {format_time(self.total / rate)}"
                                       f" · frame {self.position}")

    # ---- drawing --------------------------------------------------------------------------------

    def request_redraw(self) -> None:
        if not self._redraw_pending:
            self._redraw_pending = True
            self.after_idle(self._redraw)

    def _frame_at(self, x: float, width: int) -> int:
        inset = self.theme.px(8)
        fraction = (x - inset) / max(1, width - 2 * inset)
        return int(min(self.total - 1, max(0, round(fraction * (self.total - 1)))))

    def _x_of(self, frame: float, width: int) -> float:
        inset = self.theme.px(8)
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

