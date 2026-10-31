"""Plays the upscaled video in the video view in real time, with sound, while it is being upscaled.

Paused, the view shows a still: the full-resolution source and upscaled frames, so zoom, pan and the divider
respond at once. Playing, decoders stream the source (for the filters) and the finished pieces of each upscaled
layer shown (already cropped to the view by ffmpeg), and a clock decides which frame is due; late frames are
dropped. Layers are versions of the upscale (settings or presets). When the next frame of a layer being
upscaled is not ready yet the player buffers like a streaming site: sound and clock stop, a spinner turns, and
playback resumes once enough is ready in every such layer. A layer that is not being upscaled (a preset shown
during a whole-file upscale) shows what it has and never holds playback up.

Playback runs at any `speed`: from 0.25x to 2x the sound plays, its pitch kept (ffmpeg's atempo), the sound
card is the clock and late frames are dropped to stay with it. Slower, the sound is off and the system clock
runs. Faster than 2x and backwards (negative speeds) the sound is off too and every frame is shown in order:
when the decoders cannot keep up, playback slows down rather than skip to frames nobody decoded. Backwards,
each decoder works in short chunks that it hands over last frame first, and the window has the renderer work
behind the playhead.

Everything runs on the player's own thread; the window sends commands and receives events through
`on_event(kind, *values)`: "position" (frame shown), "state" (playing, buffering), "ended", "notice"
(a message for the user, e.g. no sound device), "error" (an exception; playback stops).
"""
from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from . import log
from .audio import AudioPlayer
from .media import FrameReader, KeyframeIndex, MediaInfo, read_frame
from .timeline import Timeline

STREAM_BUFFERS = 6
PREFETCH_FRAMES = 30
POSITION_INTERVAL = 0.1
SPINNER_INTERVAL = 0.05
BUFFER_MIN_SECONDS = 2.0
BUFFER_MAX_SECONDS = 10.0
BUFFER_HORIZON_SECONDS = 15.0
AUDIO_STALL_SECONDS = 0.25
RESTART_SETTLE_SECONDS = 0.15
FIRST_FRAME_TIMEOUT = 10.0
LENIENT_FIRST_FRAME_TIMEOUT = 1.0
SOUND_SPEEDS = (0.25, 2.0)
REVERSE_CHUNK_BYTES = 64 * 1024 ** 2
REVERSE_CHUNK_FRAMES = (4, 30)
PIECE_CONTAINER = "mov,mp4,m4a,3gp,3g2,mj2"

Segment = tuple[Callable[[int], FrameReader], int, int]
"""(open(first) -> a reader from frame `first`, start, end): a stretch of frames [start, end) one file holds."""
SEGMENT_POLL_SECONDS = 0.02


def merge_seeks(first: tuple, second: tuple) -> tuple:
    """One command for two consecutive "seek" (absolute) or "step" (relative) commands: a drag on the seek bar
    sends dozens, and only where it ends matters."""
    if second[0] == "seek":
        return second
    if first[0] == "seek":
        return ("seek", first[1] + second[1])
    return ("step", first[1] + second[1])


def reverse_chunk(width: int, height: int) -> int:
    """Frames a reverse stream decodes per seek: as many as REVERSE_CHUNK_BYTES of BGRA frames, within bounds."""
    low, high = REVERSE_CHUNK_FRAMES
    return max(low, min(high, REVERSE_CHUNK_BYTES // max(1, width * height * 4)))


class _Stream:
    """Frames decoded on a thread into a pool of `buffers` BGRA buffers and taken in playing order: increasing
    frame numbers (`direction` 1) or decreasing (-1). `segments(frame)` returns the Segment holding `frame`, or
    None while nothing can be decoded there yet. A `strict` stream treats a segment that ends early as an error
    (a damaged piece); otherwise it is the end of the video (the source, whose frame count may be an estimate).
    With `skip`, gaps between segments are jumped instead of waited for."""

    direction = 1

    def __init__(self, segments: Callable[[int], Segment | None], start: int, width: int, height: int, name: str,
                 strict: bool, buffers: int, skip: Callable[[int], int | None] | None = None):
        self.name = name
        self.strict = strict
        self.error: Exception | None = None
        self._segments = segments
        self._skip = skip
        self._pool: queue.Queue = queue.Queue()
        for _ in range(buffers):
            self._pool.put(np.empty((height, width, 4), np.uint8))
        self._ready: queue.Queue = queue.Queue()
        self._waiting: tuple[int, np.ndarray] | None = None
        self._ended = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._guarded_run, args=(start,), name=f"stream-{name}", daemon=True)
        self._thread.start()

    def _guarded_run(self, frame: int) -> None:
        try:
            self._run(frame)
        except Exception as error:  # noqa: BLE001 - the player reports it
            if not self._stop.is_set():
                self.error = error
        finally:
            self._ready.put(None)

    def _run(self, frame: int) -> None:
        raise NotImplementedError

    def _take_free(self) -> np.ndarray | None:
        while not self._stop.is_set():
            try:
                return self._pool.get(timeout=0.1)
            except queue.Empty:
                continue
        return None

    def take(self, frame: int, timeout: float) -> np.ndarray | None:
        """The buffer holding `frame`, dropping frames already passed; None when it is not decoded within
        `timeout` (or the stream ended: see `ended`). Release the buffer when it is no longer shown."""
        deadline = time.perf_counter() + timeout
        while True:
            if self._waiting is not None:
                number, buffer = self._waiting
                if number == frame:
                    self._waiting = None
                    return buffer
                if (number - frame) * self.direction > 0:
                    return None  # the next frame decoded comes later in playing order
                self._waiting = None
                self._pool.put(buffer)
            if self._ended:
                return None
            try:
                item = self._ready.get(timeout=max(0.0, deadline - time.perf_counter()))
            except queue.Empty:
                return None
            if item is None:
                self._ended = True
                if self.error is not None:
                    raise self.error
                return None
            self._waiting = item

    @property
    def ended(self) -> bool:
        return self._ended

    def release(self, buffer: np.ndarray) -> None:
        self._pool.put(buffer)

    def request_stop(self) -> None:
        """Ask the decoding thread to stop (`stop` waits for it)."""
        self._stop.set()

    def put_back(self, frame: int, buffer: np.ndarray) -> None:
        """Return the buffer `take` just gave for `frame`, unshown: the next `take` of that frame gets it again."""
        self._waiting = (frame, buffer)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


class FrameStream(_Stream):
    """Decodes forward from `start`, segment after segment; the next segment's decoder starts before the current
    one ends. With `skip` (frame -> the next frame that has a segment, or None), gaps are jumped."""

    def __init__(self, segments: Callable[[int], Segment | None], start: int, width: int, height: int, name: str,
                 strict: bool, skip: Callable[[int], int | None] | None = None):
        super().__init__(segments, start, width, height, name, strict, STREAM_BUFFERS, skip)

    def _open(self, frame: int) -> tuple[FrameReader, int] | None:
        segment = self._segments(frame)
        return None if segment is None else (segment[0](frame), segment[2])

    def _open_from(self, frame: int) -> tuple[FrameReader, int, int] | None:
        """(reader, end, first frame) of the segment holding `frame` (or the next one, for a stream that skips
        gaps), waiting while nothing is ready yet; None once stopped or past the last segment."""
        while not self._stop.is_set():
            segment = self._open(frame)
            if segment is not None:
                return segment[0], segment[1], frame
            if self._skip is not None:
                following = self._skip(frame)
                if following is None or following <= frame:
                    return None
                frame = following
                continue
            self._stop.wait(SEGMENT_POLL_SECONDS)
        return None

    def _run(self, frame: int) -> None:
        current = upcoming = None
        try:
            current = self._open_from(frame)
            while current is not None and not self._stop.is_set():
                reader, end, frame = current
                if frame >= end:
                    reader.stop()
                    if upcoming is not None:
                        current, upcoming = (upcoming[0], upcoming[1], frame), None
                    else:
                        current = self._open_from(frame)
                    continue
                if upcoming is None and end - frame <= PREFETCH_FRAMES:
                    upcoming = self._open(end)
                buffer = self._take_free()
                if buffer is None:
                    break
                if not reader.read_into(buffer):
                    self._pool.put(buffer)
                    if self.strict:
                        raise RuntimeError(f"{self.name} ended at frame {frame}, before {end}")
                    break
                self._ready.put((frame, buffer))
                current = (reader, end, frame + 1)
        finally:
            for segment in (current, upcoming):
                if segment is not None:
                    segment[0].stop()


class ReverseFrameStream(_Stream):
    """Decodes backwards from `start`: a chunk of frames ending at the next frame wanted is decoded forward (one
    seek) and handed over last frame first while the chunk before it decodes. With `skip` (frame -> the last
    frame at or before it that has a segment, or None), gaps are jumped."""

    direction = -1

    def __init__(self, segments: Callable[[int], Segment | None], start: int, width: int, height: int, name: str,
                 strict: bool, skip: Callable[[int], int | None] | None = None):
        self._chunk = reverse_chunk(width, height)
        super().__init__(segments, start, width, height, name, strict, 2 * self._chunk + 2, skip)

    def _run(self, frame: int) -> None:
        upcoming = frame
        while upcoming >= 0 and not self._stop.is_set():
            segment = self._segments(upcoming)
            if segment is None:
                if self._skip is None:
                    self._stop.wait(SEGMENT_POLL_SECONDS)  # not upscaled yet: wait for the renderer
                    continue
                previous = self._skip(upcoming)
                if previous is None:
                    return
                upcoming = previous
                continue
            open_reader, start, _ = segment
            first = max(start, upcoming - self._chunk + 1)
            buffers = []
            while len(buffers) < upcoming + 1 - first:
                buffer = self._take_free()
                if buffer is None:
                    return
                buffers.append(buffer)
            decoded = 0
            reader = open_reader(first)
            try:
                for buffer in buffers:
                    if self._stop.is_set() or not reader.read_into(buffer):
                        break
                    decoded += 1
            finally:
                reader.stop()
            if self._stop.is_set():
                return
            for buffer in buffers[decoded:]:
                self._pool.put(buffer)
            if decoded < len(buffers) and self.strict:
                raise RuntimeError(f"{self.name} ended at frame {first + decoded}, before {upcoming + 1}")
            for index in range(decoded - 1, -1, -1):  # past the end of a source that ends early: what exists
                self._ready.put((first + index, buffers[index]))
            upcoming = first - 1


class Player:
    """See the module docstring. Public methods are called from the Tk thread."""

    def __init__(self, view, source: Path, info: MediaInfo, keyframes: KeyframeIndex,
                 on_event: Callable[..., None]):
        self.view = view
        self.source = source
        self.info = info
        self.keyframes = keyframes
        self.on_event = on_event
        self.rate = float(info.frame_rate)
        self.timeline: Timeline | None = None
        self.output_size = (info.width, info.height)
        self.layers: dict[str, tuple[Timeline, bool]] = {}
        self.position = 0
        self.playing = False
        self.buffering = False
        self.speed = 1.0
        self.volume = 0.8
        self.muted = False
        self.render_speed = 0.0
        self.render_active = False
        self.follow_live = False
        self._commands: queue.Queue = queue.Queue()
        self._deferred: deque[tuple] = deque()
        self._still_wanted: int | None = None
        self._audio: AudioPlayer | None = None
        # Opening the sound device takes about half a second: do it now, not when Play is pressed.
        self._audio_thread = threading.Thread(target=self._prepare_audio, name="audio-open", daemon=True)
        if info.audio:
            self._audio_thread.start()
        self._streams: dict[str, _Stream] = {}
        self._stream_keys: dict[str, tuple] = {}
        self._stream_layers: set[str] = set()
        self._held: list[tuple[_Stream, np.ndarray]] = []
        self._resume_frame = 0
        self._clock_frame0 = 0
        self._clock_perf0 = 0.0
        self._last_audio = (None, 0.0)
        self._last_position_event = 0.0
        self._last_spinner = 0.0
        self._showing_unready = False
        self._missing_layers: set[str] = set()
        self._target_key: tuple | None = None
        self._target_since = 0.0
        self._caught_up = False
        self._decoders = ThreadPoolExecutor(max_workers=3, thread_name_prefix="still")
        self._thread = threading.Thread(target=self._main, name="player", daemon=True)
        self._thread.start()

    # ---- commands (Tk thread) -------------------------------------------------------------------

    @property
    def total(self) -> int:
        return self.timeline.total if self.timeline is not None else (self.info.frame_count or 1)

    @property
    def direction(self) -> int:
        """1 playing forward, -1 backwards."""
        return 1 if self.speed > 0 else -1

    def attach(self, timeline: Timeline | None, output_size: tuple[int, int],
               layers: dict[str, tuple[Timeline, bool]] | None = None) -> None:
        """A new renderer, new settings or a new choice of what the view shows: the current version's timeline
        (it sets the video's length), the output size, and for each upscaled layer the view can show, its
        timeline and whether playback waits for it (a version being upscaled) or shows what it has."""
        self._commands.put(("attach", timeline, output_size, dict(layers or {})))

    def play(self) -> None:
        self._commands.put(("play",))

    def _next_command(self, timeout: float | None) -> tuple | None:
        """The next command, waiting up to `timeout` (None: forever, 0: not at all). Consecutive seeks queued
        behind each other become the last one, so a drag never leaves a backlog."""
        if self._deferred:
            return self._deferred.popleft()
        try:
            command = self._commands.get(timeout=timeout) if timeout != 0.0 else self._commands.get_nowait()
        except queue.Empty:
            return None
        while command[0] in ("seek", "step"):
            try:
                following = self._commands.get_nowait()
            except queue.Empty:
                break
            if following[0] not in ("seek", "step"):
                self._deferred.append(following)
                break
            command = merge_seeks(command, following)
        return command

    def pause(self) -> None:
        self._commands.put(("pause",))

    def toggle(self) -> None:
        self._commands.put(("toggle",))

    def seek(self, frame: int) -> None:
        self._commands.put(("seek", frame))

    def step(self, frames: int) -> None:
        self._commands.put(("step", frames))

    def set_speed(self, speed: float) -> None:
        """Playing speed: 1 is real time, 0.5 half speed, -1 backwards in real time (never 0)."""
        if speed == 0:
            raise ValueError("a speed of 0 is a pause")
        self._commands.put(("speed", speed))

    def set_volume(self, volume: float, muted: bool) -> None:
        self._commands.put(("volume", volume, muted))

    def refresh(self) -> None:
        """A piece became ready: update a still that was waiting for it, end buffering when possible."""
        self._commands.put(("refresh",))

    def show_live(self, frame: int, source_rgba: np.ndarray, upscaled_rgba: np.ndarray) -> None:
        """A frame the renderer just made; shown while paused and following the renderer."""
        self._commands.put(("live", frame, source_rgba, upscaled_rgba))

    def close(self) -> None:
        self._commands.put(("close",))
        self._thread.join()
        self._decoders.shutdown(wait=True)

    # ---- the player thread ----------------------------------------------------------------------

    def _main(self) -> None:
        try:
            while True:
                if (self.playing and not self.buffering) or (not self.playing and self._still_wanted is not None):
                    timeout = 0.0
                elif self.buffering or (self._showing_unready and self.render_active):
                    timeout = SPINNER_INTERVAL
                else:
                    timeout = None
                command = self._next_command(timeout)
                if command is not None and command[0] == "close":
                    break
                try:
                    if command is not None:
                        self._handle(command)
                    elif self.playing:
                        self._play_step()
                    elif self._still_wanted is not None:
                        self._load_still()
                    elif self._showing_unready and self.render_active:
                        self.view.set_spinner(True, advance=True)
                except Exception as error:  # noqa: BLE001 - reported to the window; the player stays usable
                    log.emit(f"[player] error: {error}")
                    try:
                        if self.playing:
                            self._stop_playback()
                    except Exception as cleanup_error:  # noqa: BLE001 - keep the thread alive, say what failed
                        log.emit(f"[player] error while stopping: {cleanup_error}")
                    self._still_wanted = None
                    self.on_event("error", error)
        finally:
            self._stop_streams()
            if self._audio_thread.is_alive():
                self._audio_thread.join()
            if self._audio is not None:
                try:
                    self._audio.close()
                except OSError as error:
                    log.emit(f"[player] could not close the sound device: {error}")

    def _handle(self, command: tuple) -> None:
        kind = command[0]
        if kind == "attach":
            _, self.timeline, self.output_size, self.layers = command
            self.position = min(self.position, self.total - 1)
            if self.playing:
                self._restart(self.position)
            else:
                self._still_wanted = self.position
        elif kind in ("play", "toggle") and not self.playing and self.timeline is None:
            self.on_event("notice", "Nothing to play yet: the upscaler is not running (see the GPU status).")
        elif kind in ("play", "toggle") and not self.playing:
            self.follow_live = False
            self.position = min(self.position, self.total - 1)  # the frame count may have shrunk to the real one
            if self.speed > 0 and self.position >= self.total - 1:
                self.position = 0
            elif self.speed < 0 and self.position <= 0:
                self.position = self.total - 1
            self.playing = True
            self._restart(self.position)
        elif kind in ("pause", "toggle") and self.playing:
            self._stop_playback()
        elif kind in ("seek", "step"):
            self.follow_live = False
            frame = command[1] if kind == "seek" else self.position + command[1]
            frame = max(0, min(self.total - 1, frame))
            self.position = frame
            if self.playing:
                self._restart(frame)
            else:
                self._still_wanted = frame
            self._position_event(force=True)
        elif kind == "speed" and command[1] != self.speed:
            self.speed = command[1]
            if self.playing:
                self._restart(self.position)
        elif kind == "volume":
            _, self.volume, self.muted = command
            self._sound(lambda audio: audio.set_volume(0.0 if self.muted else self.volume))
        elif kind == "refresh":
            if not self.playing and self._missing_layers:
                for layer in self._missing_layers:
                    timeline = self.layers.get(layer, (None, False))[0]
                    if timeline is not None and timeline.piece_at(self.position, ready_only=True) is not None:
                        self._still_wanted = self.position
        elif kind == "live":
            _, frame, source_rgba, upscaled_rgba = command
            if self.follow_live and not self.playing:
                import cv2
                self.position = frame
                upscaled = cv2.cvtColor(upscaled_rgba, cv2.COLOR_RGBA2BGRA)
                # The renderer's frames are the current version's: every layer showing that version gets them.
                layers = {layer: upscaled for layer, (timeline, _) in self.layers.items() if timeline is self.timeline}
                self.view.show_still(cv2.cvtColor(source_rgba, cv2.COLOR_RGBA2BGRA), layers)
                self._showing_unready = False
                self._missing_layers = set()
                self._position_event(force=True)

    # ---- playback -------------------------------------------------------------------------------

    def _restart(self, frame: int) -> None:
        """(Re)start playing at `frame`: buffering first when it is not upscaled yet."""
        self._still_wanted = None  # a still asked for before Play: playing shows the frame instead
        self._caught_up = False
        self._stop_streams()
        self._release_held()
        self.position = frame
        self._resume_frame = frame
        self._position_event(force=True)
        self._sound(lambda audio: audio.pause())
        if not self._try_resume():
            self._set_buffering(True)
            # Show where the playhead is while the renderer catches up there.
            source = read_frame(self.source, self.info, frame, self.keyframes.get(), "bgra")
            self.view.show_still(source, {}, "upscaling…", spinner=True)

    def _paced(self, layers) -> list[str]:
        """The layers among `layers` that playback waits for."""
        return [layer for layer in layers if self.layers.get(layer, (None, False))[1]]

    def _ready_limit(self, timeline: Timeline, frame: int) -> int:
        """The first frame from `frame` on, in the playing direction, that is not upscaled yet (-1 backwards
        when everything down to the start is)."""
        return timeline.ready_until(frame) if self.speed > 0 else timeline.ready_back_to(frame) - 1

    def _buffered(self, timeline: Timeline, frame: int, versions: int) -> bool:
        """Whether enough of `timeline` is ready from `frame` on, in the playing direction, to start playing."""
        limit = self._ready_limit(timeline, frame)
        ready = (limit - frame) * self.direction
        to_the_end = limit < 0 if self.speed < 0 else timeline.exact and limit >= timeline.total
        return ready > 0 and (to_the_end or ready >= self._buffer_needed(versions))

    def _try_resume(self) -> bool:
        """Start the decoders, show the first frame and start the clock once enough is upscaled ahead in every
        layer shown that is being upscaled (the source and the other layers never wait)."""
        if self.timeline is None:
            return False
        frame = self._resume_frame
        layers = self.view.needed_layers()
        paced = self._paced(layers)
        if not all(self._buffered(self.layers[layer][0], frame, len(paced)) for layer in paced):
            return False
        self._start_streams(frame, layers)
        source_stream = self._streams["source"]
        source = source_stream.take(frame, FIRST_FRAME_TIMEOUT)
        if source is None:
            raise RuntimeError(f"could not decode frame {frame} for playback")
        held = [(source_stream, source)]
        shown = {}
        for layer in layers:
            stream = self._streams.get(layer)
            timeout = FIRST_FRAME_TIMEOUT if layer in paced else LENIENT_FIRST_FRAME_TIMEOUT
            image = stream.take(frame, timeout) if stream is not None else None
            if image is None and layer in paced:
                raise RuntimeError(f"could not decode frame {frame} for playback")
            shown[layer] = None if image is None else (image, self._stream_keys[layer])
            if image is not None:
                held.append((stream, image))
        self.view.present(source, shown)
        self._held = held
        self.position = frame
        self._resume_frame = frame + self.direction
        self._start_clock(frame)
        self._set_buffering(False)
        self._position_event(force=True)
        return True

    def _buffer_needed(self, versions: int) -> int:
        """Frames to have ready ahead before playing: a short wait to start, and after playback caught up with a
        renderer slower than playback, enough to play about BUFFER_HORIZON_SECONDS before the next wait. The
        renderer shares its speed among the `versions` being upscaled."""
        seconds = BUFFER_MIN_SECONDS
        if self._caught_up and self.render_speed > 0 and self.rate:
            ratio = self.render_speed / max(1, versions) / (self.rate * abs(self.speed))
            seconds = min(BUFFER_MAX_SECONDS, max(BUFFER_MIN_SECONDS, (1 - ratio) * BUFFER_HORIZON_SECONDS))
        return max(1, int(round(seconds * self.rate)))

    def _set_buffering(self, buffering: bool) -> None:
        if buffering != self.buffering:
            self.buffering = buffering
            self.view.set_spinner(buffering)
        self.on_event("state", self.playing, self.buffering)
        self._showing_unready = False

    def _start_streams(self, frame: int, layers: set[str]) -> None:
        keyframes = self.keyframes.get()
        source, info = self.source, self.info
        stream_type = FrameStream if self.speed > 0 else ReverseFrameStream
        self._streams["source"] = stream_type(
            lambda start: ((lambda first: FrameReader(source, info, start_frame=first, keyframes=keyframes,
                                                      pixel_format="bgra")), 0, self.total),
            frame, info.width, info.height, "source", strict=False)
        self._stream_layers = set(layers)
        for layer in layers:
            self._start_layer_stream(layer, frame)

    def _start_layer_stream(self, layer: str, frame: int) -> None:
        """Decode `layer`'s ready pieces from `frame`, cropped and scaled to the view by ffmpeg."""
        timeline, paced = self.layers.get(layer, (None, False))
        if timeline is None:
            return
        output_size = self.output_size
        key, crop, size, flags = self.view.decode_target()
        rate = self.info.frame_rate

        def segment(start: int) -> Segment | None:
            piece = timeline.piece_at(start, ready_only=True)
            if piece is None:
                return None
            info = MediaInfo(width=output_size[0], height=output_size[1], frame_rate=rate, frame_count=piece.frames,
                             duration=None, codec="", color_matrix="bt709", container=PIECE_CONTAINER)
            return ((lambda first: FrameReader(piece.path, info, start_frame=first - piece.start, crop=crop,
                                               size=size, scale_flags=flags, pixel_format="bgra")),
                    piece.start, piece.end)

        if self.speed > 0:
            self._streams[layer] = FrameStream(segment, frame, size[0], size[1], "upscaled pieces", strict=True,
                                               skip=None if paced else timeline.next_ready_start)
        else:
            self._streams[layer] = ReverseFrameStream(segment, frame, size[0], size[1], "upscaled pieces",
                                                      strict=True, skip=None if paced else timeline.last_ready_before)
        self._stream_keys[layer] = key

    def _stop_streams(self) -> None:
        for stream in self._streams.values():
            stream.request_stop()  # all at once: each finishes its frame (or chunk) while the others do
        for stream in self._streams.values():
            stream.stop()
        self._streams.clear()

    def _release_held(self) -> None:
        for stream, buffer in self._held:
            stream.release(buffer)
        self._held = []

    def _sound_plays(self) -> bool:
        """Whether the sound plays at this speed (and the sound card keeps time)."""
        return bool(self.info.audio) and SOUND_SPEEDS[0] <= self.speed <= SOUND_SPEEDS[1]

    def _every_frame(self) -> bool:
        """Whether every frame is shown in order (faster than the sound plays, and backwards), slowing down when
        the decoders cannot keep up."""
        return self.speed < 0 or self.speed > SOUND_SPEEDS[1]

    def _start_clock(self, frame: int) -> None:
        self._clock_frame0 = frame
        self._clock_perf0 = time.perf_counter()
        self._last_audio = (None, 0.0)
        if self._sound_plays() and self._open_audio() is not None:
            self._sound(lambda audio: audio.start(frame / self.rate, self.speed))

    def _sound(self, call: Callable[[AudioPlayer], None]) -> None:
        """Run `call` on the sound device; one that fails (unplugged headset) is dropped, playback goes on."""
        if self._audio is None:
            return
        try:
            call(self._audio)
        except OSError as error:
            self._drop_audio(error)

    def _drop_audio(self, error: Exception) -> None:
        log.emit(f"[player] sound stopped: {error}")
        audio, self._audio = self._audio, None
        try:
            audio.close()
        except OSError as close_error:
            log.emit(f"[player] could not close the sound device: {close_error}")
        # Keep time from here with the system clock.
        self._clock_frame0 = self.position + self.direction
        self._clock_perf0 = time.perf_counter()
        self.on_event("notice", f"Sound stopped ({error}); playing on without it.")

    def _prepare_audio(self) -> None:
        try:
            self._audio = AudioPlayer(self.source, self.info)
        except OSError as error:
            log.emit(f"[player] no sound: {error}")
            self.on_event("notice", f"Playing without sound: {error}")

    def _open_audio(self) -> AudioPlayer | None:
        if self._audio_thread.is_alive():
            self._audio_thread.join()
        self._sound(lambda audio: audio.set_volume(0.0 if self.muted else self.volume))
        return self._audio

    def _clock_frame(self) -> float:
        """Frames since the start of the file: from the sound card while the sound plays, else from the system
        clock at the playing speed."""
        now = time.perf_counter()
        audio = self._audio if self._sound_plays() else None
        if audio is not None and audio.error is not None:
            self._drop_audio(audio.error)
            audio = None
        if audio is not None:
            try:
                seconds = audio.position()
            except OSError as error:
                self._drop_audio(error)
                return self._clock_frame()
            previous, since = self._last_audio
            if seconds != previous:
                self._last_audio = (seconds, now)
            elif previous is not None and now - since > AUDIO_STALL_SECONDS:
                # The track ended (or the device stalled): keep time with the system clock from there.
                return (previous + (now - since) * self.speed) * self.rate
            return seconds * self.rate
        return self._clock_frame0 + (now - self._clock_perf0) * self.rate * self.speed

    def _play_step(self) -> None:
        if self.buffering:
            now = time.perf_counter()
            if now - self._last_spinner >= SPINNER_INTERVAL:
                self._last_spinner = now
                self.view.set_spinner(True, advance=True)
            self._try_resume()
            return
        direction = self.direction
        clock = self._clock_frame()
        timeline = self.timeline
        if direction > 0:
            due = max(int(math.floor(clock + 1e-6)), self._clock_frame0)
        else:
            due = min(int(math.ceil(clock - 1e-6)), self._clock_frame0)
        if self._every_frame():
            following = self._resume_frame
            if (due - following) * direction < 0:
                self._wait((following - clock) * direction / (self.rate * abs(self.speed)))
                return
            if (due - following) * direction > 1:
                # Late (the decoders are slower than this speed): go on from this frame at the right pace.
                self._clock_frame0, self._clock_perf0 = following, time.perf_counter()
            due = following
        if (direction > 0 and timeline.exact and due >= timeline.total) or due < 0:
            self._finish()
            return
        paced = self._paced(self._stream_layers)
        limits = [self._ready_limit(self.layers[layer][0], self._resume_frame) for layer in paced]
        limit = (min(limits) if direction > 0 else max(limits)) if limits else None
        if limit is not None and (limit - due) * direction <= 0:
            # The next frames are not upscaled yet in every layer shown: wait for the renderer.
            self._resume_frame = limit
            if (timeline.exact and limit >= timeline.total) or limit < 0:
                self._finish()
                return
            self._stop_streams()
            self._release_held()
            self._sound(lambda audio: audio.pause())
            self._caught_up = True
            self._position_event(force=True)  # the renderer works from the playhead, not from a late one
            self._set_buffering(True)
            return
        frame_time = 1 / (self.rate * abs(self.speed))
        source_stream = self._streams["source"]
        held, shown = [], {}
        for layer in paced:
            # The upscaled layers first: if a decoder is late, skip this tick without consuming the source.
            stream = self._streams[layer]
            image = stream.take(due, frame_time)
            if image is None:
                for owner, buffer in held:
                    owner.put_back(due, buffer)  # this frame is tried again (frame by frame, it must be shown)
                if stream.ended:
                    self._resume_frame = due
                    self._stop_streams()
                    self._release_held()
                    self._sound(lambda audio: audio.pause())
                    self._position_event(force=True)
                    self._set_buffering(True)
                return
            held.append((stream, image))
            shown[layer] = (image, self._stream_keys[layer])
        source = source_stream.take(due, frame_time)
        if source is None:
            for stream, buffer in held:
                stream.put_back(due, buffer)
            if source_stream.ended:
                self._finish()
            return
        held.append((source_stream, source))
        for layer in self._stream_layers - set(paced):
            stream = self._streams.get(layer)
            image = stream.take(due, frame_time / 2) if stream is not None else None
            shown[layer] = None if image is None else (image, self._stream_keys[layer])
            if image is not None:
                held.append((stream, image))
        self.view.present(source, shown)
        self._release_held()
        self._held = held
        self.position = due
        self._resume_frame = due + direction
        self._position_event()
        # Zoomed or panned: decode at the new geometry once it has settled (the view stretches the old meanwhile).
        target, now = self.view.decode_target()[0], time.perf_counter()
        if target != self._target_key:
            self._target_key, self._target_since = target, now
        for layer in self._stream_layers:
            stream = self._streams.get(layer)
            if (stream is not None and target != self._stream_keys.get(layer)
                    and now - self._target_since >= RESTART_SETTLE_SECONDS):
                stream.stop()
                self._held = [(owner, buffer) for owner, buffer in self._held if owner is not stream]
                self._start_layer_stream(layer, due + direction)
        self._wait((due + direction - self._clock_frame()) * direction / (self.rate * abs(self.speed)))

    def _wait(self, seconds: float) -> None:
        """Wait up to `seconds` for the next frame's time, handling a command that arrives meanwhile."""
        if seconds <= 0:
            return
        command = self._next_command(seconds)
        if command is None:
            return
        if command[0] == "close":
            self._deferred.appendleft(command)
        else:
            self._handle(command)

    def _finish(self) -> None:
        """Playback reached the end (or the start, backwards): stop there."""
        end = self.total - 1 if self.speed > 0 else 0
        self._stop_playback()
        self.position = end
        self._still_wanted = end
        self._position_event(force=True)
        self.on_event("ended")

    def _stop_playback(self) -> None:
        self.playing = False
        self._stop_streams()
        self._sound(lambda audio: audio.pause())
        self._set_buffering(False)
        self._still_wanted = self.position
        self._position_event(force=True)

    def _position_event(self, force: bool = False) -> None:
        now = time.perf_counter()
        if force or now - self._last_position_event >= POSITION_INTERVAL:
            self._last_position_event = now
            self.on_event("position", self.position)

    # ---- stills ---------------------------------------------------------------------------------

    def _load_still(self) -> None:
        frame = self._still_wanted
        self._still_wanted = None
        source_future = self._decoders.submit(read_frame, self.source, self.info, frame, self.keyframes.get(), "bgra")
        futures = {}
        size = self.output_size
        for layer in self.view.needed_layers():
            timeline = self.layers.get(layer, (None, False))[0]
            piece = timeline.piece_at(frame, ready_only=True) if timeline is not None else None
            if piece is None:
                futures[layer] = None
                continue
            info = MediaInfo(width=size[0], height=size[1], frame_rate=self.info.frame_rate, frame_count=piece.frames,
                             duration=None, codec="", color_matrix="bt709", container=PIECE_CONTAINER)
            futures[layer] = self._decoders.submit(read_frame, piece.path, info, frame - piece.start, None, "bgra")
        source = source_future.result()
        layers = {layer: future.result() if future is not None else None for layer, future in futures.items()}
        if self._still_wanted is not None or self.playing:
            return  # a newer request arrived while decoding
        self._missing_layers = {layer for layer, image in layers.items() if image is None}
        self._showing_unready = bool(self._paced(self._missing_layers))
        busy = self._showing_unready and self.render_active
        self.view.show_still(source, layers, "upscaling…" if busy else "", spinner=busy)
