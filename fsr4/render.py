"""Upscales a video into pieces on a worker thread, following the playhead, and joins them into the output.

The window plays the upscaled video while it is being made: the renderer works where the playhead is
(preview mode keeps a stretch ahead of it upscaled, batch mode does the whole file starting there), writes
consecutive frames into piece files, and the player plays the finished pieces. A seek makes the renderer
finish its current piece and start over at the new position, with a few frames of pre-roll so FSR's history
and the optical flow are warm. Saving joins the pieces without re-encoding and adds the source's audio.

Changing a setting starts a new version; the previous one is kept as "before" (once it has a second or more
upscaled) so either side of the comparison can show it, and returning to its settings switches back to it at once.

Events reach the window through `on_event(kind, *values)`, called from the renderer's threads:
"piece" (a Piece became playable), "live" (frame, source RGBA, upscaled RGBA; twice a second while
`want_live`), "status" (frame being rendered or None when idle, frames per second), "total" (the real frame
count), "complete" (every frame is ready), "error" (an exception; rendering stops until settings change).
"""
from __future__ import annotations

import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import log, usage
from .flow import create_flow
from .media import FrameReader, FrameWriter, KeyframeIndex, MediaInfo, join_pieces
from .timeline import Piece, Timeline, keep_going, next_start, piece_lengths

PREROLL_FRAMES = 8
LIVE_INTERVAL = 0.5
STATUS_INTERVAL = 0.25
PREVIEW_LOW_SECONDS = 10.0
PREVIEW_HIGH_SECONDS = 30.0
GOP_SECONDS = 2.0
THUMBNAIL_HEIGHT = 108
MAX_THUMBNAILS = 240
DIRECTORY_PREFIX = "fsr4-"
STALE_DIRECTORY_SECONDS = 24 * 3600
MIN_FREE_BYTES = 2 * 1024 ** 3

EventCallback = Callable[..., None]


UPSCALER_NAMES = {None: "FSR 4", "3.1.6": "FSR 3.1", "2.3.4": "FSR 2.3"}
ENCODER_SHORT_NAMES = {"amf": "HEVC GPU", "av1_amf": "AV1 GPU", "h264_amf": "H.264 GPU", "x264": "x264",
                       "x265": "x265"}


@dataclass(frozen=True)
class RenderSettings:
    """Everything the upscaled pieces depend on; changing any of it starts a new version. `upscaler` names
    one of the FSR DLL's upscalers ("3.1.6", "2.3.4"); None is FSR 4."""
    output_size: tuple[int, int]
    flow: str = "dis"
    flow_preset: str = "fast"
    sharpness: float | None = None
    encoder: str = "amf"
    upscaler: str | None = None

    @property
    def motion(self) -> str:
        return "no motion" if self.flow == "zero" else f"flow {self.flow_preset}"

    def describe(self, other: "RenderSettings | None" = None) -> str:
        """A short name: the upscaler, plus whatever differs from `other` (the version it is compared with)."""
        parts = [UPSCALER_NAMES.get(self.upscaler, f"FSR {self.upscaler}")]
        if other is None:
            return parts[0]
        if self.output_size != other.output_size:
            parts.append(f"{self.output_size[0]} × {self.output_size[1]}")
        if self.motion != other.motion:
            parts.append(self.motion)
        if self.sharpness != other.sharpness:
            parts.append(f"sharpening {self.sharpness:.2f}" if self.sharpness is not None else "no sharpening")
        if self.encoder != other.encoder:
            parts.append(ENCODER_SHORT_NAMES.get(self.encoder, self.encoder))
        return " · ".join(parts)


@dataclass
class Version:
    """The pieces upscaled with one set of settings, and their filmstrip pictures."""
    settings: RenderSettings
    timeline: Timeline
    generation: int
    thumbnails: dict[int, np.ndarray] = field(default_factory=dict)


def clean_stale_directories() -> list[Path]:
    """Delete piece folders left by sessions that ended without cleaning up (older than a day)."""
    removed = []
    cutoff = time.time() - STALE_DIRECTORY_SECONDS
    for directory in Path(tempfile.gettempdir()).glob(f"{DIRECTORY_PREFIX}*"):
        if directory.is_dir() and directory.stat().st_mtime < cutoff:
            shutil.rmtree(directory)
            removed.append(directory)
    return removed


class Renderer:
    """See the module docstring. Thread-safe; `close()` stops it and deletes its pieces."""

    def __init__(self, source: Path, info: MediaInfo, settings: RenderSettings, sdk_dir: Path,
                 on_event: EventCallback, keyframes: KeyframeIndex | None = None, target: int = 0):
        if not info.frame_rate:
            raise ValueError(f"{source.name} has no frame rate")
        self.source = source
        self.info = info
        self.sdk_dir = sdk_dir
        self.on_event = on_event
        self.keyframes = keyframes or KeyframeIndex(source, info)
        self.rate = float(info.frame_rate)
        self.directory = Path(tempfile.mkdtemp(prefix=DIRECTORY_PREFIX))
        self._known_total: int | None = None
        self._current = Version(settings, self._new_timeline(), 0)
        self._before: Version | None = None
        self._thumbnail_lock = threading.Lock()
        self.thumbnail_stride = max(1, int(round(self.rate)), (info.frame_count or 1) // MAX_THUMBNAILS)
        self.want_live = False
        self._condition = threading.Condition()
        self._generation = 0
        self._target = target
        self._batch = False
        self._closing = False
        self._restart_run = False
        self._failed = False
        self._idle_reported = False
        self._gpu = None
        self._pipeline = None
        self._pipeline_generation = -1
        self._stale_files: list[Path] = []
        self._closer = ThreadPoolExecutor(max_workers=2, thread_name_prefix="piece-closer")
        self._thread = threading.Thread(target=self._main, name="renderer", daemon=True)
        self._thread.start()

    # ---- control (any thread) -------------------------------------------------------------------

    @property
    def settings(self) -> RenderSettings:
        return self._current.settings

    @property
    def timeline(self) -> Timeline:
        """The pieces of the current settings."""
        return self._current.timeline

    @property
    def before(self) -> Version | None:
        """The version made with the previous settings, kept for comparison (None if there is none)."""
        return self._before

    def _new_timeline(self) -> Timeline:
        if self._known_total is not None:
            return Timeline(self._known_total, exact=True)
        return Timeline(self.info.frame_count or 1)

    @property
    def batch(self) -> bool:
        return self._batch

    def set_target(self, frame: int) -> None:
        """The playhead moved: render from there."""
        with self._condition:
            self._target = max(0, frame)
            self._condition.notify_all()

    def set_batch(self, batch: bool) -> None:
        """Batch mode upscales the whole file (Upscale); preview mode only what is ahead of the playhead.
        Starting a batch also retries after an error (the user may have freed disk space, for example)."""
        with self._condition:
            self._batch = batch
            if batch:
                self._failed = False
            self._idle_reported = False
            self._condition.notify_all()

    def restart_processes(self) -> None:
        """Processor threads or priority changed: end the current run at the next frame (its piece is kept) so
        the next one starts its decoder and encoders with the new limits."""
        with self._condition:
            self._restart_run = True
            self._condition.notify_all()

    def change_settings(self, settings: RenderSettings) -> None:
        """New settings: a new version upscales from the playhead. The current one becomes "before" if it has a
        second or more upscaled (else it is dropped); settings equal to "before" switch back to it."""
        with self._condition:
            if settings == self._current.settings:
                if self._failed:  # the same settings again: retry after an error
                    self._failed = False
                    self._idle_reported = False
                    self._condition.notify_all()
                return
            self._failed = False
            self._idle_reported = False
            dropped: Version | None = None
            if self._before is not None and settings == self._before.settings:
                self._current, self._before = self._before, self._current
            else:
                self._generation += 1
                fresh = Version(settings, self._new_timeline(), self._generation)
                if self._current.timeline.ready_frames() >= self._frames(1.0):
                    dropped, self._before = self._before, self._current
                else:
                    dropped = self._current
                self._current = fresh
            self._condition.notify_all()
        if dropped is not None:
            self._delete([piece.path for piece in dropped.timeline.clear() if piece.ready])

    def _delete(self, paths: list[Path]) -> None:
        """Delete piece files; one the player still has open is kept and retried after the next piece."""
        kept = []
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except PermissionError:
                kept.append(path)  # a playback decoder still reads it; Windows refuses until it closes
        with self._condition:
            self._stale_files.extend(kept)

    def close(self) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join()
        self._closer.shutdown(wait=True)
        try:
            shutil.rmtree(self.directory)
        except OSError as error:
            # A file still open elsewhere (a save in progress): the folder is removed a day later.
            log.emit(f"[render] could not delete {self.directory} now: {error}")

    # ---- saving ---------------------------------------------------------------------------------

    def export(self, output: Path, partial: bool = False) -> int:
        """Join the ready pieces into `output` with the source's audio; returns the frames written.

        Everything must be ready unless `partial`, which saves the frames from the start up to the first gap.
        """
        version = self._current
        pieces = version.timeline.ready_prefix()
        frames = pieces[-1].end if pieces else 0
        if frames == 0:
            raise RuntimeError("nothing is upscaled from the start of the video yet")
        if not partial and not version.timeline.complete():
            raise RuntimeError(f"only {frames} of {version.timeline.total} frames are upscaled")
        seconds = frames / self.rate if frames < version.timeline.total else None
        join_pieces([piece.path for piece in pieces], output, version.settings.encoder, self.source,
                    self.info.audio, seconds)
        return frames

    # ---- the worker thread ----------------------------------------------------------------------

    def _main(self) -> None:
        from .gpu import Gpu
        try:
            self._gpu = Gpu(verbose=False)
            self._pipeline_flow = None
            while True:
                with self._condition:
                    start = self._wait_for_work()
                    if start is None:
                        break
                    generation, settings = self._current.generation, self._current.settings
                try:
                    self._ensure_pipeline(generation, settings)
                    self._run(start, generation, settings)
                except Exception as error:  # noqa: BLE001 - reported, then the renderer waits for new settings
                    with self._condition:
                        self._failed = True
                    log.emit(f"[render] error: {error}")
                    self.on_event("error", error)
        except Exception as error:  # noqa: BLE001 - the GPU could not start
            log.emit(f"[render] error: {error}")
            self.on_event("error", error)
        finally:
            if self._pipeline is not None:
                self._pipeline.destroy()
            if self._gpu is not None:
                self._gpu.destroy()

    def _wait_for_work(self) -> int | None:
        """Block until there is something to render (returns its first frame) or the renderer closes."""
        while not self._closing:
            start = None if self._failed else next_start(self.timeline, self._target, self._batch,
                                                         self._frames(PREVIEW_LOW_SECONDS))
            if start is not None:
                self._idle_reported = False
                return start
            if not self._idle_reported:
                self._idle_reported = True
                self.on_event("status", None, 0.0)
            self._condition.wait()
        return None

    def _frames(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.rate)))

    def _ensure_pipeline(self, generation: int, settings: RenderSettings) -> None:
        """The pipeline for `settings`: the same FSR context when only sharpening, motion or encoder changed."""
        if self._pipeline is not None and self._pipeline_generation == generation:
            return
        from .pipeline import Pipeline
        size = (self.info.width, self.info.height)
        flow_settings = (settings.flow, settings.flow_preset)
        pipeline = self._pipeline
        if (pipeline is not None and pipeline.output_size == settings.output_size
                and pipeline.provider == settings.upscaler):
            pipeline.sharpness = settings.sharpness
            if flow_settings != self._pipeline_flow:
                pipeline.flow = create_flow(settings.flow, *size, settings.flow_preset)
        else:
            if pipeline is not None:
                self._pipeline = None
                pipeline.destroy()
            flow = create_flow(settings.flow, *size, settings.flow_preset)
            self._pipeline = Pipeline(self._gpu, size, settings.output_size, self.sdk_dir, flow,
                                      sharpness=settings.sharpness, frame_time_ms=self.info.frame_time_ms,
                                      provider=settings.upscaler)
        self._pipeline_flow = flow_settings
        self._pipeline_generation = generation

    def _should_continue(self, generation: int, busy: tuple[int, int] | None, position: int) -> bool:
        with self._condition:
            if self._closing or generation != self._current.generation:
                return False
            if self._restart_run:
                self._restart_run = False
                return False
            return keep_going(self.timeline, self._target, self._batch, self._frames(PREVIEW_HIGH_SECONDS), busy,
                              position)

    def _run(self, start: int, generation: int, settings: RenderSettings) -> None:
        """Render from `start` until the stretch is filled, a seek, a settings change, or far enough ahead."""
        from .pipeline import upscale_stream
        usage.lower_current_thread()
        pipeline = self._pipeline
        pipeline.reset()
        preroll_start = max(0, start - PREROLL_FRAMES)
        gap_end = self.timeline.gap_end(start)
        open_end = gap_end >= self.timeline.total
        keyframes = self.keyframes.get() if preroll_start > 0 else None
        reader = FrameReader(self.source, self.info, max_frames=None if open_end else gap_end - preroll_start,
                             start_frame=preroll_start, keyframes=keyframes, background=True)
        lengths = piece_lengths(self.rate)
        writer: FrameWriter | None = None
        piece_start = piece_end = start
        position = preroll_start
        exhausted = False
        last_live = last_status = run_started = time.perf_counter()
        written = 0
        try:
            with ThreadPoolExecutor(max_workers=1, initializer=usage.lower_current_thread) as executor:
                stream = upscale_stream(iter(reader), pipeline, executor)
                for source_frame, upscaled in stream:
                    if position < start:
                        position += 1
                        continue
                    if writer is None:
                        piece_start = position
                        piece_end = position + next(lengths)
                        if not open_end:
                            piece_end = min(piece_end, gap_end)
                        writer = self._open_piece(piece_start, generation, settings)
                    writer.write(upscaled)
                    position += 1
                    written += 1
                    self._keep_thumbnail(position - 1, upscaled, generation)
                    now = time.perf_counter()
                    if self.want_live and now - last_live >= LIVE_INTERVAL:
                        self.on_event("live", position - 1, source_frame.copy(), upscaled.copy())
                        last_live = now
                    if now - last_status >= STATUS_INTERVAL:
                        self.on_event("status", position, written / max(now - run_started, 1e-6))
                        last_status = now
                    if position >= piece_end:
                        self._finish_piece(writer, piece_start, position, generation)
                        writer = None
                    busy = (piece_start, position) if writer is not None else None
                    if (not open_end and position >= gap_end) or not self._should_continue(generation, busy, position):
                        break
                else:
                    exhausted = True
                stream.close()
        except BaseException:
            reader.stop()
            if writer is not None:
                writer.abort()
                writer.path.unlink(missing_ok=True)
            raise
        if not exhausted:
            reader.stop()
        if writer is not None:
            if self._closing:
                writer.abort()  # the window is closing: the piece would be deleted right away
                writer.path.unlink(missing_ok=True)
            else:
                self._finish_piece(writer, piece_start, position, generation)
        if exhausted:
            self._reached_end(start, position)

    def _reached_end(self, start: int, position: int) -> None:
        """The decoder ran out of frames: that is the real end of the video (the probe's count is an estimate)."""
        if position == start and start > 0 and start < self.timeline.total - self._frames(1.0):
            raise RuntimeError(f"cannot read {self.source.name} from frame {start}")
        self._known_total = position
        self.timeline.set_total(position)
        self.on_event("total", self.timeline.total)
        if self.timeline.complete():
            self.on_event("complete")

    def _open_piece(self, start: int, generation: int, settings: RenderSettings) -> FrameWriter:
        free = shutil.disk_usage(self.directory).free
        if free < MIN_FREE_BYTES:
            raise RuntimeError(f"only {free / 1024 ** 3:.1f} GB free for the upscaled pieces in {self.directory.parent}; "
                               f"free some space (4K takes about 10 MB per second of video)")
        path = self.directory / f"piece_{generation}_{start:08d}.mp4"
        width, height = settings.output_size
        return FrameWriter(path, width, height, self.info.frame_rate, settings.encoder,
                           gop=self._frames(GOP_SECONDS), background=True)

    def _finish_piece(self, writer: FrameWriter, start: int, end: int, generation: int) -> None:
        """Hand a piece to the closer thread; it becomes playable once its encoder has finished."""
        with self._condition:
            timeline = self._current.timeline
            current = generation == self._current.generation
            piece = Piece(start, end, writer.path)
            total = timeline.total
            if current:
                timeline.add(piece)
        if current and timeline.total != total:
            self.on_event("total", timeline.total)
        if not current:
            writer.abort()
            writer.path.unlink(missing_ok=True)
            return
        self._closer.submit(self._close_piece, writer, piece, generation)

    def _version_of(self, generation: int) -> Version | None:
        for version in (self._current, self._before):
            if version is not None and version.generation == generation:
                return version
        return None

    def _close_piece(self, writer: FrameWriter, piece: Piece, generation: int) -> None:
        try:
            writer.close()
        except Exception as error:  # noqa: BLE001 - reported to the window
            with self._condition:
                version = self._version_of(generation)
                if version is not None:
                    version.timeline.remove(piece)
                if version is self._current:
                    self._failed = True
            writer.path.unlink(missing_ok=True)
            log.emit(f"[render] error: {error}")
            self.on_event("error", error)
            return
        with self._condition:
            version = self._version_of(generation)
            current = version is self._current
            if version is not None:
                version.timeline.mark_ready(piece)  # a piece of "before" finishing late still counts
                self._condition.notify_all()
        if version is None:
            writer.path.unlink(missing_ok=True)
            return
        if not current:
            return
        with self._condition:
            stale, self._stale_files = self._stale_files, []
        self._delete(stale)
        self.on_event("piece", piece)
        if self.timeline.complete():
            self.on_event("complete")

    def _keep_thumbnail(self, frame: int, upscaled: np.ndarray, generation: int) -> None:
        if frame % self.thumbnail_stride:
            return
        import cv2
        height, width = upscaled.shape[:2]
        size = (max(1, int(round(width * THUMBNAIL_HEIGHT / height))), THUMBNAIL_HEIGHT)
        step = max(1, height // (2 * THUMBNAIL_HEIGHT))  # decimate first: an area resize of 4K costs 30 ms
        thumbnail = cv2.resize(np.ascontiguousarray(upscaled[::step, ::step, :3]), size, interpolation=cv2.INTER_AREA)
        with self._thumbnail_lock:
            if generation == self._current.generation:
                self._current.thumbnails[frame] = thumbnail

    def thumbnails(self) -> dict[int, np.ndarray]:
        """Small RGB pictures of upscaled frames of the current version (every `thumbnail_stride` frames)."""
        with self._thumbnail_lock:
            return dict(self._current.thumbnails)
