"""Upscales a video into pieces on a worker thread, following the playhead, and joins them into the output.

The window plays the upscaled video while it is being made: the renderer works where the playhead is
(preview mode keeps a stretch ahead of it upscaled, batch mode does the whole file starting there), writes
consecutive frames into piece files, and the player plays the finished pieces. A seek makes the renderer
finish its current piece and start over at the new position, with a few frames of pre-roll so FSR's history
and the optical flow are warm. Saving joins the pieces without re-encoding and adds the source's audio.

Each set of settings has its own version (pieces and timeline). The current settings' version is the one
saved; the view can show other versions (presets) beside it, and in preview mode the renderer upscales every
version shown, taking turns: the one with the least upscaled ahead of the playhead goes first, and a run
yields once it is ahead of another one by as much as that one has (at least FIRST_SWITCH_SECONDS, so each
shows its first frames soon after a seek, at most SWITCH_SECONDS, so turns stay long while playing). Versions
no longer shown are kept (up to KEPT_VERSIONS, the most recently used) so going back to them is instant.

Events reach the window through `on_event(kind, *values)`, called from the renderer's threads:
"piece" (a Piece of any version became playable), "live" (frame, source RGBA, upscaled RGBA of the current
version; twice a second while `want_live`), "status" (frame being rendered or None when idle, frames per
second), "total" (the real frame count), "complete" (every frame of the current version is ready), "error"
(an exception; rendering stops until the settings or what is shown change, or a batch starts).
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
# The player may wait for up to its BUFFER_MAX_SECONDS ahead of the frame after the playhead: the renderer must
# still have work while any version has less than that (tests/test_timeline.py checks it), or both wait forever.
PREVIEW_LOW_SECONDS = 12.0
PREVIEW_HIGH_SECONDS = 30.0
SWITCH_SECONDS = 2.0
FIRST_SWITCH_SECONDS = 0.5
KEPT_VERSIONS = 4
GOP_SECONDS = 2.0
THUMBNAIL_HEIGHT = 108
MAX_THUMBNAILS = 240
DIRECTORY_PREFIX = "fsr4-"
STALE_DIRECTORY_SECONDS = 24 * 3600
MIN_FREE_BYTES = 2 * 1024 ** 3

EventCallback = Callable[..., None]


UPSCALER_NAMES = {None: "FSR 4", "3.1.6": "FSR 3.1", "2.3.4": "FSR 2.3"}
MOTION_NAMES = {"ultrafast": "rough motion", "medium": "best motion"}


@dataclass(frozen=True)
class RenderSettings:
    """Everything the upscaled pieces depend on; each distinct value has its own version. `upscaler` names
    one of the FSR DLL's upscalers ("3.1.6", "2.3.4"); None is FSR 4."""
    output_size: tuple[int, int]
    flow: str = "dis"
    flow_preset: str = "fast"
    sharpness: float | None = None
    encoder: str = "amf"
    upscaler: str | None = None

    def describe(self) -> str:
        """A short name: the upscaler, plus the motion and the sharpening when they are not the defaults."""
        parts = [UPSCALER_NAMES.get(self.upscaler, f"FSR {self.upscaler}")]
        if self.flow == "zero":
            parts.append("no motion")
        elif self.flow_preset in MOTION_NAMES:
            parts.append(MOTION_NAMES[self.flow_preset])
        if self.sharpness is not None:
            parts.append(f"sharpening {self.sharpness:.2f}")
        return " · ".join(parts)


@dataclass(eq=False)
class Version:
    """The pieces upscaled with one set of settings, and their filmstrip pictures (current version only)."""
    settings: RenderSettings
    timeline: Timeline
    generation: int
    thumbnails: dict[int, np.ndarray] = field(default_factory=dict)
    used: float = 0.0


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
        self._condition = threading.Condition()
        self._generation = 0
        self._current = Version(settings, self._new_timeline(), 0, used=time.monotonic())
        self._versions: dict[RenderSettings, Version] = {settings: self._current}
        self._shown: list[Version] = []
        self._thumbnail_lock = threading.Lock()
        self.thumbnail_stride = max(1, int(round(self.rate)), (info.frame_count or 1) // MAX_THUMBNAILS)
        self.want_live = False
        self._target = target
        self._batch = False
        self._closing = False
        self._restart_run = False
        self._failed = False
        self._idle_reported = False
        self._gpu = None
        self._pipelines: dict[tuple, object] = {}
        self._pipeline_flows: dict[tuple, tuple[str, str]] = {}
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

    def timeline_of(self, settings: RenderSettings) -> Timeline:
        """The pieces of a version that is current or shown (see `show`)."""
        with self._condition:
            return self._versions[settings].timeline

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
        """Batch mode upscales the whole file with the current settings (Upscale); preview mode only what is
        ahead of the playhead, for every version shown. Starting a batch also retries after an error (the
        user may have freed disk space, for example)."""
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
        """New current settings: their version (kept from earlier, or new) becomes the one saved. The same
        settings again retry after an error."""
        with self._condition:
            if settings == self._current.settings and not self._failed:
                return
            self._failed = False
            self._idle_reported = False
            self._current = self._version(settings)
            dropped = self._evict()
            self._condition.notify_all()
        self._drop(dropped)

    def show(self, shown: list[RenderSettings]) -> None:
        """The versions the view shows (any number, duplicates ignored); preview mode upscales those, or the
        current version when the view shows none."""
        with self._condition:
            versions = [self._version(settings) for settings in dict.fromkeys(shown)]
            if versions != self._shown:
                self._failed = False
                self._idle_reported = False
            self._shown = versions
            dropped = self._evict()
            self._condition.notify_all()
        self._drop(dropped)

    def _version(self, settings: RenderSettings) -> Version:
        """The version of `settings`, made if needed; marks it as just used. Call with the lock held."""
        version = self._versions.get(settings)
        if version is None:
            self._generation += 1
            version = Version(settings, self._new_timeline(), self._generation)
            self._versions[settings] = version
        version.used = time.monotonic()
        return version

    def _evict(self) -> list[Version]:
        """Forget the versions beyond the KEPT_VERSIONS most recently used ones that are neither current nor
        shown; returns them. Call with the lock held."""
        kept = [self._current] + self._shown
        others = sorted((version for version in self._versions.values() if version not in kept),
                        key=lambda version: version.used, reverse=True)
        dropped = others[KEPT_VERSIONS:]
        for version in dropped:
            del self._versions[version.settings]
        return dropped

    def _drop(self, versions: list[Version]) -> None:
        self._delete([piece.path for version in versions for piece in version.timeline.clear() if piece.ready])

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
        """Join the ready pieces of the current version into `output` with the source's audio; returns the
        frames written. Everything must be ready unless `partial`, which saves the frames from the start up to
        the first gap."""
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
            while True:
                with self._condition:
                    job = self._wait_for_work()
                    if job is None:
                        break
                    version, start = job
                    needed = {self._pipeline_key(active.settings) for active in self._active()}
                try:
                    pipeline = self._pipeline_for(version.settings, needed)
                    self._run(start, version, pipeline)
                except Exception as error:  # noqa: BLE001 - reported, then the renderer waits for a change
                    with self._condition:
                        self._failed = True
                    log.emit(f"[render] error: {error}")
                    self.on_event("error", error)
        except Exception as error:  # noqa: BLE001 - the GPU could not start
            log.emit(f"[render] error: {error}")
            self.on_event("error", error)
        finally:
            for pipeline in self._pipelines.values():
                pipeline.destroy()
            if self._gpu is not None:
                self._gpu.destroy()

    def _active(self) -> list[Version]:
        """The versions to upscale: the current one in batch mode (it is what gets saved), else the ones the
        view shows, or the current one when it shows none. Call with the lock held."""
        if self._batch or not self._shown:
            return [self._current]
        return list(self._shown)

    def _next_job(self) -> tuple[Version, int] | None:
        """The version that needs frames most (the fewest ready ahead of the playhead) and where to start."""
        best = None
        for version in self._active():
            start = next_start(version.timeline, self._target, self._batch, self._frames(PREVIEW_LOW_SECONDS))
            if start is not None and (best is None or start - self._target < best[1] - self._target):
                best = (version, start)
        return best

    def _wait_for_work(self) -> tuple[Version, int] | None:
        """Block until there is something to render (returns the version and its first frame) or the renderer
        closes. Call with the lock held."""
        while not self._closing:
            job = None if self._failed else self._next_job()
            if job is not None:
                self._idle_reported = False
                return job
            if not self._idle_reported:
                self._idle_reported = True
                self.on_event("status", None, 0.0)
            self._condition.wait()
        return None

    def _frames(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.rate)))

    @staticmethod
    def _pipeline_key(settings: RenderSettings) -> tuple:
        return settings.output_size, settings.upscaler

    def _pipeline_for(self, settings: RenderSettings, needed: set[tuple]):
        """A pipeline set up for `settings`. One FSR context per size and upscaler is kept while an active
        version needs it (sharpening and motion are changed in place); the others are freed."""
        from .pipeline import Pipeline
        for key in [key for key in self._pipelines if key not in needed]:
            self._pipelines.pop(key).destroy()
            del self._pipeline_flows[key]
        key = self._pipeline_key(settings)
        size = (self.info.width, self.info.height)
        flow_settings = (settings.flow, settings.flow_preset)
        pipeline = self._pipelines.get(key)
        if pipeline is None:
            pipeline = Pipeline(self._gpu, size, settings.output_size, self.sdk_dir,
                                create_flow(settings.flow, *size, settings.flow_preset), sharpness=settings.sharpness,
                                frame_time_ms=self.info.frame_time_ms, provider=settings.upscaler)
            self._pipelines[key] = pipeline
        elif self._pipeline_flows[key] != flow_settings:
            pipeline.flow = create_flow(settings.flow, *size, settings.flow_preset)
        self._pipeline_flows[key] = flow_settings
        pipeline.sharpness = settings.sharpness
        return pipeline

    def _should_continue(self, version: Version, busy: tuple[int, int] | None, position: int) -> bool:
        """Whether the run of `version` about to render frame `position` goes on: not after a seek, a change of
        what is shown, the end of its stretch, or once another version shown needs frames more."""
        with self._condition:
            active = self._active()
            if self._closing or version not in active:
                return False
            if self._restart_run:
                self._restart_run = False
                return False
            if not keep_going(version.timeline, self._target, self._batch, self._frames(PREVIEW_HIGH_SECONDS), busy,
                              position):
                return False
            ahead = position - self._target
            for other in active:
                if other is version:
                    continue
                start = next_start(other.timeline, self._target, self._batch, self._frames(PREVIEW_LOW_SECONDS))
                if start is None:
                    continue
                other_ahead = start - self._target
                margin = min(self._frames(SWITCH_SECONDS), max(self._frames(FIRST_SWITCH_SECONDS), other_ahead))
                if ahead >= other_ahead + margin:
                    return False
            return True

    def _run(self, start: int, version: Version, pipeline) -> None:
        """Render `version` from `start` until its stretch is filled, a seek, a change, or far enough ahead."""
        from .pipeline import upscale_stream
        usage.lower_current_thread()
        timeline = version.timeline
        pipeline.reset()
        preroll_start = max(0, start - PREROLL_FRAMES)
        gap_end = timeline.gap_end(start)
        open_end = gap_end >= timeline.total
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
                        writer = self._open_piece(piece_start, version)
                    writer.write(upscaled)
                    position += 1
                    written += 1
                    current = version is self._current
                    if current:
                        self._keep_thumbnail(position - 1, upscaled, version)
                    now = time.perf_counter()
                    if current and self.want_live and now - last_live >= LIVE_INTERVAL:
                        self.on_event("live", position - 1, source_frame.copy(), upscaled.copy())
                        last_live = now
                    if now - last_status >= STATUS_INTERVAL:
                        self.on_event("status", position, written / max(now - run_started, 1e-6))
                        last_status = now
                    if position >= piece_end:
                        self._finish_piece(writer, piece_start, position, version)
                        writer = None
                    busy = (piece_start, position) if writer is not None else None
                    if ((not open_end and position >= gap_end)
                            or not self._should_continue(version, busy, position)):
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
                self._finish_piece(writer, piece_start, position, version)
        if exhausted:
            self._reached_end(start, position, timeline)

    def _reached_end(self, start: int, position: int, timeline: Timeline) -> None:
        """The decoder ran out of frames: that is the real end of the video (the probe's count is an estimate)."""
        if position == start and start > 0 and start < timeline.total - self._frames(1.0):
            raise RuntimeError(f"cannot read {self.source.name} from frame {start}")
        with self._condition:
            self._known_total = position
            versions = list(self._versions.values())
        for version in versions:
            version.timeline.set_total(position)
        self.on_event("total", self.timeline.total)
        if self.timeline.complete():
            self.on_event("complete")

    def _open_piece(self, start: int, version: Version) -> FrameWriter:
        free = shutil.disk_usage(self.directory).free
        if free < MIN_FREE_BYTES:
            raise RuntimeError(f"only {free / 1024 ** 3:.1f} GB free for the upscaled pieces in {self.directory.parent}; "
                               f"free some space (4K takes about 10 MB per second of video)")
        path = self.directory / f"piece_{version.generation}_{start:08d}.mp4"
        width, height = version.settings.output_size
        return FrameWriter(path, width, height, self.info.frame_rate, version.settings.encoder,
                           gop=self._frames(GOP_SECONDS), background=True)

    def _alive(self, version: Version) -> bool:
        """Whether `version` is still kept (not forgotten since). Call with the lock held."""
        return self._versions.get(version.settings) is version

    def _finish_piece(self, writer: FrameWriter, start: int, end: int, version: Version) -> None:
        """Hand a piece to the closer thread; it becomes playable once its encoder has finished."""
        with self._condition:
            alive = self._alive(version)
            piece = Piece(start, end, writer.path)
            total = version.timeline.total
            if alive:
                version.timeline.add(piece)
            grew = alive and version is self._current and version.timeline.total != total
        if grew:
            self.on_event("total", version.timeline.total)
        if not alive:
            writer.abort()
            writer.path.unlink(missing_ok=True)
            return
        self._closer.submit(self._close_piece, writer, piece, version)

    def _close_piece(self, writer: FrameWriter, piece: Piece, version: Version) -> None:
        try:
            writer.close()
        except Exception as error:  # noqa: BLE001 - reported to the window
            with self._condition:
                if self._alive(version):
                    version.timeline.remove(piece)
                    self._failed = True
            writer.path.unlink(missing_ok=True)
            log.emit(f"[render] error: {error}")
            self.on_event("error", error)
            return
        with self._condition:
            alive = self._alive(version)
            if alive:
                version.timeline.mark_ready(piece)
                self._condition.notify_all()
            stale, self._stale_files = self._stale_files, []
        self._delete(stale)
        if not alive:
            writer.path.unlink(missing_ok=True)
            return
        self.on_event("piece", piece)
        if version is self._current and version.timeline.complete():
            self.on_event("complete")

    def _keep_thumbnail(self, frame: int, upscaled: np.ndarray, version: Version) -> None:
        if frame % self.thumbnail_stride:
            return
        import cv2
        height, width = upscaled.shape[:2]
        size = (max(1, int(round(width * THUMBNAIL_HEIGHT / height))), THUMBNAIL_HEIGHT)
        step = max(1, height // (2 * THUMBNAIL_HEIGHT))  # decimate first: an area resize of 4K costs 30 ms
        thumbnail = cv2.resize(np.ascontiguousarray(upscaled[::step, ::step, :3]), size, interpolation=cv2.INTER_AREA)
        with self._thumbnail_lock:
            version.thumbnails[frame] = thumbnail

    def thumbnails(self) -> dict[int, np.ndarray]:
        """Small RGB pictures of upscaled frames of the current version (every `thumbnail_stride` frames)."""
        with self._thumbnail_lock:
            return dict(self._current.thumbnails)
