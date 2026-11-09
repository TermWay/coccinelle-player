"""Real-ESRGAN (the realesrgan-ncnn-vulkan program, installed apart from this tool): an upscaler that redraws.

A neural network draws each frame on its own at a larger size: crisp, with no motion vectors, no history and no
sharpening to set, and about 5 frames a second at 4K. The program only takes files and each run costs 0.34 s to
start, so frames go through it in chunks of PNG files in a temporary folder. `RealEsrganPipeline.feed` stands
between the frame reader and `upscale_stream`, which stays the one frame loop: it holds a frame back until its
chunk is done, and `process` then hands over the upscaled frame.

Two runs overlap, 27% faster than one at a time (a third adds nothing, measured). The first chunks are short so
that a picture comes soon after a seek; the longest is short too, because a seek waits for the chunk being read.

Still images have two more models (`STILL_ONLY`), a general one and one for drawings. They take seconds a picture, so
`RealEsrganPipeline.stop` ends one from another thread.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

from . import log, temporary, usage
from .flow import ZeroFlow
from .media import _popen, tool_path
from .pipeline import Timings

REAL_ESRGAN = "realesrgan"
REAL_ESRGAN_X4PLUS = "realesrgan-x4plus"
REAL_ESRGAN_ANIME = "realesrgan-x4plus-anime"
PROGRAM = "realesrgan-ncnn-vulkan.exe"
# Where the program goes, next to the exe or the package; PATH works too. Its `models` folder sits beside it.
FOLDER = Path("bin") / "realesrgan"
DOWNLOAD = "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0"
NOT_INSTALLED = (f"Real-ESRGAN is not installed: unzip realesrgan-ncnn-vulkan-20220424-windows.zip from {DOWNLOAD} "
                 f"into {FOLDER}")
# 9.5 s and 4.2 s a 1080p picture (measured): in a video, a seek would wait minutes for its chunk.
STILL_ONLY = frozenset({REAL_ESRGAN_X4PLUS, REAL_ESRGAN_ANIME})
STILL_ONLY_NOTE = "This Real-ESRGAN model takes seconds a picture: it is for still images. Use Real-ESRGAN for a video."
FIRST_CHUNKS = (4, 8)
CHUNK_FRAMES = 16
# The program's default: one thread to load, two to run, two to save. Under a lower cap it gets one of each.
PROGRAM_THREADS = 5
# Two chunks are in flight, and the frame the consumer holds must outlive the start of the next chunk.
SOURCE_BUFFERS = 2 * CHUNK_FRAMES + 2
FRAME_NAME = "%08d.png"
LOG_TAIL_CHARACTERS = 600
# Windows can hold a file for a moment after the program that wrote it was killed (seen once in about 60 stops).
REMOVE_ATTEMPTS = 10
REMOVE_WAIT_SECONDS = 0.1
STOP_POLL_SECONDS = 0.05


@dataclass(frozen=True)
class Model:
    """A network of the program: its name, the scales it draws at, its weights' file in the `models` folder, and the
    filter that takes its picture down to the output size (each is the one that measured best for that model)."""
    name: str
    scales: tuple[int, ...]
    weights: str
    resize: str


MODELS = {
    REAL_ESRGAN: Model("realesr-animevideov3", (2, 3, 4), "realesr-animevideov3-x2.bin", "lanczos"),
    REAL_ESRGAN_X4PLUS: Model("realesrgan-x4plus", (4,), "realesrgan-x4plus.bin", "area"),
    REAL_ESRGAN_ANIME: Model("realesrgan-x4plus-anime", (4,), "realesrgan-x4plus-anime.bin", "area"),
}


class Stopped(Exception):
    """`RealEsrganPipeline.stop` ended the picture being made."""


def find_program(upscaler: str = REAL_ESRGAN) -> Path | None:
    """Real-ESRGAN's program, with the model of `upscaler` beside it, or None when it is not installed."""
    places = [Path(__file__).resolve().parent.parent / FOLDER / PROGRAM]
    if getattr(sys, "frozen", False):
        places.insert(0, Path(sys.executable).parent / FOLDER / PROGRAM)
    found = shutil.which(PROGRAM)
    if found is not None:
        places.append(Path(found))
    weights = MODELS[upscaler].weights
    return next((path for path in places if path.is_file() and (path.parent / "models" / weights).is_file()), None)


def model_scale(render_size: tuple[int, int], output_size: tuple[int, int], model: Model) -> int:
    """The smallest scale of `model` that reaches `output_size` on both sides, else its largest."""
    needed = max(output / source for output, source in zip(output_size, render_size))
    return next((scale for scale in model.scales if scale >= needed - 1e-6), model.scales[-1])


def chunk_lengths() -> Iterator[int]:
    """Frames per run of the program: few at first, then `CHUNK_FRAMES`."""
    yield from FIRST_CHUNKS
    while True:
        yield CHUNK_FRAMES


def _frames_pattern(folder: Path) -> str:
    """ffmpeg's name for the numbered files of `folder`, whose path may hold a %."""
    return f"{str(folder).replace('%', '%%')}/{FRAME_NAME}"


def _remove(folder: Path) -> None:
    """Delete `folder`, trying again for a moment; what stays is logged, a temporary file is no reason to stop."""
    for attempt in range(REMOVE_ATTEMPTS):
        try:
            shutil.rmtree(folder)
            return
        except FileNotFoundError:
            return
        except OSError as error:
            if attempt == REMOVE_ATTEMPTS - 1:
                log.emit(f"[realesrgan] could not remove {folder}: {error}")
            else:
                time.sleep(REMOVE_WAIT_SECONDS)


def _ffmpeg_threads() -> list[str]:
    threads = usage.current().ffmpeg_threads
    return [] if threads is None else ["-threads", str(threads)]


@dataclass
class _Chunk:
    """Frames on their way through the program."""
    sources: list[np.ndarray]
    folder: Path
    program: subprocess.Popen
    log_path: Path
    log_file: BinaryIO

    def failure(self, what: str) -> RuntimeError:
        return RuntimeError(f"{what}: {self.log_path.read_text('utf-8', 'replace')[-LOG_TAIL_CHARACTERS:].strip()}")

    def stop(self) -> None:
        """End the program, whether it is done or not, and let go of its files."""
        self.program.kill()
        self.program.wait()
        self.log_file.close()


class RealEsrganPipeline:
    """Stands in for `Pipeline` with Real-ESRGAN; see the module's text."""
    keeps_history = False
    encoder_scale = None

    def __init__(self, render_size: tuple[int, int], output_size: tuple[int, int], upscaler: str = REAL_ESRGAN):
        program = find_program(upscaler)
        if program is None:
            raise FileNotFoundError(NOT_INSTALLED)
        self.program = program
        self.render_size = render_size
        self.output_size = output_size
        self.sharpness = None
        self.model = MODELS[upscaler]
        self.scale = model_scale(render_size, output_size, self.model)
        self.stop = threading.Event()  # set from another thread, it ends the run with `Stopped`
        self.flow = ZeroFlow(*render_size)
        self.timings = Timings()
        self.upscaled: deque[np.ndarray] = deque()
        self.sources: list[np.ndarray] = []
        self.next_source = 0
        self.buffers: list[np.ndarray] = []

    def reset(self) -> None:
        """Nothing to do: the model keeps no history."""

    def feed(self, frames: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        """Yield `frames`, each once its upscaled frame is waiting for `process`. Closing this ends what is running
        and removes its files."""
        self.upscaled.clear()
        folder = Path(tempfile.mkdtemp(prefix="fsr4-realesrgan-", dir=temporary.folder()))
        chunks: deque[_Chunk] = deque()
        try:
            for number, length in enumerate(chunk_lengths()):
                chunk = self._start_chunk(islice(frames, length), folder / str(number))
                if chunk is not None:
                    chunks.append(chunk)
                if chunks and (chunk is None or len(chunks) == 2):
                    yield from self._read_chunk(chunks[0])
                    finished = chunks.popleft()
                    finished.stop()
                    _remove(finished.folder)
                if chunk is None and not chunks:
                    return
        finally:
            for chunk in chunks:
                chunk.stop()
            _remove(folder)

    def _keep_source(self, frame: np.ndarray) -> np.ndarray:
        """`frame` copied into the next buffer of the ring; the reader will overwrite its own."""
        index = self.next_source
        self.next_source = (index + 1) % SOURCE_BUFFERS
        if index == len(self.sources):
            self.sources.append(np.empty(frame.shape, np.uint8))
        np.copyto(self.sources[index], frame)
        return self.sources[index]

    def _start_chunk(self, frames: Iterator[np.ndarray], folder: Path) -> _Chunk | None:
        """Write `frames` as PNG files and start the program on them; None when there are no frames."""
        first = next(frames, None)
        if first is None:
            return None
        pictures, output = folder / "in", folder / "out"
        pictures.mkdir(parents=True)
        output.mkdir()
        log_path = folder / "log.txt"
        log_file = open(log_path, "wb")
        try:
            sources = self._write_pictures(first, frames, pictures, log_file)
            program = _popen(self._program_command(pictures, output), background=True, stdin=subprocess.DEVNULL,
                             stdout=log_file, stderr=log_file)
        except BaseException:
            log_file.close()
            raise
        return _Chunk(sources, folder, program, log_path, log_file)

    def _write_pictures(self, first: np.ndarray, frames: Iterator[np.ndarray], folder: Path,
                        log_file: BinaryIO) -> list[np.ndarray]:
        """RGB PNG files of `first` and `frames` (no alpha: the program would upscale it too), through one ffmpeg."""
        width, height = self.render_size
        command = [tool_path("ffmpeg"), "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgba",
                   "-s", f"{width}x{height}", "-i", "-", *_ffmpeg_threads(), "-pix_fmt", "rgb24", "-c:v", "png",
                   "-compression_level", "1", _frames_pattern(folder)]
        writer = _popen(command, background=True, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log_file)
        sources = []
        try:
            for frame in chain((first,), frames):
                source = self._keep_source(frame)
                sources.append(source)
                writer.stdin.write(memoryview(source).cast("B"))
            writer.stdin.close()
        except OSError as error:
            writer.kill()
            writer.wait()
            raise RuntimeError(f"ffmpeg stopped writing the frames for Real-ESRGAN: {error}") from error
        except BaseException:
            writer.kill()
            writer.wait()
            raise
        if writer.wait() != 0:
            raise RuntimeError(f"ffmpeg could not write the frames for Real-ESRGAN ({writer.returncode})")
        return sources

    def _program_command(self, pictures: Path, output: Path) -> list[str]:
        threads = usage.current().ffmpeg_threads
        few = threads is not None and threads < PROGRAM_THREADS
        return [str(self.program), "-i", str(pictures), "-o", str(output), "-n", self.model.name, "-s", str(self.scale),
                "-m", str(self.program.parent / "models"), *(["-j", "1:1:1"] if few else [])]

    def _wait(self, chunk: _Chunk) -> int:
        """The exit code of `chunk`'s program; raises `Stopped` when `stop` comes first."""
        while True:
            try:
                return chunk.program.wait(STOP_POLL_SECONDS)
            except subprocess.TimeoutExpired:
                if self.stop.is_set():
                    raise Stopped from None

    def _read_chunk(self, chunk: _Chunk) -> Iterator[np.ndarray]:
        """Wait for `chunk` and yield its sources, each with its upscaled frame read back, at the output size."""
        if self._wait(chunk) != 0:
            raise chunk.failure(f"Real-ESRGAN failed ({chunk.program.returncode})")
        width, height = self.output_size
        source_width, source_height = self.render_size
        resize = []
        if (source_width * self.scale, source_height * self.scale) != self.output_size:
            resize = ["-vf", f"scale={width}:{height}:flags={self.model.resize}+accurate_rnd+full_chroma_int+full_chroma_inp"]
        command = [tool_path("ffmpeg"), "-v", "error", "-nostdin", *_ffmpeg_threads(), "-i",
                   _frames_pattern(chunk.folder / "out"), *resize, "-f", "rawvideo", "-pix_fmt", "rgba", "-"]
        if not self.buffers:
            self.buffers = [np.empty((height, width, 4), np.uint8) for _ in range(2)]
        reader = _popen(command, background=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=chunk.log_file)
        try:
            for source in chunk.sources:
                self.buffers.reverse()  # the consumer may still hold the other one: `upscale_stream` reads ahead
                upscaled = self.buffers[0]
                if reader.stdout.readinto(memoryview(upscaled).cast("B")) != upscaled.nbytes:
                    raise chunk.failure("Real-ESRGAN wrote fewer frames than it was given")
                self.upscaled.append(upscaled)
                yield source
        finally:
            reader.kill()
            reader.stdout.close()
            reader.wait()

    def process(self, rgba: np.ndarray, motion: np.ndarray | None = None) -> np.ndarray:
        """The upscaled frame of `rgba`, which came from `feed`: frames come back in the order it gave them."""
        self.timings.frames += 1
        return self.upscaled.popleft()

    def picture(self, rgba: np.ndarray) -> np.ndarray:
        """Upscale one picture; it comes back opaque."""
        feed = self.feed(iter((rgba,)))
        try:
            next(feed)
            return self.process(rgba).copy()
        finally:
            feed.close()

    def destroy(self) -> None:
        """Nothing to free: `feed` removes what it made."""
