"""How much of the processor background work may take: a number of threads and a priority.

Upscaling keeps the CPU busy: optical flow in OpenCV, ffmpeg decoding and, with x264/x265, encoding. The thread
count caps each of them (`-threads` for ffmpeg, OpenCV's pool), and the priority decides who wins when other
programs want the processor too: normal, low (below normal) or background (idle: only time nobody else uses).
Playback is never limited. `apply` sets both for the whole process; new processes and runs pick them up.

OpenCV's optical flow gains nothing past about 8 threads (measured on a 16-core CPU: 4 threads 44.5 fps,
8 threads 47 fps, 32 threads 39 fps for 1080p to 4K), so it never gets more than that; the encoders and decoders
use the whole count.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

from . import log

NORMAL_PRIORITY_CLASS = 0x20
BELOW_NORMAL_PRIORITY_CLASS = 0x4000
IDLE_PRIORITY_CLASS = 0x40
OPENCV_THREADS_MAX = 8
CPU_THREADS = os.cpu_count() or 4


@dataclass(frozen=True)
class Priority:
    name: str
    label: str
    process_class: int
    thread_priority: int
    description: str


PRIORITIES = {
    "normal": Priority("normal", "Normal", NORMAL_PRIORITY_CLASS, 0,
                       "Same priority as other programs: the quickest upscale, but games and other work can "
                       "stutter, mostly with the CPU encoders."),
    "low": Priority("low", "Low", BELOW_NORMAL_PRIORITY_CLASS, -1,
                    "Below other programs: they and this window stay smooth, and the upscale takes what is left. "
                    "Barely slower on an idle machine."),
    "idle": Priority("idle", "Background", IDLE_PRIORITY_CLASS, -2,
                     "Only processor time nobody else wants: unnoticeable while you work or play, and the upscale "
                     "slows down whenever the machine is busy."),
}


@dataclass(frozen=True)
class Usage:
    """Threads background work may use (1 to all of them) and its priority (a key of PRIORITIES)."""
    threads: int
    priority: str

    @property
    def ffmpeg_threads(self) -> int | None:
        """The `-threads` value for background ffmpeg processes; None leaves ffmpeg its own choice (all)."""
        return None if self.threads >= CPU_THREADS else self.threads

    @property
    def opencv_threads(self) -> int:
        return max(1, min(OPENCV_THREADS_MAX, self.threads))


DEFAULT = Usage(max(1, CPU_THREADS // 2), "low")

_current = DEFAULT


def current() -> Usage:
    return _current


def apply(threads: int, priority: str) -> Usage:
    """Use `threads` (clamped to 1..all) at `priority` for background work started from now on."""
    global _current
    if priority not in PRIORITIES:
        raise ValueError(f"unknown priority {priority}; choose from {', '.join(PRIORITIES)}")
    _current = Usage(max(1, min(CPU_THREADS, int(threads))), priority)
    try:
        import cv2
        cv2.setNumThreads(_current.opencv_threads)
    except ImportError:
        pass  # without OpenCV there is no optical flow to throttle
    return _current


def background_creation_flags() -> int:
    """`creationflags` priority bits for an ffmpeg process doing background work."""
    if os.name != "nt":
        return 0
    return PRIORITIES[_current.priority].process_class


def lower_current_thread() -> None:
    """Give the calling (background worker) thread the current priority."""
    if os.name != "nt":
        return
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentThread.restype = ctypes.c_void_p
    kernel32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
    if not kernel32.SetThreadPriority(kernel32.GetCurrentThread(), PRIORITIES[_current.priority].thread_priority):
        log.emit(f"[usage] could not lower a worker thread's priority (error {ctypes.GetLastError()})")
