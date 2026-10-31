"""Sound for the player: one audio track decoded by ffmpeg to 48 kHz stereo PCM, played through waveOut.

waveOut (winmm, part of every Windows) is driven with ctypes: a feeder thread keeps a few short buffers queued,
`position()` reads how many samples the device has played, which is the clock the video follows. Pause and
resume act on the device at once; a seek resets it and restarts the decoder at the new time.
"""
from __future__ import annotations

import ctypes
import subprocess
import threading
from ctypes import wintypes
from pathlib import Path

from .media import MediaInfo, _popen, _StderrTail, tool_path

SAMPLE_RATE = 48000
CHANNELS = 2
BYTES_PER_SAMPLE_FRAME = 4
BLOCK_FRAMES = 1024
BUFFERS = 6
WAVE_MAPPER = 0xFFFFFFFF
CALLBACK_EVENT = 0x00050000
WHDR_DONE = 0x00000001
TIME_SAMPLES = 0x0002
WAIT_MILLISECONDS = 100


class WAVEFORMATEX(ctypes.Structure):
    _fields_ = [("wFormatTag", wintypes.WORD), ("nChannels", wintypes.WORD), ("nSamplesPerSec", wintypes.DWORD),
                ("nAvgBytesPerSec", wintypes.DWORD), ("nBlockAlign", wintypes.WORD),
                ("wBitsPerSample", wintypes.WORD), ("cbSize", wintypes.WORD)]


class WAVEHDR(ctypes.Structure):
    _fields_ = [("lpData", ctypes.c_void_p), ("dwBufferLength", wintypes.DWORD), ("dwBytesRecorded", wintypes.DWORD),
                ("dwUser", ctypes.c_size_t), ("dwFlags", wintypes.DWORD), ("dwLoops", wintypes.DWORD),
                ("lpNext", ctypes.c_void_p), ("reserved", ctypes.c_size_t)]


class _MMTIME_UNION(ctypes.Union):
    _fields_ = [("ms", wintypes.DWORD), ("sample", wintypes.DWORD), ("cb", wintypes.DWORD),
                ("ticks", wintypes.DWORD), ("smpte", ctypes.c_ubyte * 8), ("midi", wintypes.DWORD)]


class MMTIME(ctypes.Structure):
    _fields_ = [("wType", wintypes.UINT), ("u", _MMTIME_UNION)]


def _winmm():
    winmm = ctypes.WinDLL("winmm")
    handle = ctypes.c_void_p
    for name, arguments in (
            ("waveOutOpen", [ctypes.POINTER(handle), wintypes.UINT, ctypes.POINTER(WAVEFORMATEX), ctypes.c_size_t,
                             ctypes.c_size_t, wintypes.DWORD]),
            ("waveOutPrepareHeader", [handle, ctypes.POINTER(WAVEHDR), wintypes.UINT]),
            ("waveOutUnprepareHeader", [handle, ctypes.POINTER(WAVEHDR), wintypes.UINT]),
            ("waveOutWrite", [handle, ctypes.POINTER(WAVEHDR), wintypes.UINT]),
            ("waveOutPause", [handle]), ("waveOutRestart", [handle]), ("waveOutReset", [handle]),
            ("waveOutClose", [handle]),
            ("waveOutGetPosition", [handle, ctypes.POINTER(MMTIME), wintypes.UINT]),
            ("waveOutSetVolume", [handle, wintypes.DWORD]),
            ("waveOutGetErrorTextW", [wintypes.UINT, wintypes.LPWSTR, wintypes.UINT])):
        function = getattr(winmm, name)
        function.argtypes = arguments
        function.restype = wintypes.UINT
    return winmm


class WaveOut:
    """One waveOut device opened for 16-bit stereo at 48 kHz, with `BUFFERS` reusable buffers."""

    def __init__(self):
        self.winmm = _winmm()
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateEventW.restype = ctypes.c_void_p
        kernel32.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self.kernel32 = kernel32
        self.event = kernel32.CreateEventW(None, False, False, None)
        if not self.event:
            raise OSError(f"CreateEvent failed ({ctypes.GetLastError()})")
        self.handle = ctypes.c_void_p()
        self.block_bytes = BLOCK_FRAMES * BYTES_PER_SAMPLE_FRAME
        self.memory = [ctypes.create_string_buffer(self.block_bytes) for _ in range(BUFFERS)]
        self.headers = [WAVEHDR(lpData=ctypes.cast(memory, ctypes.c_void_p), dwBufferLength=self.block_bytes)
                        for memory in self.memory]
        audio_format = WAVEFORMATEX(wFormatTag=1, nChannels=CHANNELS, nSamplesPerSec=SAMPLE_RATE,
                                    nAvgBytesPerSec=SAMPLE_RATE * BYTES_PER_SAMPLE_FRAME,
                                    nBlockAlign=BYTES_PER_SAMPLE_FRAME, wBitsPerSample=16, cbSize=0)
        try:
            self._check(self.winmm.waveOutOpen(ctypes.byref(self.handle), WAVE_MAPPER, ctypes.byref(audio_format),
                                               self.event, 0, CALLBACK_EVENT), "open the audio output")
        except OSError:
            kernel32.CloseHandle(self.event)
            raise
        try:
            for header in self.headers:
                self._check(self.winmm.waveOutPrepareHeader(self.handle, ctypes.byref(header),
                                                            ctypes.sizeof(WAVEHDR)), "prepare an audio buffer")
                header.dwFlags |= WHDR_DONE  # free until first written
        except OSError:
            self.close()
            raise

    def _check(self, result: int, what: str) -> None:
        if result != 0:
            text = ctypes.create_unicode_buffer(256)
            self.winmm.waveOutGetErrorTextW(result, text, 256)
            raise OSError(f"cannot {what}: {text.value or f'error {result}'}")

    def free_buffer(self) -> int | None:
        """Index of a buffer the device has finished with, or None."""
        for index, header in enumerate(self.headers):
            if header.dwFlags & WHDR_DONE:
                return index
        return None

    def wait(self) -> None:
        """Sleep until the device finishes a buffer (or a short timeout)."""
        self.kernel32.WaitForSingleObject(self.event, WAIT_MILLISECONDS)

    def write(self, index: int, data: bytes) -> None:
        header = self.headers[index]
        ctypes.memmove(self.memory[index], data, len(data))
        header.dwBufferLength = len(data)
        header.dwFlags &= ~WHDR_DONE
        self._check(self.winmm.waveOutWrite(self.handle, ctypes.byref(header), ctypes.sizeof(WAVEHDR)),
                    "queue audio")

    def pause(self) -> None:
        self._check(self.winmm.waveOutPause(self.handle), "pause the audio")

    def resume(self) -> None:
        self._check(self.winmm.waveOutRestart(self.handle), "resume the audio")

    def reset(self) -> None:
        """Stop and drop every queued buffer; the position starts again at 0."""
        self._check(self.winmm.waveOutReset(self.handle), "reset the audio")

    def played_samples(self) -> int:
        position = MMTIME(wType=TIME_SAMPLES)
        self._check(self.winmm.waveOutGetPosition(self.handle, ctypes.byref(position), ctypes.sizeof(MMTIME)),
                    "read the audio position")
        if position.wType != TIME_SAMPLES:
            raise OSError("the audio device does not report its position in samples")
        return position.u.sample

    def set_volume(self, volume: float) -> None:
        level = int(round(max(0.0, min(1.0, volume)) * 0xFFFF))
        self._check(self.winmm.waveOutSetVolume(self.handle, level | level << 16), "set the volume")

    def close(self) -> None:
        self.winmm.waveOutReset(self.handle)
        for header in self.headers:
            self.winmm.waveOutUnprepareHeader(self.handle, ctypes.byref(header), ctypes.sizeof(WAVEHDR))
        result = self.winmm.waveOutClose(self.handle)
        self.kernel32.CloseHandle(self.event)
        self._check(result, "close the audio output")


def audio_command(path: Path, track: int, seconds: float) -> list[str]:
    """ffmpeg decoding audio track `track` of `path` from `seconds` to raw 48 kHz stereo 16-bit PCM."""
    command = [tool_path("ffmpeg"), "-v", "error", "-nostdin"]
    if seconds > 0:
        command += ["-ss", f"{seconds:.6f}"]
    return command + ["-i", str(path), "-map", f"0:a:{track}", "-vn", "-sn", "-dn", "-ac", str(CHANNELS),
                      "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]


def leading_silence(info: MediaInfo, track: int, seconds: float) -> int:
    """Sample frames of silence before the track's first sample when playback starts at `seconds`."""
    offset = info.audio[track].start_time - info.start_time - seconds
    return max(0, int(round(offset * SAMPLE_RATE)))


class AudioPlayer:
    """Plays one audio track of a file through waveOut from any time, on a feeder thread. Raises OSError at
    construction when the machine has no audio output."""

    def __init__(self, path: Path, info: MediaInfo, track: int = 0):
        self.path = path
        self.info = info
        self.track = track
        self.device = WaveOut()
        self.origin = 0.0
        self.error: Exception | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None

    def start(self, seconds: float, paused: bool = False) -> None:
        """Play from `seconds` (from the file start); `paused` queues the sound without starting it."""
        self._halt()
        with self._lock:
            self.device.reset()
            # waveOutReset keeps a paused device paused: set the state explicitly either way.
            if paused:
                self.device.pause()
            else:
                self.device.resume()
            self.origin = seconds
            self.error = None
            self._stop = threading.Event()
            silence = leading_silence(self.info, self.track, seconds)
            decode_from = max(0.0, seconds)
            self._process = _popen(audio_command(self.path, self.track, decode_from), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
            self._thread = threading.Thread(target=self._feed, args=(self._process, self._stop, silence),
                                            name="audio-feeder", daemon=True)
            self._thread.start()

    def _feed(self, process: subprocess.Popen, stop: threading.Event, silence: int) -> None:
        stderr = _StderrTail(process.stderr)
        pending = bytes(silence * BYTES_PER_SAMPLE_FRAME)
        try:
            while not stop.is_set():
                index = self.device.free_buffer()
                if index is None:
                    self.device.wait()
                    continue
                if len(pending) < self.device.block_bytes:
                    data = process.stdout.read(self.device.block_bytes - len(pending))
                    if not data:
                        if pending:
                            self.device.write(index, pending)
                        break
                    pending += data
                block, pending = pending[:self.device.block_bytes], pending[self.device.block_bytes:]
                with self._lock:
                    if stop.is_set():
                        break
                    self.device.write(index, block)
        except OSError as error:
            self.error = error
        finally:
            if process.poll() is None:
                process.kill()
            process.stdout.close()
            code = process.wait()
            if code != 0 and not stop.is_set() and self.error is None:
                self.error = RuntimeError(f"audio decoding failed: {stderr.text()}")

    def _halt(self) -> None:
        self._stop.set()
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
        if self._thread is not None:
            self._thread.join()
        self._thread = None
        self._process = None

    def position(self) -> float:
        """Seconds from the file start of the sound playing now."""
        with self._lock:
            return self.origin + self.device.played_samples() / SAMPLE_RATE

    def pause(self) -> None:
        with self._lock:
            self.device.pause()

    def resume(self) -> None:
        with self._lock:
            self.device.resume()

    def set_volume(self, volume: float) -> None:
        with self._lock:
            self.device.set_volume(volume)

    def close(self) -> None:
        self._halt()
        self.device.close()

