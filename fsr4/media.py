"""Image and video input/output through ffmpeg and ffprobe subprocesses, as raw RGBA frames."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".exr", ".gif"}

ENCODERS = {
    "amf": ["-c:v", "hevc_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", "18", "-qp_p", "20", "-tag:v", "hvc1"],
    "h264_amf": ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", "18", "-qp_p", "20"],
    "x264": ["-c:v", "libx264", "-crf", "17", "-preset", "medium"],
    "x265": ["-c:v", "libx265", "-crf", "18", "-preset", "medium", "-tag:v", "hvc1"],
    "png": ["-c:v", "png"],
}


def tool_path(name: str) -> str:
    """Path of ffmpeg or ffprobe: next to the frozen exe, else on PATH."""
    if getattr(sys, "frozen", False):
        bundled = Path(sys.executable).parent / f"{name}.exe"
        if bundled.is_file():
            return str(bundled)
    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(f"{name} not found on PATH; install ffmpeg (https://ffmpeg.org) or put {name}.exe next to the exe")
    return found


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


@dataclass
class MediaInfo:
    """What ffprobe reports about the first video stream."""
    width: int
    height: int
    frame_rate: Fraction
    frame_count: int | None
    duration: float | None
    has_audio: bool
    codec: str

    @property
    def frame_time_ms(self) -> float:
        return float(1000 / self.frame_rate) if self.frame_rate else 1000 / 30


def probe(path: Path) -> MediaInfo:
    command = [
        tool_path("ffprobe"), "-v", "error", "-show_entries",
        "stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {completed.stderr.strip()}")
    data = json.loads(completed.stdout)
    streams = data.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"no video stream in {path}")
    rate_text = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/0"
    numerator, denominator = (int(part) for part in rate_text.split("/"))
    frame_rate = Fraction(numerator, denominator) if denominator and numerator else Fraction(0)
    frame_count = int(video["nb_frames"]) if video.get("nb_frames", "N/A").isdigit() else None
    duration_text = data.get("format", {}).get("duration")
    duration = float(duration_text) if duration_text else None
    if frame_count is None and duration and frame_rate:
        frame_count = int(round(duration * frame_rate))
    return MediaInfo(
        width=int(video["width"]), height=int(video["height"]), frame_rate=frame_rate, frame_count=frame_count,
        duration=duration, has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
        codec=video.get("codec_name", "?"))


def _popen(command: list[str], **kwargs) -> subprocess.Popen:
    creation = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen(command, creationflags=creation, **kwargs)


class FrameReader:
    """Yields RGBA uint8 frames of shape (height, width, 4) decoded by ffmpeg.

    Frames alternate between two buffers, so the frame from the previous iteration stays valid while the
    next one is being read (the flow worker needs that); anything older is overwritten.
    """

    def __init__(self, path: Path, info: MediaInfo, max_frames: int | None = None):
        self.info = info
        self.frame_bytes = info.width * info.height * 4
        command = [tool_path("ffmpeg"), "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0",
                   "-f", "rawvideo", "-pix_fmt", "rgba"]
        if max_frames is not None:
            command += ["-frames:v", str(max_frames)]
        command.append("-")
        self.process = _popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._buffers = [bytearray(self.frame_bytes), bytearray(self.frame_bytes)]
        self._next_buffer = 0

    def __iter__(self):
        while True:
            frame = self._read_frame()
            if frame is None:
                break
            yield frame
        self.close()

    def _read_frame(self):
        buffer = self._buffers[self._next_buffer]
        self._next_buffer ^= 1
        view = memoryview(buffer)
        filled = 0
        while filled < self.frame_bytes:
            count = self.process.stdout.readinto(view[filled:])
            if not count:
                break
            filled += count
        if filled == 0:
            return None
        if filled != self.frame_bytes:
            raise RuntimeError(f"ffmpeg stopped mid-frame ({filled} of {self.frame_bytes} bytes)")
        return np.frombuffer(buffer, dtype=np.uint8).reshape(self.info.height, self.info.width, 4)

    def close(self) -> None:
        self.process.stdout.close()
        stderr = self.process.stderr.read().decode("utf-8", "replace")
        self.process.stderr.close()
        code = self.process.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg decode failed ({code}): {stderr.strip()}")


class FrameWriter:
    """Feeds RGBA uint8 frames to ffmpeg, which encodes them (and copies the source audio when asked)."""

    def __init__(self, path: Path, width: int, height: int, frame_rate: Fraction, encoder: str,
                 audio_source: Path | None = None):
        if encoder not in ENCODERS:
            raise ValueError(f"unknown encoder {encoder}; choose from {', '.join(ENCODERS)}")
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 4
        rate = f"{frame_rate.numerator}/{frame_rate.denominator}" if frame_rate else "30"
        command = [tool_path("ffmpeg"), "-v", "error", "-nostdin", "-y",
                   "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r", rate, "-i", "-"]
        if audio_source is not None:
            command += ["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy"]
        command += ENCODERS[encoder]
        if encoder != "png":
            command += ["-pix_fmt", "yuv420p"]
        command.append(str(path))
        self.process = _popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.nbytes != self.frame_bytes:
            raise ValueError(f"frame has {frame.nbytes} bytes, writer expects {self.frame_bytes}")
        try:
            self.process.stdin.write(memoryview(frame).cast("B"))
        except BrokenPipeError as error:
            raise RuntimeError(f"ffmpeg encoder stopped: {self._stderr()}") from error

    def _stderr(self) -> str:
        text = self.process.stderr.read().decode("utf-8", "replace").strip()
        self.process.stderr.close()
        return text

    def close(self) -> None:
        """Finish the stream and wait for the encoder; raises when ffmpeg failed."""
        if not self.process.stdin.closed:
            self.process.stdin.close()
        stderr = self._stderr() if not self.process.stderr.closed else ""
        code = self.process.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg encode failed ({code}): {stderr}")

    def abort(self) -> None:
        """Kill the encoder without waiting for it (after an error elsewhere)."""
        self.process.kill()
        self.process.stdin.close()
        self.process.stderr.close()
        self.process.wait()


def read_image(path: Path) -> np.ndarray:
    """Decode an image file to an RGBA uint8 array of shape (height, width, 4)."""
    info = probe(path)
    frames = [frame.copy() for frame in FrameReader(path, info, max_frames=1)]
    if not frames:
        raise RuntimeError(f"ffmpeg decoded no frame from {path}")
    return frames[0]


def write_image(path: Path, frame: np.ndarray) -> None:
    """Encode an RGBA uint8 array to an image file (format from the suffix)."""
    height, width = frame.shape[:2]
    command = [tool_path("ffmpeg"), "-v", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-i", "-",
               "-frames:v", "1", "-pix_fmt", "rgb24", str(path)]
    completed = subprocess.run(command, input=memoryview(np.ascontiguousarray(frame)).cast("B"),
                               capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg could not write {path}: {completed.stderr.decode('utf-8', 'replace').strip()}")
