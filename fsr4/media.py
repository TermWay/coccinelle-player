"""Image and video input/output through ffmpeg and ffprobe subprocesses, as raw RGBA frames.

Decoding picks the real video stream (not a cover picture), converts YUV to RGB with the source's colour
matrix (BT.709 for untagged HD, as players assume), and forces a constant frame rate so the encoder side
can be fed at a fixed rate in sync with the source audio. Encoding converts back with BT.709 for HD output
and tags the file, copies every audio track an MP4 can hold and re-encodes the others (WMA, PCM, Vorbis,
Opus, FLAC, ...) to AAC.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".exr", ".gif")
VIDEO_SUFFIXES = (".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".wmv", ".asf", ".flv", ".f4v", ".mpg", ".mpeg",
                  ".m2v", ".vob", ".ts", ".m2ts", ".mts", ".3gp", ".3g2", ".ogv", ".mxf", ".dv", ".rm", ".rmvb",
                  ".divx", ".y4m")

ENCODERS = {
    "amf": ["-c:v", "hevc_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", "18", "-qp_p", "20", "-tag:v", "hvc1"],
    "h264_amf": ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", "18", "-qp_p", "20"],
    "x264": ["-c:v", "libx264", "-crf", "17", "-preset", "medium"],
    "x265": ["-c:v", "libx265", "-crf", "18", "-preset", "medium", "-tag:v", "hvc1"],
    "png": ["-c:v", "png"],
}

# Audio codecs an MP4/MOV file holds and every common player decodes: these are copied, the rest re-encoded.
MP4_AUDIO_COPY = frozenset({"aac", "mp3", "ac3", "eac3", "alac"})
COPY_ANY_AUDIO_SUFFIXES = frozenset({".mkv"})
AAC_KBPS_PER_CHANNEL = 96
HD_HEIGHT = 720
STDERR_LINES = 40
_CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


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


def _parse_rate(text: str | None) -> Fraction:
    if not text or "/" not in text:
        return Fraction(0)
    numerator, denominator = (int(part) for part in text.split("/"))
    return Fraction(numerator, denominator) if numerator > 0 and denominator > 0 else Fraction(0)


def guess_frame_rate(real_rate: Fraction, average_rate: Fraction) -> Fraction:
    """ffmpeg's own choice (av_guess_frame_rate): the real base rate, unless it is a container time base."""
    if real_rate and average_rate and average_rate < 70 and real_rate > 210:
        return average_rate
    return real_rate or average_rate or Fraction(30)


@dataclass
class AudioStream:
    """One audio track; `position` is its rank among the file's audio streams (`-map 1:a:<position>`)."""
    position: int
    codec: str
    channels: int


@dataclass
class MediaInfo:
    """What ffprobe reports about the video stream that gets upscaled, and the audio around it."""
    width: int
    height: int
    frame_rate: Fraction
    frame_count: int | None
    duration: float | None
    codec: str
    stream_index: int = 0
    color_matrix: str | None = None
    rotation: int = 0
    audio: list[AudioStream] = field(default_factory=list)
    container: str = ""

    @property
    def has_audio(self) -> bool:
        return bool(self.audio)

    @property
    def frame_time_ms(self) -> float:
        return float(1000 / self.frame_rate) if self.frame_rate else 1000 / 30

    @property
    def needs_hd_matrix(self) -> bool:
        """Untagged HD YUV decodes with BT.709, like players do (ffmpeg would assume BT.601)."""
        return self.color_matrix is None and self.height >= HD_HEIGHT and self.width >= HD_HEIGHT


def probe(path: Path) -> MediaInfo:
    command = [
        tool_path("ffprobe"), "-v", "error", "-show_entries",
        "stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,channels,"
        "color_space,pix_fmt:stream_disposition=attached_pic:stream_side_data=rotation:format=duration,format_name",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               creationflags=_CREATE_NO_WINDOW)
    if completed.returncode != 0:
        raise RuntimeError(f"not a video or image ffmpeg can read ({ffmpeg_reason(completed.stderr, path)})")
    return parse_probe(json.loads(completed.stdout), path)


def ffmpeg_reason(stderr: str, path: Path | None = None) -> str:
    """The last line of ffmpeg's error output without its '[demuxer @ 0x...]' and file name prefixes."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return "no error message"
    reason = re.sub(r"^\[[^\]]*@ [0-9a-fA-Fx]+\]\s*", "", lines[-1])
    if path is not None:
        reason = reason.replace(f"{path}: ", "")
    return reason


def parse_probe(data: dict, path: Path) -> MediaInfo:
    """Turn ffprobe's JSON into a MediaInfo: the first video stream that is not a cover picture."""
    streams = data.get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    video = next((stream for stream in videos if not stream.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise RuntimeError(f"no video stream in {path}" + (" (only a cover picture)" if videos else ""))
    frame_rate = guess_frame_rate(_parse_rate(video.get("r_frame_rate")), _parse_rate(video.get("avg_frame_rate")))
    duration_text = video.get("duration") or data.get("format", {}).get("duration")
    duration = float(duration_text) if duration_text else None
    count_text = video.get("nb_frames", "N/A")
    frame_count = int(count_text) if count_text.isdigit() else None
    if frame_count is None and duration and frame_rate:
        frame_count = max(1, int(round(duration * frame_rate)))
    rotation = 0
    for side_data in video.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = int(side_data["rotation"]) % 360
    width, height = int(video["width"]), int(video["height"])
    if rotation in (90, 270):
        # ffmpeg rotates the decoded frames upright, so the RGBA frames have the swapped size.
        width, height = height, width
    color_space = video.get("color_space")
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    return MediaInfo(
        width=width, height=height, frame_rate=frame_rate, frame_count=frame_count, duration=duration,
        codec=video.get("codec_name", "?"), stream_index=int(video["index"]),
        color_matrix=color_space if color_space and color_space not in ("unknown", "reserved") else None,
        rotation=rotation,
        audio=[AudioStream(position, stream.get("codec_name", "?"), int(stream.get("channels") or 2))
               for position, stream in enumerate(audio_streams)],
        container=data.get("format", {}).get("format_name", ""))


def _popen(command: list[str], **kwargs) -> subprocess.Popen:
    return subprocess.Popen(command, creationflags=_CREATE_NO_WINDOW, **kwargs)


class _StderrTail:
    """Drains a process's stderr on a thread (a full pipe would block ffmpeg) and keeps the last lines."""

    def __init__(self, stream):
        self.lines: deque[str] = deque(maxlen=STDERR_LINES)
        self.thread = threading.Thread(target=self._drain, args=(stream,), daemon=True)
        self.thread.start()

    def _drain(self, stream) -> None:
        for raw in iter(stream.readline, b""):
            self.lines.append(raw.decode("utf-8", "replace").rstrip())
        stream.close()

    def text(self) -> str:
        self.thread.join(timeout=5)
        return "\n".join(self.lines)


def rate_text(rate: Fraction) -> str:
    return f"{rate.numerator}/{rate.denominator}"


class FrameReader:
    """Yields RGBA uint8 frames of shape (height, width, 4) decoded by ffmpeg.

    Frames alternate between two buffers, so the frame from the previous iteration stays valid while the
    next one is being read (the flow worker needs that); anything older is overwritten. Video comes out at
    the constant rate `info.frame_rate` (duplicating or dropping frames of variable-rate sources), starting
    at the file's start time so it stays in sync with the audio.
    """

    def __init__(self, path: Path, info: MediaInfo, max_frames: int | None = None,
                 start_seconds: float | None = None, constant_rate: bool = True):
        self.info = info
        self.frame_bytes = info.width * info.height * 4
        command = [tool_path("ffmpeg"), "-v", "error", "-nostdin"]
        if start_seconds:
            command += ["-ss", f"{start_seconds:.3f}"]
        command += ["-i", str(path), "-map", f"0:{info.stream_index}", "-an", "-sn", "-dn"]
        filters = ["scale=in_color_matrix=bt709"] if info.needs_hd_matrix else []
        if filters:
            command += ["-vf", ",".join(filters + ["format=rgba"])]
        if constant_rate and info.frame_rate:
            command += ["-fps_mode", "cfr", "-r", rate_text(info.frame_rate)]
        if max_frames is not None:
            command += ["-frames:v", str(max_frames)]
        command += ["-f", "rawvideo", "-pix_fmt", "rgba", "-"]
        self.process = _popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._stderr = _StderrTail(self.process.stderr)
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
            raise RuntimeError(f"ffmpeg stopped mid-frame ({filled} of {self.frame_bytes} bytes): {self._stderr.text()}")
        return np.frombuffer(buffer, dtype=np.uint8).reshape(self.info.height, self.info.width, 4)

    def close(self) -> None:
        """Wait for the decoder after the last frame; raises when ffmpeg failed."""
        self.process.stdout.close()
        code = self.process.wait()
        stderr = self._stderr.text()
        if code != 0:
            raise RuntimeError(f"ffmpeg decode failed ({code}): {stderr.strip()}")

    def stop(self) -> None:
        """Kill the decoder before the end of the file (cancel) and release its pipes."""
        self.process.kill()
        self.process.stdout.close()
        self.process.wait()
        self._stderr.text()


def audio_arguments(audio: list[AudioStream], output: Path) -> list[str]:
    """Map every source audio track (input 1) and copy or re-encode it to suit the output container."""
    copy_any = output.suffix.lower() in COPY_ANY_AUDIO_SUFFIXES
    arguments: list[str] = []
    for stream in audio:
        arguments += ["-map", f"1:a:{stream.position}"]
        if copy_any or stream.codec in MP4_AUDIO_COPY:
            arguments += [f"-c:a:{stream.position}", "copy"]
        else:
            bitrate = AAC_KBPS_PER_CHANNEL * max(2, stream.channels)
            arguments += [f"-c:a:{stream.position}", "aac", f"-b:a:{stream.position}", f"{bitrate}k"]
    return arguments


def describe_audio(audio: list[AudioStream], output: Path) -> list[str]:
    """One line per audio track saying whether it is copied or re-encoded."""
    copy_any = output.suffix.lower() in COPY_ANY_AUDIO_SUFFIXES
    lines = []
    for stream in audio:
        if copy_any or stream.codec in MP4_AUDIO_COPY:
            lines.append(f"audio track {stream.position + 1}: {stream.codec}, copied")
        else:
            bitrate = AAC_KBPS_PER_CHANNEL * max(2, stream.channels)
            lines.append(f"audio track {stream.position + 1}: {stream.codec} cannot go in "
                         f"{output.suffix or 'this file'}, re-encoded to AAC {bitrate} kb/s")
    return lines


def color_arguments(height: int) -> list[str]:
    """RGB to YUV with the matrix players assume for this size, and the matching tags."""
    if height >= HD_HEIGHT:
        matrix, tag = "bt709", "bt709"
    else:
        matrix, tag = "bt601", "smpte170m"
    # ffmpeg takes the encoder's colour tags from the frames, so they are set on the frames (setparams).
    return ["-vf", f"scale=out_color_matrix={matrix}:out_range=tv,format=yuv420p,"
                   f"setparams=colorspace={tag}:color_primaries={tag}:color_trc={tag}:range=tv",
            "-colorspace", tag, "-color_primaries", tag, "-color_trc", tag, "-color_range", "tv"]


class FrameWriter:
    """Feeds RGBA uint8 frames to ffmpeg, which encodes them and adds the source's audio tracks."""

    def __init__(self, path: Path, width: int, height: int, frame_rate: Fraction, encoder: str,
                 audio_source: Path | None = None, audio: list[AudioStream] | None = None,
                 audio_seconds: float | None = None):
        if encoder not in ENCODERS:
            raise ValueError(f"unknown encoder {encoder}; choose from {', '.join(ENCODERS)}")
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 4
        rate = rate_text(frame_rate) if frame_rate else "30"
        command = [tool_path("ffmpeg"), "-v", "error", "-nostdin", "-y",
                   "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r", rate, "-i", "-"]
        tracks = audio or []
        if audio_source is not None and tracks:
            if audio_seconds is not None:
                command += ["-t", f"{audio_seconds:.3f}"]
            command += ["-i", str(audio_source), "-map", "0:v:0"] + audio_arguments(tracks, path)
        command += ENCODERS[encoder]
        if encoder != "png":
            command += color_arguments(height)
        command.append(str(path))
        self.process = _popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._stderr = _StderrTail(self.process.stderr)

    def write(self, frame: np.ndarray) -> None:
        if frame.nbytes != self.frame_bytes:
            raise ValueError(f"frame has {frame.nbytes} bytes, writer expects {self.frame_bytes}")
        try:
            self.process.stdin.write(memoryview(frame).cast("B"))
        except (BrokenPipeError, OSError) as error:
            self.process.wait()
            raise RuntimeError(f"ffmpeg encoder stopped: {self._stderr.text()}") from error

    def close(self) -> None:
        """Finish the stream and wait for the encoder; raises when ffmpeg failed."""
        if not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except OSError:
                pass  # the encoder already exited; its exit code and stderr below say why
        code = self.process.wait()
        stderr = self._stderr.text()
        if code != 0:
            raise RuntimeError(f"ffmpeg encode failed ({code}): {stderr}")

    def abort(self) -> None:
        """Kill the encoder without waiting for it (after an error elsewhere)."""
        self.process.kill()
        try:
            self.process.stdin.close()
        except OSError:
            pass  # killed mid-write; nothing left to flush
        self.process.wait()
        self._stderr.text()


def read_frame(path: Path, info: MediaInfo, seconds: float | None = None) -> np.ndarray:
    """Decode one RGBA frame of `path` at `seconds` (the first one when None)."""
    frames = [frame.copy() for frame in FrameReader(path, info, max_frames=1, start_seconds=seconds,
                                                    constant_rate=False)]
    if not frames:
        raise RuntimeError(f"ffmpeg decoded no frame from {path}" + (f" at {seconds:.1f} s" if seconds else ""))
    return frames[0]


def read_image(path: Path) -> np.ndarray:
    """Decode an image file to an RGBA uint8 array of shape (height, width, 4)."""
    return read_frame(path, probe(path))


def write_image(path: Path, frame: np.ndarray) -> None:
    """Encode an RGBA uint8 array to an image file (format from the suffix)."""
    height, width = frame.shape[:2]
    command = [tool_path("ffmpeg"), "-v", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-i", "-",
               "-frames:v", "1", "-pix_fmt", "rgb24", str(path)]
    completed = subprocess.run(command, input=memoryview(np.ascontiguousarray(frame)).cast("B"),
                               capture_output=True, creationflags=_CREATE_NO_WINDOW)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg could not write {path}: {completed.stderr.decode('utf-8', 'replace').strip()}")
