"""Image and video input/output through ffmpeg and ffprobe subprocesses, as raw RGBA (or BGRA) frames.

Decoding picks the real video stream (not a cover picture), converts YUV to RGB with the source's colour
matrix (BT.709 for untagged HD, as players assume), and puts the video on a constant-rate timeline anchored at
the file's start: frame n is the picture shown at `start_time + n / frame_rate`. Reads that start in the middle
of a file land on exactly the same frame n as a read from the start, whatever the container, so separately
upscaled pieces join without a duplicated or missing frame and stay in sync with the audio.

Encoding converts back with BT.709 for HD output and tags the file, copies every audio track an MP4 can hold
and re-encodes the others (WMA, PCM, Vorbis, Opus, FLAC, ...) to AAC. Pieces encoded separately are joined
losslessly with ffmpeg's concat demuxer.
"""
from __future__ import annotations

import bisect
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

from . import log, usage

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".exr", ".gif")
VIDEO_SUFFIXES = (".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".wmv", ".asf", ".flv", ".f4v", ".mpg", ".mpeg",
                  ".m2v", ".vob", ".ts", ".m2ts", ".mts", ".3gp", ".3g2", ".ogg", ".ogv", ".ogm", ".mxf", ".dv",
                  ".rm", ".rmvb", ".divx", ".y4m")

ENCODERS = {
    "amf": ["-c:v", "hevc_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", "18", "-qp_p", "20", "-tag:v", "hvc1"],
    "h264_amf": ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", "18", "-qp_p", "20"],
    # AV1 quantizers run 0-255: 40/48 gives the same PSNR as the HEVC settings with smaller files (README).
    "av1_amf": ["-c:v", "av1_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", "40", "-qp_p", "48"],
    "x264": ["-c:v", "libx264", "-crf", "17", "-preset", "medium"],
    "x265": ["-c:v", "libx265", "-crf", "18", "-preset", "medium", "-tag:v", "hvc1"],
    "png": ["-c:v", "png"],
}
GPU_ENCODERS = frozenset({"amf", "h264_amf", "av1_amf"})
# Smallest and largest frame (width, height) each GPU encoder accepts on a Radeon RX 9000 (measured; outside
# them the encoder fails to start).
ENCODER_LIMITS = {"amf": ((384, 128), (8192, 4352)), "av1_amf": ((320, 128), (8192, 4352)),
                  "h264_amf": ((96, 66), (4096, 4096))}
ENCODER_NAMES = {"amf": "HEVC on the GPU", "av1_amf": "AV1 on the GPU", "h264_amf": "H.264 on the GPU",
                 "x264": "x264", "x265": "x265", "png": "PNG"}

# Audio codecs an MP4/MOV file holds and every common player decodes: these are copied, the rest re-encoded.
MP4_AUDIO_COPY = frozenset({"aac", "mp3", "ac3", "eac3", "alac"})
COPY_ANY_AUDIO_SUFFIXES = frozenset({".mkv"})
AAC_KBPS_PER_CHANNEL = 96
HD_HEIGHT = 720
STDERR_LINES = 40
# Containers whose demuxer seeks to a keyframe before the requested time (they carry an index). Others
# (MPEG-TS/PS, FLV, Ogg, raw streams) can land on a frame that does not decode, so they get a keyframe index.
INDEXED_CONTAINERS = frozenset({"mov", "mp4", "matroska", "webm", "asf", "avi"})
SEEK_MARGIN_SECONDS = 1.0
KEYFRAME_SLACK_SECONDS = 0.25
SEEK_ATTEMPTS = 3
FIRST_FRAME_TIMEOUT = 20.0
_CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
_SHOWINFO_TIME = re.compile(r"Parsed_showinfo.*\bn:\s*0\b.*\bpts_time:\s*(-?[\d.]+)")
_decode_errors_reported: set[Path] = set()
# ffmpeg logs a decoder's messages under the decoder's name: the codec's name except for these.
DECODER_NAMES = {"av1": "libdav1d"}


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


def _parse_seconds(text) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


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
    start_time: float = 0.0


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
    start_time: float = 0.0

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

    @property
    def seeks_by_index(self) -> bool:
        """True when ffmpeg's own seeking lands on a decodable keyframe (MP4, MOV, MKV, WebM, WMV, AVI)."""
        return any(name in INDEXED_CONTAINERS for name in self.container.split(","))

    def frame_seconds(self, frame: int) -> float:
        """Seconds from the file's start to frame `frame` (what `-ss` takes)."""
        return float(frame / self.frame_rate) if self.frame_rate else 0.0


def probe(path: Path) -> MediaInfo:
    command = [
        tool_path("ffprobe"), "-v", "error", "-show_entries",
        "stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,channels,"
        "color_space,pix_fmt,start_time:stream_disposition=attached_pic:stream_side_data=rotation"
        ":format=duration,format_name,start_time",
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
        # An .ogg is often music only: say so rather than leave the file looking broken.
        sound = any(stream.get("codec_type") == "audio" for stream in streams)
        raise RuntimeError(f"no video stream in {path}" + (" (only a cover picture)" if videos
                                                           else " (only sound)" if sound else ""))
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
    start_time = _parse_seconds(data.get("format", {}).get("start_time"))
    if start_time is None:
        start_time = _parse_seconds(video.get("start_time")) or 0.0
    return MediaInfo(
        width=width, height=height, frame_rate=frame_rate, frame_count=frame_count, duration=duration,
        codec=video.get("codec_name", "?"), stream_index=int(video["index"]),
        color_matrix=color_space if color_space and color_space not in ("unknown", "reserved") else None,
        rotation=rotation,
        audio=[AudioStream(position, stream.get("codec_name", "?"), int(stream.get("channels") or 2),
                           _parse_seconds(stream.get("start_time")) or start_time)
               for position, stream in enumerate(audio_streams)],
        container=data.get("format", {}).get("format_name", ""), start_time=start_time)


def _popen(command: list[str], background: bool = False, **kwargs) -> subprocess.Popen:
    """Start ffmpeg without a console window; `background` work runs at the Speed setting's priority."""
    flags = _CREATE_NO_WINDOW | (usage.background_creation_flags() if background else 0)
    return subprocess.Popen(command, creationflags=flags, **kwargs)


class _StderrTail:
    """Drains a process's stderr on a thread (a full pipe would block ffmpeg) and keeps the last lines.

    With `watch_first_time`, it also picks the first decoded frame's time out of the showinfo filter's lines.
    `first_error` is the first message ffmpeg tagged as an error (it tags them with `-v level+...`) on a line
    starting with `errors_from` ("[theora @ ": the video decoder, not the audio ffmpeg decodes while opening).
    """

    def __init__(self, stream, watch_first_time: bool = False, errors_from: str = ""):
        self.lines: deque[str] = deque(maxlen=STDERR_LINES)
        self.first_error: str | None = None
        self.errors_from = errors_from
        self.first_time: float | None = None
        self.first_time_seen = threading.Event()
        self.watch_first_time = watch_first_time
        self.thread = threading.Thread(target=self._drain, args=(stream,), daemon=True)
        self.thread.start()

    def _drain(self, stream) -> None:
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if self.watch_first_time and "Parsed_showinfo" in line:
                if self.first_time is None:
                    match = _SHOWINFO_TIME.search(line)
                    if match:
                        self.first_time = float(match.group(1))
                        self.first_time_seen.set()
                continue
            if self.first_error is None and "[error] " in line and line.startswith(self.errors_from):
                self.first_error = line.split("[error] ", 1)[1]
            self.lines.append(line)
        stream.close()
        self.first_time_seen.set()

    def text(self) -> str:
        self.thread.join(timeout=5)
        return "\n".join(self.lines)


def rate_text(rate: Fraction) -> str:
    return f"{rate.numerator}/{rate.denominator}"


def seek_position(info: MediaInfo, start_frame: int, keyframes: list[float] | None,
                  margin: float = SEEK_MARGIN_SECONDS) -> float | None:
    """Where ffmpeg should seek (seconds from the file start) so decoding begins before frame `start_frame`.

    With a keyframe index (absolute timestamps), that is just before the last keyframe at least a frame and a
    half before the target; without one, `margin` seconds before it, trusting the container's own index to
    snap back to a keyframe. None means decode from the start.
    """
    if start_frame <= 0 or not info.frame_rate:
        return None
    target = info.frame_seconds(start_frame)
    if keyframes is not None:
        before = info.start_time + target - 1.5 / float(info.frame_rate)
        index = bisect.bisect_right(keyframes, before) - 1
        if index < 0:
            return None
        position = keyframes[index] - info.start_time - KEYFRAME_SLACK_SECONDS
    else:
        position = target - margin
    return position if position > 0 else None


def scan_keyframes(path: Path, info: MediaInfo) -> list[float]:
    """Absolute timestamps of the video stream's keyframes, from its packets (no decoding)."""
    command = [tool_path("ffprobe"), "-v", "error", "-select_streams", str(info.stream_index),
               "-show_entries", "packet=pts_time,dts_time,flags", "-of", "csv=p=0", str(path)]
    completed = subprocess.run(command, capture_output=True, text=True, creationflags=_CREATE_NO_WINDOW)
    if completed.returncode != 0:
        raise RuntimeError(f"cannot index the keyframes of {path.name}: {ffmpeg_reason(completed.stderr, path)}")
    keyframes = []
    for line in completed.stdout.splitlines():
        parts = line.split(",")
        if len(parts) < 3 or "K" not in parts[2]:
            continue
        seconds = _parse_seconds(parts[0]) if parts[0] not in ("", "N/A") else _parse_seconds(parts[1])
        if seconds is not None:
            keyframes.append(seconds)
    keyframes.sort()
    return keyframes


class KeyframeIndex:
    """The keyframes of a file whose container does not seek reliably, scanned on a background thread.

    `get()` returns None for containers with their own index (ffmpeg's seeking is enough), else waits for the
    scan and returns the keyframe timestamps for `FrameReader(keyframes=...)`.
    """

    def __init__(self, path: Path, info: MediaInfo):
        self.path = path
        self.info = info
        self._ready = threading.Event()
        self._keyframes: list[float] | None = None
        self._error: Exception | None = None
        if info.seeks_by_index or is_image(path):
            self._ready.set()
        else:
            threading.Thread(target=self._scan, daemon=True).start()

    def _scan(self) -> None:
        try:
            self._keyframes = scan_keyframes(self.path, self.info)
        except Exception as error:  # noqa: BLE001 - handed to whoever needs the index
            self._error = error
        finally:
            self._ready.set()

    def get(self) -> list[float] | None:
        self._ready.wait()
        if self._error is not None:
            raise RuntimeError(f"cannot seek in {self.path.name}: {self._error}") from self._error
        return self._keyframes


class FrameReader:
    """Yields uint8 frames of shape (height, width, 4) decoded by ffmpeg, numbered on the file's timeline.

    Frame n is the picture shown at `info.start_time + n / info.frame_rate` (ffmpeg's fps filter on a grid
    anchored at the file start, which repeats or drops pictures of variable-rate sources and pads a video
    that starts after the audio, like a constant-rate player). A read from `start_frame` seeks a margin
    before it (or to the keyframe before it, from `keyframes`), trims to the exact frame, and checks that the
    first decoded picture comes no later than the target; a seek that overshot is retried from further back.
    `timeline=False` decodes the first pictures as they come (images).

    `crop` (x, y, width, height in source pixels) and `size` (output width, height, scaled with `scale_flags`)
    shape what comes out; `pixel_format` is "rgba" or "bgra". `hardware` decodes on the GPU when ffmpeg can.
    Frames alternate between two buffers, so the frame from the previous iteration stays valid while the next
    one is being read (the flow worker needs that); anything older is overwritten.
    """

    def __init__(self, path: Path, info: MediaInfo, max_frames: int | None = None, start_frame: int = 0,
                 keyframes: list[float] | None = None, pixel_format: str = "rgba",
                 crop: tuple[int, int, int, int] | None = None, size: tuple[int, int] | None = None,
                 scale_flags: str = "bicubic", hardware: bool = False, background: bool = False,
                 timeline: bool = True):
        self.path = path
        self.info = info
        self.max_frames = max_frames
        self.start_frame = start_frame
        self.keyframes = keyframes
        self.pixel_format = pixel_format
        self.crop = crop
        self.size = size
        self.scale_flags = scale_flags
        self.hardware = hardware
        self.background = background
        self.timeline = timeline and bool(info.frame_rate)
        self.width, self.height = size or ((crop[2], crop[3]) if crop else (info.width, info.height))
        self.frame_bytes = self.width * self.height * 4
        self._buffers = [bytearray(self.frame_bytes), bytearray(self.frame_bytes)]
        self._next_buffer = 0
        self._attempt = 0
        self._start(SEEK_MARGIN_SECONDS)

    def command(self, seek: float | None) -> list[str]:
        info = self.info
        command = [tool_path("ffmpeg"), "-hide_banner", "-nostats", "-nostdin",
                   "-v", "level+info" if seek is not None else "level+error"]
        if self.hardware:
            command += ["-hwaccel", "auto"]
        threads = usage.current().ffmpeg_threads if self.background else None
        if threads is not None:
            command += ["-threads", str(threads)]
        filters = []
        if self.timeline:
            command += ["-copyts"]
            if seek is not None:
                command += ["-noaccurate_seek", "-ss", f"{seek:.6f}"]
            # Timestamps relative to the file start, a constant-rate grid from there, then the exact frame. The
            # check sits before the grid: the fps filter would pad copies in front of a picture that came late.
            filters.append(f"setpts=PTS-({info.start_time:.6f})/TB")
            if seek is not None:
                filters.append("showinfo=checksum=0")
            filters += [f"fps=fps={rate_text(info.frame_rate)}:start_time=0", f"trim=start_pts={self.start_frame}"]
        if self.crop:
            x, y, width, height = self.crop
            filters.append(f"crop={width}:{height}:{x}:{y}:exact=1")
        scale = []
        if self.size:
            scale += [f"w={self.size[0]}", f"h={self.size[1]}", f"flags={self.scale_flags}"]
        if info.needs_hd_matrix:
            scale.append("in_color_matrix=bt709")
        if scale:
            filters.append("scale=" + ":".join(scale))
        filters.append(f"format={self.pixel_format}")
        command += ["-i", str(self.path), "-map", f"0:{info.stream_index}", "-an", "-sn", "-dn",
                    "-vf", ",".join(filters)]
        if self.timeline:
            command += ["-fps_mode", "passthrough"]
        if self.max_frames is not None:
            command += ["-frames:v", str(self.max_frames)]
        command += ["-f", "rawvideo", "-pix_fmt", self.pixel_format, "-"]
        return command

    def _start(self, margin: float) -> None:
        seek = seek_position(self.info, self.start_frame, self.keyframes, margin) if self.timeline else None
        self._margin = margin
        self._verify = seek is not None
        # After a seek, the pictures decoded before the first keyframe fail by design: only a read from the
        # start tells a damaged file.
        self._judge_errors = seek is None
        self.process = _popen(self.command(seek), background=self.background, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        decoder = DECODER_NAMES.get(self.info.codec, self.info.codec)
        self._stderr = _StderrTail(self.process.stderr, watch_first_time=self._verify, errors_from=f"[{decoder} @ ")

    def __iter__(self):
        while True:
            frame = self._read_frame()
            if frame is None:
                break
            yield frame
        self.close()

    def read_into(self, frame: np.ndarray) -> bool:
        """Decode the next frame into `frame` (height, width, 4) uint8; False at the end (then call `close`)."""
        if frame.nbytes != self.frame_bytes or not frame.flags.c_contiguous:
            raise ValueError(f"buffer must be a contiguous {self.height}x{self.width}x4 array")
        return self._fill(memoryview(frame).cast("B"))

    def _fill(self, view: memoryview) -> bool:
        """Read one frame into `view`; False when the decoder has no more frames."""
        filled = 0
        while filled < self.frame_bytes:
            count = self.process.stdout.readinto(view[filled:])
            if not count:
                break
            filled += count
        if filled not in (0, self.frame_bytes):
            raise RuntimeError(f"ffmpeg stopped mid-frame ({filled} of {self.frame_bytes} bytes): {self._stderr.text()}")
        if self._verify:
            # Right after a seek: the first picture decoded must come no later than the target. With no frame
            # at all, the end of the file only counts when the decoder did start before the target.
            self._verify = False
            if not self._stderr.first_time_seen.wait(FIRST_FRAME_TIMEOUT):
                raise RuntimeError(f"ffmpeg did not report the first frame's time: {self._stderr.text()}")
            first_time = self._stderr.first_time
            landed = None if first_time is None else int(round(first_time * float(self.info.frame_rate)))
            if landed is None or landed > self.start_frame:
                # The seek landed after the target (its keyframe was further back than the margin).
                self.stop()
                self._attempt += 1
                if self._attempt >= SEEK_ATTEMPTS:
                    raise RuntimeError(f"cannot seek to frame {self.start_frame} of {self.path.name} "
                                       f"(landed on {landed if landed is not None else 'no frame'})")
                self._start(self._margin * 8)
                return self._fill(view)
        return filled != 0

    def _read_frame(self) -> np.ndarray | None:
        buffer = self._buffers[self._next_buffer]
        self._next_buffer ^= 1
        if not self._fill(memoryview(buffer)):
            return None
        return np.frombuffer(buffer, dtype=np.uint8).reshape(self.height, self.width, 4)

    def close(self) -> None:
        """Wait for the decoder after the last frame; raises when ffmpeg failed."""
        self.process.stdout.close()
        code = self.process.wait()
        stderr = self._stderr.text()
        if code != 0:
            raise RuntimeError(f"ffmpeg decode failed ({code}): {stderr.strip()}")
        self._report_decode_errors()

    def stop(self) -> None:
        """Kill the decoder before the end of the file (cancel, seek) and release its pipes."""
        self.process.kill()
        self.process.stdout.close()
        self.process.wait()
        self._stderr.text()
        self._report_decode_errors()

    def _report_decode_errors(self) -> None:
        """ffmpeg leaves out the pictures it cannot decode and still succeeds, and the timeline repeats the
        picture before each: say so, once per file, so a damaged file does not pass for a good one."""
        error = self._stderr.first_error
        if self._judge_errors and error is not None and self.path not in _decode_errors_reported:
            _decode_errors_reported.add(self.path)
            log.emit(f"[media] ffmpeg could not decode some pictures of {self.path.name} ({error}): the picture "
                     f"before each one is shown in its place")


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


def codec_tag_arguments(encoder: str, output: Path) -> list[str]:
    """The encoder's `-tag:v` for MP4/MOV outputs (Apple players want hvc1 for HEVC); none for MKV."""
    arguments = ENCODERS[encoder]
    if output.suffix.lower() not in (".mp4", ".mov", ".m4v") or "-tag:v" not in arguments:
        return []
    index = arguments.index("-tag:v")
    return arguments[index:index + 2]


class FrameWriter:
    """Feeds RGBA uint8 frames to ffmpeg, which encodes them and adds the source's audio tracks.

    `gop` sets the keyframe interval (pieces use a short one so playback can seek inside them quickly);
    `background` runs the encoder at the Speed setting's priority and thread count.
    """

    def __init__(self, path: Path, width: int, height: int, frame_rate: Fraction, encoder: str,
                 audio_source: Path | None = None, audio: list[AudioStream] | None = None,
                 audio_seconds: float | None = None, gop: int | None = None, background: bool = False):
        if encoder not in ENCODERS:
            raise ValueError(f"unknown encoder {encoder}; choose from {', '.join(ENCODERS)}")
        self.path = path
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 4
        self.frames = 0
        rate = rate_text(frame_rate) if frame_rate else "30"
        command = [tool_path("ffmpeg"), "-v", "error", "-nostdin", "-y",
                   "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r", rate, "-i", "-"]
        tracks = audio or []
        if audio_source is not None and tracks:
            if audio_seconds is not None:
                command += ["-t", f"{audio_seconds:.3f}"]
            command += ["-i", str(audio_source), "-map", "0:v:0"] + audio_arguments(tracks, path)
        command += ENCODERS[encoder]
        if gop is not None:
            command += ["-g", str(gop)]
        threads = usage.current().ffmpeg_threads if background and encoder not in GPU_ENCODERS else None
        if threads is not None and encoder == "x265":
            # x265 maps -threads to frame threads (at most 16); its worker pool is what sets the CPU use.
            command += ["-x265-params", f"pools={threads}"]
        elif threads is not None:
            command += ["-threads", str(threads)]
        if encoder != "png":
            command += color_arguments(height)
        command.append(str(path))
        self.process = _popen(command, background=background, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._stderr = _StderrTail(self.process.stderr)

    def write(self, frame: np.ndarray) -> None:
        if frame.nbytes != self.frame_bytes:
            raise ValueError(f"frame has {frame.nbytes} bytes, writer expects {self.frame_bytes}")
        try:
            self.process.stdin.write(memoryview(frame).cast("B"))
        except (BrokenPipeError, OSError) as error:
            self.process.wait()
            raise RuntimeError(f"ffmpeg encoder stopped: {self._stderr.text()}") from error
        self.frames += 1

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


def concat_list(paths: list[Path]) -> str:
    """The concat demuxer's list file for `paths` (absolute, forward slashes, quotes escaped)."""
    lines = []
    for path in paths:
        text = path.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{text}'\n")
    return "".join(lines)


def join_pieces(pieces: list[Path], output: Path, encoder: str, audio_source: Path | None,
                audio: list[AudioStream], seconds: float | None = None) -> None:
    """Join separately encoded pieces of one stream into `output` without re-encoding, adding the source's
    audio from its start (cut to `seconds` when the pieces cover only the beginning)."""
    if not pieces:
        raise ValueError("nothing to join")
    listing = pieces[0].parent / f"{output.stem}.join.txt"
    listing.write_text(concat_list(pieces), encoding="utf-8")
    command = [tool_path("ffmpeg"), "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0",
               "-i", str(listing)]
    if audio_source is not None and audio:
        command += ["-i", str(audio_source), "-map", "0:v:0"] + audio_arguments(audio, output)
    command += ["-c:v", "copy"] + codec_tag_arguments(encoder, output)
    if seconds is not None:
        command += ["-t", f"{seconds:.6f}"]
    command.append(str(output))
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               creationflags=_CREATE_NO_WINDOW | usage.background_creation_flags())
    listing.unlink()
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg could not join the pieces into {output.name}: {completed.stderr.strip()}")


def read_frame(path: Path, info: MediaInfo, frame: int = 0, keyframes: list[float] | None = None,
               pixel_format: str = "rgba") -> np.ndarray:
    """Decode frame `frame` of `path` (the first picture of an image)."""
    reader = FrameReader(path, info, max_frames=1, start_frame=frame, keyframes=keyframes,
                         pixel_format=pixel_format, timeline=not is_image(path))
    frames = [picture.copy() for picture in reader]
    if not frames:
        raise RuntimeError(f"ffmpeg decoded no frame from {path}" + (f" at frame {frame}" if frame else ""))
    return frames[0]


def read_thumbnail(path: Path, info: MediaInfo, frame: int, height: int) -> np.ndarray | None:
    """A small RGB picture near frame `frame` (the first keyframe from there: fast and always clean), or None
    past the end. For the filmstrip, where the exact frame does not matter."""
    width = max(2, 2 * int(round(info.width * height / info.height / 2)))
    scale = f"scale={width}:{height}:flags=area" + (":in_color_matrix=bt709" if info.needs_hd_matrix else "")
    command = [tool_path("ffmpeg"), "-v", "error", "-nostdin", "-noaccurate_seek",
               "-ss", f"{info.frame_seconds(frame):.6f}", "-skip_frame", "nokey", "-i", str(path),
               "-map", f"0:{info.stream_index}", "-an", "-sn", "-dn", "-frames:v", "1", "-vf", scale,
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    completed = subprocess.run(command, capture_output=True,
                               creationflags=_CREATE_NO_WINDOW | usage.background_creation_flags())
    if completed.returncode != 0:
        raise RuntimeError(f"cannot read a thumbnail of {path.name}: "
                           f"{ffmpeg_reason(completed.stderr.decode('utf-8', 'replace'), path)}")
    if len(completed.stdout) < width * height * 3:
        return None
    return np.frombuffer(completed.stdout[:width * height * 3], np.uint8).reshape(height, width, 3)


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
