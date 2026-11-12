"""Container and codec coverage: every sample goes through the decoder and the encoder (no GPU needed).

Each sample is generated with ffmpeg, decoded with FrameReader, re-encoded with FrameWriter together with the
source's audio, and the result is probed: frame count, frame size, audio codecs and A/V duration must match.
"""
import dataclasses
import json
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

import needs
from fsr4 import log, media

WIDTH, HEIGHT, FRAMES, RATE = 96, 64, 20, 10
# Ogg samples carry VP8: the libtheora in ffmpeg 2025-07 writes frames its own Theora decoder rejects
# ("error in unpack_block_qpis", "Coefficient index overflow"), whatever the settings, while Theora files made
# elsewhere decode cleanly (a 1080p game intro was checked by hand, seeks included).
OGG_VIDEO = ["-c:v", "libvpx", "-b:v", "300k"]


def ffmpeg(*arguments: str) -> None:
    completed = subprocess.run([media.tool_path("ffmpeg"), "-v", "error", "-y", *arguments], capture_output=True)
    if completed.returncode != 0:
        raise AssertionError(completed.stderr.decode("utf-8", "replace"))


def video_source(width: int = WIDTH, height: int = HEIGHT) -> list[str]:
    return ["-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={RATE}"]


def audio_source(frequency: int = 440, channels: int = 2) -> list[str]:
    layout = "stereo" if channels == 2 else "mono"
    return ["-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate=48000,aformat=channel_layouts={layout}"]


def moving_source(seconds: float = 3) -> list[str]:
    """A test picture scrolling sideways at 20 pictures a second: every moment differs from the one before."""
    return ["-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate=20,scroll=horizontal=0.04", "-t", str(seconds)]


def interlaced_sample(root: Path, name: str, *extra: str) -> Path:
    """`moving_source` woven into interlaced frames at 10 a second (top field first): MPEG-2 for .mpg, else H.264."""
    source = root / name
    codec = (["-c:v", "mpeg2video", "-q:v", "2", "-top", "1"] if name.endswith(".mpg")
             else ["-c:v", "libx264", "-x264opts", "tff=1", "-pix_fmt", "yuv420p"])
    ffmpeg(*moving_source(), "-vf", "tinterlace=mode=interleave_top,setfield=tff", "-flags", "+ildct+ilme", *codec,
           *extra, str(source))
    return source


def luma(frame: np.ndarray) -> np.ndarray:
    return frame[..., :3].astype(np.float64).mean(axis=2)


def rgb(frame: np.ndarray) -> np.ndarray:
    return frame[..., :3].astype(np.float64)


def psnr(first: np.ndarray, second: np.ndarray) -> float:
    return float(10 * np.log10(255 ** 2 / max(1e-9, np.mean((first - second) ** 2))))


def probe_streams(path: Path) -> list[dict]:
    completed = subprocess.run(
        [media.tool_path("ffprobe"), "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_type,codec_name,width,height,nb_read_frames,duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)["streams"]


@needs.ffmpeg
class FormatTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def roundtrip(self, source: Path, output_name: str = "out.mp4") -> tuple[media.MediaInfo, list[dict]]:
        """Decode `source`, re-encode its frames with its audio, return the probe info and the output streams."""
        info = media.probe(source)
        output = self.root / output_name
        writer = media.FrameWriter(output, info.width, info.height, info.frame_rate, "x264",
                                   audio_source=source, audio=info.audio)
        count = 0
        for frame in media.FrameReader(source, info):
            self.assertEqual(frame.shape, (info.height, info.width, 4))
            writer.write(frame)
            count += 1
        writer.close()
        self.assertEqual(count, info.frame_count, "reader frame count vs probe estimate")
        streams = probe_streams(output)
        video = [stream for stream in streams if stream["codec_type"] == "video"]
        self.assertEqual(len(video), 1)
        self.assertEqual(int(video[0]["nb_read_frames"]), count)
        return info, streams

    def audio_codecs(self, streams: list[dict]) -> list[str]:
        return [stream["codec_name"] for stream in streams if stream["codec_type"] == "audio"]

    def assert_in_sync(self, streams: list[dict]) -> None:
        durations = {stream["codec_type"]: float(stream["duration"]) for stream in streams}
        self.assertAlmostEqual(durations["video"], durations["audio"], delta=0.15)

    def test_wmv_with_wma_audio(self):
        source = self.root / "sample.wmv"
        ffmpeg(*video_source(), *audio_source(), "-t", str(FRAMES / RATE), "-c:v", "wmv2", "-c:a", "wmav2",
               "-b:a", "128k", str(source))
        info, streams = self.roundtrip(source)
        self.assertEqual(info.codec, "wmv2")
        self.assertEqual([track.codec for track in info.audio], ["wmav2"])
        self.assertEqual(self.audio_codecs(streams), ["aac"])
        self.assert_in_sync(streams)

    def test_cover_picture_before_the_video(self):
        cover = self.root / "cover.jpg"
        ffmpeg("-f", "lavfi", "-i", "color=c=red:size=32x32", "-frames:v", "1", str(cover))
        source = self.root / "covered.mp4"
        ffmpeg("-i", str(cover), *video_source(), *audio_source(), "-t", str(FRAMES / RATE),
               "-map", "0:v", "-map", "1:v", "-map", "2:a", "-c:v:0", "mjpeg", "-disposition:v:0", "attached_pic",
               "-c:v:1", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source))
        info, streams = self.roundtrip(source)
        self.assertEqual((info.width, info.height, info.codec), (WIDTH, HEIGHT, "h264"))
        self.assertEqual(self.audio_codecs(streams), ["aac"])

    def test_cover_picture_as_first_stream_like_windows_movie_maker(self):
        # ffprobe's view of a real WMV saved by Windows Movie Maker: the cover JPEG is stream 0.
        data = {"streams": [
            {"index": 0, "codec_name": "mjpeg", "codec_type": "video", "width": 640, "height": 480,
             "r_frame_rate": "90000/1", "avg_frame_rate": "0/0", "color_space": "bt470bg",
             "disposition": {"attached_pic": 1}},
            {"index": 1, "codec_name": "wmav2", "codec_type": "audio", "channels": 2,
             "r_frame_rate": "0/0", "avg_frame_rate": "0/0", "disposition": {"attached_pic": 0}},
            {"index": 2, "codec_name": "wmv3", "codec_type": "video", "width": 640, "height": 480,
             "r_frame_rate": "30000/1001", "avg_frame_rate": "30000/1001", "color_space": "unknown",
             "disposition": {"attached_pic": 0}}],
            "format": {"format_name": "asf", "duration": "203.349000"}}
        info = media.parse_probe(data, Path("movie.wmv"))
        self.assertEqual((info.stream_index, info.codec), (2, "wmv3"))
        self.assertEqual(info.frame_count, 6094)
        self.assertIsNone(info.color_matrix)
        self.assertEqual([(track.position, track.codec) for track in info.audio], [(0, "wmav2")])

    def test_avi_with_pcm_audio(self):
        source = self.root / "sample.avi"
        ffmpeg(*video_source(), *audio_source(), "-t", str(FRAMES / RATE), "-c:v", "mpeg4", "-c:a", "pcm_s16le",
               str(source))
        info, streams = self.roundtrip(source)
        self.assertEqual(self.audio_codecs(streams), ["aac"])
        self.assert_in_sync(streams)

    def test_mkv_with_two_tracks_mp4_cannot_hold(self):
        source = self.root / "sample.mkv"
        ffmpeg(*video_source(), *audio_source(440), *audio_source(660, channels=1), "-t", str(FRAMES / RATE),
               "-map", "0:v", "-map", "1:a", "-map", "2:a", "-c:v", "libvpx-vp9", "-b:v", "200k",
               "-c:a:0", "libopus", "-c:a:1", "libvorbis", str(source))
        info, streams = self.roundtrip(source)
        self.assertEqual([track.codec for track in info.audio], ["opus", "vorbis"])
        self.assertEqual(self.audio_codecs(streams), ["aac", "aac"])
        _, mkv_streams = self.roundtrip(source, "out.mkv")
        self.assertEqual(self.audio_codecs(mkv_streams), ["opus", "vorbis"], "MKV output copies any audio")

    def subtitled_sample(self) -> Path:
        """An MKV with an English SRT track and an ASS track, both shown during the 2 seconds of video."""
        subtitles = self.root / "lines.srt"
        subtitles.write_text("1\n00:00:00,200 --> 00:00:00,900\nHello\n\n2\n00:00:01,200 --> 00:00:01,800\nWorld\n",
                             encoding="utf-8")
        source = self.root / "subtitled.mkv"
        ffmpeg(*video_source(), *audio_source(), "-i", str(subtitles), "-i", str(subtitles), "-t", str(FRAMES / RATE),
               "-map", "0:v", "-map", "1:a", "-map", "2", "-map", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-c:a", "libopus", "-c:s:0", "srt", "-c:s:1", "ass", "-metadata:s:s:0", "language=eng", str(source))
        return source

    def subtitle_codecs(self, path: Path) -> list[str]:
        return [stream["codec_name"] for stream in probe_streams(path) if stream["codec_type"] == "subtitle"]

    def test_subtitles_are_kept(self):
        source = self.subtitled_sample()
        info = media.probe(source)
        self.assertEqual([(track.position, track.codec, track.language) for track in info.subtitles],
                         [(0, "subrip", "eng"), (1, "ass", "")])
        for name, expected in (("out.mkv", ["subrip", "ass"]), ("out.mp4", ["mov_text", "mov_text"])):
            output = self.root / name
            writer = media.FrameWriter(output, info.width, info.height, info.frame_rate, "x264", audio_source=source,
                                       audio=info.audio, subtitles=info.subtitles)
            for frame in media.FrameReader(source, info):
                writer.write(frame)
            writer.close()
            self.assertEqual(self.subtitle_codecs(output), expected, name)
            self.assertEqual(media.probe(output).subtitles[0].language, "eng", name)

    def test_joined_pieces_keep_the_subtitles_up_to_their_end(self):
        source = self.subtitled_sample()
        info = media.probe(source)
        frames = [frame.copy() for frame in media.FrameReader(source, info)]
        pieces = []
        for index, (start, end) in enumerate(((0, 6), (6, 12))):
            piece = self.root / f"piece_{index}.mp4"
            writer = media.FrameWriter(piece, info.width, info.height, info.frame_rate, "x264")
            for frame in frames[start:end]:
                writer.write(frame)
            writer.close()
            pieces.append(piece)
        output = self.root / "joined.mkv"
        media.join_pieces(pieces, output, "x264", source, info.audio, 12 / RATE, info.subtitles)
        self.assertEqual(self.subtitle_codecs(output), ["subrip", "ass"])
        durations = [float(stream["duration"]) for stream in probe_streams(output) if "duration" in stream]
        self.assertLessEqual(max(durations, default=0), 12 / RATE + 0.05, "every track ends with the video")

    def test_ogg_with_vorbis_audio(self):
        source = self.root / "sample.ogg"
        ffmpeg(*video_source(), *audio_source(), "-t", str(FRAMES / RATE), *OGG_VIDEO, "-c:a", "libvorbis",
               str(source))
        info, streams = self.roundtrip(source)
        self.assertEqual((info.codec, [track.codec for track in info.audio]), ("vp8", ["vorbis"]))
        self.assertEqual(self.audio_codecs(streams), ["aac"])
        self.assert_in_sync(streams)

    def test_pictures_ffmpeg_cannot_decode_are_reported(self):
        source = self.root / "damaged.avi"
        ffmpeg(*video_source(), "-t", "2", "-c:v", "mpeg4", "-g", "100", str(source))
        data = bytearray(source.read_bytes())
        data[len(data) // 2:len(data) // 2 + 200] = bytes(range(200))
        source.write_bytes(data)
        lines = []
        log.set_sink(lines.append)
        try:
            for _ in media.FrameReader(source, media.probe(source)):
                pass
        finally:
            log.set_sink(None)
        self.assertTrue(any("could not decode some pictures of damaged.avi" in line for line in lines), lines)

    def test_ogg_with_only_sound_is_refused_clearly(self):
        source = self.root / "music.ogg"
        ffmpeg(*audio_source(), "-t", "1", "-c:a", "libvorbis", str(source))
        with self.assertRaisesRegex(RuntimeError, r"no video stream .*\(only sound\)"):
            media.probe(source)

    def test_mpeg2_program_stream(self):
        source = self.root / "sample.mpg"
        ffmpeg(*video_source(), *audio_source(), "-t", str(FRAMES / RATE), "-c:v", "mpeg2video", "-r", "25",
               "-c:a", "mp2", str(source))
        info, streams = self.roundtrip(source)
        self.assertEqual(info.codec, "mpeg2video")
        self.assertEqual(self.audio_codecs(streams), ["aac"])

    def test_flv_with_mp3_is_copied(self):
        source = self.root / "sample.flv"
        ffmpeg(*video_source(), *audio_source(), "-t", str(FRAMES / RATE), "-c:v", "flv", "-c:a", "libmp3lame",
               "-ar", "44100", str(source))
        _, streams = self.roundtrip(source)
        self.assertEqual(self.audio_codecs(streams), ["mp3"])

    def test_rotated_phone_video(self):
        plain = self.root / "plain.mp4"
        ffmpeg(*video_source(), "-t", str(FRAMES / RATE), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(plain))
        source = self.root / "rotated.mov"
        ffmpeg("-display_rotation:v:0", "90", "-i", str(plain), "-c", "copy", str(source))
        info, _ = self.roundtrip(source)
        self.assertEqual(info.rotation, 90)
        self.assertEqual((info.width, info.height), (HEIGHT, WIDTH))

    def test_variable_frame_rate_comes_out_constant(self):
        source = self.root / "vfr.mkv"
        ffmpeg(*video_source(), "-t", str(FRAMES / RATE), "-vf", "select='not(eq(mod(n\\,3)\\,2))'",
               "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source))
        info = media.probe(source)
        frames = [frame for frame in media.FrameReader(source, info)]
        self.assertEqual(len(frames), info.frame_count)
        self.assertAlmostEqual(len(frames) / float(info.frame_rate), info.duration, delta=1.5 / float(info.frame_rate))

    def test_untagged_hd_decodes_as_bt709(self):
        source = self.root / "untagged_hd.mp4"
        ffmpeg("-f", "lavfi", "-i", "smptehdbars=size=1280x720:rate=10", "-frames:v", "2",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv,"
                      "setparams=colorspace=unknown:color_primaries=unknown:color_trc=unknown",
               "-c:v", "libx264", "-crf", "1",
               "-pix_fmt", "yuv420p", str(source))
        info = media.probe(source)
        self.assertIsNone(info.color_matrix)
        self.assertTrue(info.needs_hd_matrix)
        frame = media.read_frame(source, info)
        green = frame[int(720 * 0.3), int(1280 * 3.5 / 7)][:3].astype(int)
        # SMPTE HD bars: the fourth bar is green, about (12, 220, 2) in BT.709; BT.601 decoding gives ~(0, 185, 0).
        self.assertGreater(green[1], 210)

    def test_hdr_is_tone_mapped_to_the_sdr_picture(self):
        """An HDR video made from an SDR picture (its white at 203 nits, as BT.2408 places it) decodes close to
        that picture; read as if it were SDR, it is the washed-out one players without tone mapping show."""
        picture = self.root / "sdr.png"
        ffmpeg("-f", "lavfi", "-i", "testsrc2=size=192x128", "-frames:v", "1", "-update", "1", str(picture))
        reference = rgb(media.read_image(picture))
        for transfer, name in (("smpte2084", "PQ"), ("arib-std-b67", "HLG")):
            source = self.root / f"{name}.mkv"
            ffmpeg("-loop", "1", "-i", str(picture), "-frames:v", "3", "-r", "10",
                   "-vf", "format=gbrpf32le,zscale=tin=709:pin=709:min=gbr:rin=full:t=linear:p=709:m=gbr:r=full:"
                          f"npl=203,zscale=p=2020,zscale=t={transfer}:m=2020_ncl:r=tv:npl=203,format=yuv420p10le",
                   "-c:v", "libx265", "-x265-params",
                   f"colorprim=bt2020:transfer={transfer}:colormatrix=bt2020nc:lossless=1:log-level=error",
                   str(source))
            info = media.probe(source)
            self.assertEqual(info.hdr, name)
            lines = []
            log.set_sink(lines.append)
            try:
                frame = media.read_frame(source, info)
            finally:
                log.set_sink(None)
            self.assertTrue(any(f"is HDR ({name})" in line for line in lines), lines)
            washed_out = media.read_frame(source, dataclasses.replace(info, color_transfer=None))
            mapped, plain = psnr(rgb(frame), reference), psnr(rgb(washed_out), reference)
            # Measured 19.3 dB (PQ) and 19.0 dB (HLG) against 8.9 and 10.7 read as SDR.
            self.assertGreater(mapped, 18, name)
            self.assertGreater(mapped, plain + 6, name)
            thumbnail = media.read_thumbnail(source, info, 0, 64).astype(np.float64)
            self.assertAlmostEqual(thumbnail.mean(), frame[..., :3].mean(), delta=6)

    def test_hdr_needs_zscale(self):
        info = media.MediaInfo(width=64, height=64, frame_rate=Fraction(25), frame_count=1, duration=None,
                               codec="hevc", color_transfer="smpte2084")
        with mock.patch.object(media, "ffmpeg_filters", return_value=frozenset({"tonemap", "scale"})):
            with self.assertRaisesRegex(RuntimeError, r"HDR \(PQ\).*zscale filter"):
                media.tone_map_filters(info)
        self.assertEqual(media.tone_map_filters(dataclasses.replace(info, color_transfer="bt709")), [])

    def test_interlaced_video_is_deinterlaced(self):
        """Each interlaced frame holds two moments, one per field; deinterlaced, it is the first one again."""
        truth_path = self.root / "progressive.mkv"
        ffmpeg(*moving_source(), "-c:v", "ffv1", "-pix_fmt", "yuv444p", str(truth_path))
        truth = [luma(frame) for frame in media.FrameReader(truth_path, media.probe(truth_path))]
        for source in (interlaced_sample(self.root, "interlaced.mpg"), interlaced_sample(self.root, "interlaced.mp4")):
            info = media.probe(source)
            self.assertTrue(info.interlaced, source.name)
            combed = [luma(frame) for frame in media.FrameReader(source, dataclasses.replace(info, interlaced=False))]
            woven = [luma(frame) for frame in media.FrameReader(source, info)]
            self.assertEqual(len(woven), len(combed))
            # Frame n of the interlaced video starts at picture 2n of the progressive one (the ends lack a neighbour).
            before = np.mean([psnr(frame, truth[2 * index]) for index, frame in enumerate(combed) if 0 < index < 9])
            after = np.mean([psnr(frame, truth[2 * index]) for index, frame in enumerate(woven) if 0 < index < 9])
            self.assertGreater(after, before + 3, source.name)

    def test_progressive_video_is_not_deinterlaced(self):
        source = self.root / "progressive.mp4"
        ffmpeg(*video_source(), "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source))
        info = media.probe(source)
        self.assertFalse(info.interlaced)
        reader = media.FrameReader(source, info, max_frames=1)
        try:
            self.assertNotIn("bwdif", " ".join(reader.command(None)))
        finally:
            reader.stop()


@needs.ffmpeg
class SeekTest(unittest.TestCase):
    """A read that starts at frame n returns exactly the frames a read from the start numbers n, n+1, ...

    Pieces of an upscale are read from the middle of the file and joined later, so an off-by-one here would
    duplicate or drop a frame at every join and shift the comparison view against the upscaled video.
    """

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def assert_seeks_exact(self, source: Path, starts: tuple[int, ...] = (1, 2, 7, 13, 31, 58)) -> None:
        info = media.probe(source)
        keyframes = None if info.seeks_by_index else media.scan_keyframes(source, info)
        sequential = [frame.copy() for frame in media.FrameReader(source, info)]
        checked = 0
        for start in starts:
            if start + 1 >= len(sequential):
                continue
            reader = media.FrameReader(source, info, max_frames=2, start_frame=start, keyframes=keyframes)
            frames = [frame.copy() for frame in reader]
            self.assertEqual(len(frames), 2, f"{source.name} from frame {start}")
            np.testing.assert_array_equal(frames[0], sequential[start], f"{source.name} frame {start}")
            np.testing.assert_array_equal(frames[1], sequential[start + 1], f"{source.name} frame {start + 1}")
            checked += 1
        self.assertGreater(checked, 2)

    def long_sample(self, name: str, *codec: str, seconds: int = 6, rate: int = 10) -> Path:
        source = self.root / name
        ffmpeg("-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate={rate}", *audio_source(),
               "-t", str(seconds), *codec, str(source))
        return source

    def test_mp4_h264_with_b_frames(self):
        self.assert_seeks_exact(self.long_sample("seek.mp4", "-c:v", "libx264", "-g", "25", "-pix_fmt", "yuv420p"))

    def test_wmv(self):
        self.assert_seeks_exact(self.long_sample("seek.wmv", "-c:v", "wmv2", "-g", "25", "-c:a", "wmav2"))

    def test_mkv_vp9(self):
        self.assert_seeks_exact(self.long_sample("seek.mkv", "-c:v", "libvpx-vp9", "-b:v", "300k", "-g", "25"))

    def test_mpeg_ts_with_long_gop_uses_the_keyframe_index(self):
        source = self.long_sample("seek.ts", "-c:v", "libx264", "-g", "40", "-pix_fmt", "yuv420p", "-c:a", "aac")
        self.assertFalse(media.probe(source).seeks_by_index)
        self.assert_seeks_exact(source)

    def test_a_seek_that_lands_late_is_retried(self):
        """Without its keyframe index, an MPEG-TS seek lands after the target; the reader must notice (the
        first decoded picture comes after the target) and read from further back, not pad or return nothing."""
        source = self.long_sample("late.ts", "-c:v", "libx264", "-g", "50", "-pix_fmt", "yuv420p", "-c:a", "aac")
        info = media.probe(source)
        sequential = [frame.copy() for frame in media.FrameReader(source, info)]
        for start in (31, 45):
            frames = [frame.copy() for frame in media.FrameReader(source, info, max_frames=2, start_frame=start)]
            self.assertEqual(len(frames), 2, f"from frame {start}")
            np.testing.assert_array_equal(frames[0], sequential[start], f"frame {start}")
            np.testing.assert_array_equal(frames[1], sequential[start + 1], f"frame {start + 1}")

    def test_mpeg_ps(self):
        self.assert_seeks_exact(self.long_sample("seek.mpg", "-c:v", "mpeg2video", "-g", "25", "-c:a", "mp2"))

    def test_ogg_uses_the_keyframe_index(self):
        source = self.long_sample("seek.ogv", *OGG_VIDEO, "-g", "25", "-c:a", "libopus")
        self.assertFalse(media.probe(source).seeks_by_index)
        self.assert_seeks_exact(source)

    def test_interlaced_mpeg_ps(self):
        source = interlaced_sample(self.root, "seek_interlaced.mpg", "-g", "12")
        self.assert_seeks_exact(source, starts=(1, 2, 7, 13, 21))

    def test_variable_frame_rate(self):
        source = self.root / "seek_vfr.mkv"
        ffmpeg("-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate=10", "-t", "6",
               "-vf", "setpts='N/10/TB+0.03*sin(N)/TB'", "-fps_mode", "vfr", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", str(source))
        self.assert_seeks_exact(source)

    def test_video_starting_after_the_audio_is_padded(self):
        plain = self.long_sample("plain.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac")
        late = self.root / "late.mp4"
        ffmpeg("-itsoffset", "0.5", "-i", str(plain), "-i", str(plain), "-map", "0:v", "-map", "1:a", "-c", "copy",
               str(late))
        info = media.probe(late)
        frames = [frame.copy() for frame in media.FrameReader(late, info)]
        # 0.5 s at 10 fps: frames 0-4 repeat the first picture, so frame 5 is shown 0.5 s after the audio starts.
        for index in range(1, 5):
            np.testing.assert_array_equal(frames[index], frames[0])
        self.assertFalse(np.array_equal(frames[5], frames[6]))
        self.assert_seeks_exact(late)


class ArgumentTest(unittest.TestCase):
    def test_audio_arguments(self):
        tracks = [media.AudioStream(0, "wmav2", 2), media.AudioStream(1, "aac", 6), media.AudioStream(2, "pcm_s24le", 6)]
        arguments = media.audio_arguments(tracks, Path("out.mp4"))
        self.assertEqual(arguments, ["-map", "1:a:0", "-c:a:0", "aac", "-b:a:0", "192k",
                                     "-map", "1:a:1", "-c:a:1", "copy",
                                     "-map", "1:a:2", "-c:a:2", "aac", "-b:a:2", "576k"])
        self.assertEqual(media.audio_arguments(tracks, Path("out.mkv")).count("copy"), 3)

    def test_subtitle_arguments(self):
        tracks = [media.SubtitleStream(0, "hdmv_pgs_subtitle"), media.SubtitleStream(1, "subrip", "fre"),
                  media.SubtitleStream(2, "mov_text")]
        self.assertEqual(media.subtitle_arguments(tracks, Path("out.mp4")),
                         ["-map", "1:s:1", "-c:s:0", "mov_text", "-map", "1:s:2", "-c:s:1", "copy"])
        self.assertEqual(media.subtitle_arguments(tracks, Path("out.mkv")).count("copy"), 3)
        self.assertEqual(media.subtitle_arguments(tracks, Path("out.webm")), [])
        self.assertEqual(media.describe_subtitles(tracks, Path("out.mp4")),
                         ["subtitle track 1: hdmv_pgs_subtitle cannot go in .mp4, left out (.mkv keeps it)",
                          "subtitle track 2: subrip (fre), converted to MP4 text (its styling is lost)",
                          "subtitle track 3: mov_text, copied"])
        self.assertEqual(media.source_arguments(Path("in.mkv"), [], tracks[:1], Path("out.mp4")), [],
                         "no input is added when nothing is taken from it")

    def test_frame_rate_guess(self):
        from fractions import Fraction
        self.assertEqual(media.guess_frame_rate(Fraction(1000), Fraction(30000, 1001)), Fraction(30000, 1001))
        self.assertEqual(media.guess_frame_rate(Fraction(30), Fraction(2987, 100)), Fraction(30))
        self.assertEqual(media.guess_frame_rate(Fraction(0), Fraction(25)), Fraction(25))
        self.assertEqual(media.guess_frame_rate(Fraction(0), Fraction(0)), Fraction(30))

    @needs.ffmpeg
    def test_unreadable_file_gives_a_plain_reason(self):
        stderr = ("[mov,mp4,m4a,3gp,3g2,mj2 @ 000002860cf942c0] moov atom not found\n"
                  "clips\\broken.mp4: Invalid data found when processing input\n")
        self.assertEqual(media.ffmpeg_reason(stderr, Path("clips\\broken.mp4")), "Invalid data found when processing input")
        self.assertEqual(media.ffmpeg_reason("[asf @ 0x1f2e] moov atom not found"), "moov atom not found")
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "broken.mp4"
            broken.write_bytes(b"not a video")
            with self.assertRaises(RuntimeError) as raised:
                media.probe(broken)
            self.assertTrue(str(raised.exception).startswith("not a video or image ffmpeg can read ("))
            self.assertNotIn("@", str(raised.exception))

    def test_an_old_ffmpeg_is_named(self):
        error = media.ffmpeg_failure("ffmpeg decode failed (1)", "Unrecognized option 'fps_mode'.\n")
        self.assertIn("version 5.1 or later is needed", str(error))
        self.assertEqual(str(media.ffmpeg_failure("ffmpeg encode failed (1)", "Conversion failed!\n")),
                         "ffmpeg encode failed (1): Conversion failed!")

    def test_every_listed_suffix_is_known(self):
        self.assertIn(".wmv", media.VIDEO_SUFFIXES)
        self.assertFalse(set(media.VIDEO_SUFFIXES) & set(media.IMAGE_SUFFIXES))
        self.assertTrue(np.all([suffix.startswith(".") for suffix in media.VIDEO_SUFFIXES]))


if __name__ == "__main__":
    unittest.main()
