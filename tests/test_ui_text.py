"""The window's wording helpers: source descriptions, audio plans and durations (no window needed)."""
import unittest
from fractions import Fraction
from pathlib import Path

import needs

if not needs.HAS_OPENCV:
    raise unittest.SkipTest("opencv-python is not installed (the window needs it)")

from fsr4 import ui
from fsr4.media import AudioStream, MediaInfo


def video_info(audio: list[AudioStream]) -> MediaInfo:
    return MediaInfo(width=1440, height=1080, frame_rate=Fraction(30000, 1001), frame_count=6094,
                     duration=203.3, codec="wmv3", audio=audio)


class TextTest(unittest.TestCase):
    def test_format_duration(self):
        self.assertEqual(ui.format_duration(0), "0:00")
        self.assertEqual(ui.format_duration(203.3), "3:23")
        self.assertEqual(ui.format_duration(3725), "1:02:05")

    def test_describe_wmv(self):
        text = ui.describe_source(Path("movie.wmv"), video_info([AudioStream(0, "wmav2", 2)]))
        self.assertEqual(text, "WMV 9 · 1440 × 1080 · 29.97 fps\n3:23 · 6,094 frames · WMA audio")

    def test_describe_hdr_and_interlaced(self):
        info = video_info([])
        info.color_transfer, info.interlaced = "arib-std-b67", True
        self.assertEqual(ui.describe_source(Path("clip.mov"), info).splitlines()[1],
                         "HLG HDR shown as SDR · interlaced, deinterlaced · 3:23 · 6,094 frames · no audio")

    def test_describe_image(self):
        info = MediaInfo(width=1600, height=900, frame_rate=Fraction(0), frame_count=1, duration=None, codec="png")
        self.assertEqual(ui.describe_source(Path("photo.png"), info), "PNG image · 1600 × 900")

    def test_audio_plan(self):
        wma = video_info([AudioStream(0, "wmav2", 2)])
        self.assertEqual(ui.describe_audio_plan(wma, Path("out.mp4")),
                         "Audio: WMA converted to AAC (.mp4 cannot hold it).")
        self.assertEqual(ui.describe_audio_plan(wma, Path("out.mkv")), "Audio: WMA copied as is.")
        mixed = video_info([AudioStream(0, "aac", 2), AudioStream(1, "pcm_s16le", 2)])
        self.assertEqual(ui.describe_audio_plan(mixed, Path("out.mp4")),
                         "Audio: AAC copied as is; PCM converted to AAC (.mp4 cannot hold it).")
        self.assertEqual(ui.describe_audio_plan(video_info([]), Path("out.mp4")), "No audio in this file.")


if __name__ == "__main__":
    unittest.main()
