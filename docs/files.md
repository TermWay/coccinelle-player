# Files

Anything ffmpeg decodes:

- **Video:** MP4, MKV, MOV, AVI, WMV/ASF, WebM, MPEG-1/2 (MPG, VOB, TS, M2TS), FLV, 3GP, Ogg (OGG, OGV, OGM:
  Theora or VP8 with Vorbis or Opus), MXF, DV, RealMedia, Y4M.
- **Images:** PNG, JPEG, BMP, TIFF, WebP, EXR, GIF (first frame).

`tests/test_formats.py` round-trips a sample of each family. An .ogg with only sound (music) is refused with
that reason.

## What the tool does to make them come out right
- **The real video stream** is upscaled, never a cover picture (WMV files from Windows Movie Maker and many
  MP4/MKV files carry a JPEG thumbnail as a video stream).
- **Audio**: every track is kept. AAC, MP3, AC-3, E-AC-3 and ALAC are copied into the MP4; WMA, PCM, Vorbis,
  Opus, FLAC, MP2 and the rest are converted to AAC (96 kb/s per channel), because MP4 cannot hold them or
  common players cannot play them there. Save as `.mkv` to copy every track as is.
- **Subtitles, title, date, chapters**: an `.mkv` copies every subtitle track; `.mp4` and `.mov` take the text
  ones as MP4 text and leave picture ones (PGS, DVD) out. The file's title, creation date, comment and chapters
  are copied; a saved start keeps the chapters it reaches.
- **Colour**: YUV is decoded with the file's own matrix, or BT.709 for untagged HD as players assume, and the
  output is encoded with BT.709 and tagged. Earlier outputs were BT.601 and untagged, which players showed
  with saturated colours off by up to 35 levels; the round trip is now within 2 levels.
- **HDR** (PQ or HLG, as phones record) is tone-mapped to SDR, and **interlaced** video is deinterlaced at its
  own frame rate: see [Measurements](measurements.md#hdr-and-interlaced-sources).
- **Frame rate**: variable-rate sources (phones, screen recorders) are read at a constant rate on a grid that
  starts at the file's start, so the video stays in sync with the audio (a video that starts after its audio is
  padded at the start, as players show it).
- **Exact frames**: a read that starts in the middle of a file returns exactly the frames a read from the start
  numbers there, in every container (`tests/test_formats.py` checks MP4, WMV, MKV, MPEG-PS, MPEG-TS, Ogg and
  variable-rate files). MPEG-TS, MPEG-PS, FLV and Ogg files cannot seek reliably on their own, so their
  keyframes are indexed first. This is what lets pieces upscaled at different times join without a skipped or
  doubled frame.
- **Rotation**: portrait phone videos are turned upright, with the output size swapped accordingly.
- **Anamorphic video** (DVD, DV, HDV: pixels stored wider or narrower than square) comes out at the shape players
  show, with square pixels: a 16:9 DVD's 720 × 480 at 2x gives 1706 × 960, FSR scaling each side on its own.
  Images are taken as square.
- **Damaged files**: pictures ffmpeg cannot decode are said in the log (once per file); the picture before each
  one stands in for it, as players do.
