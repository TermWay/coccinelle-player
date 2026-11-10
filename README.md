# Coccinelle Player

Upscales an image or a video with **AMD FSR Upscaling 4** (the RDNA 4 machine-learning upscaler, FSR SDK 2.4,
upscaler 4.1.2) on a Radeon RX 9000 series card: in a dark desktop window that plays the video upscaled while it
is being made, with sound, next to the source enlarged by a classic filter; from the command line; or by
dropping a file on `upscale.bat`.

FSR 4 is a game upscaler: it expects a low-resolution render plus depth, per-pixel motion vectors and a
sub-pixel camera jitter every frame. A video file has none of those, so this tool substitutes:

| FSR input      | What the tool feeds it                                                   |
|----------------|--------------------------------------------------------------------------|
| Colour         | The decoded frame (sRGB 8-bit, declared as non-linear colour)            |
| Motion vectors | DIS optical flow between consecutive frames (`--flow dis`), or zero      |
| Depth          | A constant plane                                                         |
| Jitter         | None (faking it by resampling the input only blurs, see the lab)         |
| Exposure       | FSR's auto exposure                                                      |

## Requirements
- Windows and a Radeon RX 9000 card (FSR 4's FP8 model) with a current Adrenalin driver. AMD's requirements
  also name the RX 7000 series (FSR 4's INT8 model; the RX 6000 series appears only in AMD's timings): untested
  here.
- ffmpeg 5.1 or later (6.0 or later for the AV1 GPU encoder, 8.0 or later for AMD's HQ scaler), with ffprobe, on PATH or next to `fsr4.exe`. All
  decoding and encoding goes through it, and it is not included (see License). One way to install it:
  `winget install Gyan.FFmpeg.Essentials`, then open a new window.
- For the exe: nothing else. For the source: Python 3.12 and `python -m pip install -r requirements.txt` (numpy,
  and `opencv-python` for the window and the optical flow; without OpenCV the command line still runs with
  `--flow zero`), plus the FSR DLLs in `bin/`: `python tools/fetch_sdk.py` downloads them from AMD's GitHub
  (tag v2.4.0) and checks their pinned SHA-256.
- Optional, for the Real-ESRGAN upscaler: `realesrgan-ncnn-vulkan-20220424-windows.zip` from
  [Real-ESRGAN's release v0.2.5.0](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0), unzipped into
  `bin\realesrgan\` in the folder of `fsr4.exe` (or the repository), or on PATH with its `models` folder beside it.
  It is a separate program under its own license (BSD 3-Clause) and is not included.

## Platforms
- **Windows:** supported. Tested on Windows 11 with an RX 9070 XT; AMD's requirements also name the RX 7000
  series (FSR 4's INT8 model), not tested here.
- **Linux:** not supported yet. AMD ships FSR 4 only as signed Windows DLLs, and the window draws, plays sound,
  takes dropped files and goes full screen through Win32. The Windows build may run under Wine or Proton;
  untested.
- **macOS:** not supported. AMD ships FSR 4 for Windows only, and Apple silicon Macs have no Radeon.

## Use
**Window:** double-click `Coccinelle Player.exe` (or `start.bat` in the source tree), or drop a file on its icon.
- Drop a video or an image anywhere on the window, click the empty preview, or use Choose file (Ctrl+O).
  Every setting explains itself when the pointer rests on it.
- **Size:** 1.5×, 2×, 3×, 1440p or 4K (the short side; chips that would leave FSR's 1×–3× range are disabled),
  or Custom with a slider. The sidebar shows the output size, what happens to each audio track and what the
  chosen encoder trades.
- **Watch it upscaled:** Play (Space) plays the video in real time with sound. The upscaler follows the
  playhead: it keeps the next seconds upscaled, jumps to wherever you seek, and keeps everything it made. When
  it is slower than the video (1080p60 to 4K runs at about 44 fps), playback waits with a spinner, like a
  streaming site, then plays on. The seek bar shows what is upscaled (lighter) and where the upscaler works
  (a tick), and hovering it gives the frame number and its time to the millisecond; the filmstrip above it
  shows the video, bright where it is upscaled. The buttons around Play step one frame back or forward (hold
  to keep stepping); the time shows the current frame's number and keeps one width, so the volume after it
  stays put. The volume slider is a decibel scale like the speed's, marked 0, 1, 3, 10, 30 and 100%, so equal
  distances sound like equal steps (click, drag, the wheel over it or the speaker, ↑ ↓); the speaker shows the
  level.
- **Speed:** the slider at the right of the controls is a log scale from .05× to 5×, with 0 at its left end: 0
  pauses, and the next speed plays on. A drag stops anywhere (0.35×) and lands on a mark when near one; [, ] and
  the wheel go from mark to mark, Backspace or a click on the value goes back to 1× forwards. The ◀ button
  beside it (R) plays backwards at the slider's speed. Dragging never interrupts the picture: the decoders keep
  running and the sound follows when you stop. The sound plays from 0.25× to 2× at its own pitch and is silent
  otherwise. Above 2× every second or third frame is shown, backwards every frame, and playback slows down when
  decoding cannot keep up. 1080p60 reaches about 3.6× at 5× here (2.6× with a 4K upscale shown), and backwards
  about 43 frames a second (32 with a 4K upscale; the source decodes in short chunks, last frame first), so −0.5×
  and slower play at their speed and anything faster plays at −0.5× to −0.7×. Backwards, the upscaler works on
  the seconds just behind the playhead.
- **Compare anything:** the divider splits the picture; the Left and Right dropdowns above it (or a click on
  the label in either top corner) choose what each side shows: a preset, or the source enlarged by a classic filter:
  Original pixels (each source pixel as a block), Bilinear (what most players do), Bicubic, or Lanczos. Two sides
  upscaled sit side by side, paused or playing (two versions share the GPU, so playing them waits more), and the
  sidebar then asks which one its settings are for, which is also the one Upscale saves. Off turns a side off: the
  other one fills the picture, without the divider. Dragging the divider also takes its handle to the pointer's height.
- **Presets** cover the choices worth comparing: FSR 4 at its best (medium optical flow, no sharpening) with faster,
  sharpened and motionless variants, FSR 3.1, FSR 2.3, AMD's HQ scaler and Real-ESRGAN. The sidebar shows the preset,
  its upscaler and only what that upscaler can change: motion vectors and sharpening for FSR, sharpening for the HQ
  scaler, nothing for Real-ESRGAN. + saves those settings as a preset of your own (the size and the encoder stay the
  file's) in `%APPDATA%\Coccinelle Player\presets.json`, and the bin deletes one.
- **Zoom:** the Fit button shows the whole picture; the slider beside it is marked like the speed's: Original
  shows the source at its own pixel size (the upscaled side shrunk to match); 100% to 800% show upscaled pixels
  1:1 or enlarged. Click or drag anywhere on it (137%), or click a mark. From 100% up every pixel is drawn as a block, exactly as the file holds it. All of it works while playing. Drag the divider; the wheel zooms around the
  pointer in finer steps (four per doubling); right-drag or the minimap moves the visible area; double-click
  switches between Fit and 100%.
- **Upscale** (Ctrl+Enter) does the rest of the file the way the saved side says, starting at the playhead, and
  saves it: the pieces are joined without re-encoding and the source's audio is added. Stop (Esc) keeps what is
  done while the file stays open, and Upscale carries on from there. Changing a setting starts a new version
  from the playhead; the earlier ones are kept, so going back to them is instant.
- **Images** are upscaled as soon as they open, with every version shown, so presets can be compared before
  Upscale saves the picture.
- **Auto upscale**, the switch beside Choose file, kept between runs: off, files open as in a plain player (the
  source, bicubic) and FSR starts only when you choose a preset in the Left or Right list.
- **Full screen** (F11, F or the button) covers the monitor the window is on and keeps the video; leaving it
  puts the window back exactly where and as it was. Its controls come up when the mouse moves near the bottom
  of the screen, and hide, with the pointer, once the mouse rests, wherever it is.
  The button at the top left of the window (Ctrl+B) folds the settings away. **Processor** in the settings
  sets how many threads upscaling may use (1 to all) and its priority (Normal, Low, Background).
- When it is saved, Open plays the file and Show in folder selects it. Show log (Ctrl+L) has the details;
  errors appear in a red banner. The foot of the sidebar names the processor and the graphics card, with the
  memory this program uses on each (RAM, VRAM) against what is installed.

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| Space | play / pause | ← → | 5 seconds back / forward |
| , . | one frame back / forward | Home End | start / end |
| F, F11 | full screen | Esc | leave full screen, else stop the upscale |
| M | mute | ↑ ↓ | volume |
| [ ] | slower / faster | Backspace | normal speed, forwards |
| R | play backwards / forwards | | |
| Ctrl+B | settings sidebar | Ctrl+O, Ctrl+Enter, Ctrl+L | open, upscale, log |

**Drag and drop on a script:** drop one or more files on `upscale.bat`: each is upscaled 2x with no questions
asked. Other settings go through `fsr4 upscale` (below). Windows hands a dropped name to a script unquoted when
it has no space, so cmd cuts one holding `&`, `^`, `,`, `;` or `=`: open those in the window or with
`fsr4 upscale`.

**Command line:**
```
fsr4 upscale clip.mp4                       # clip.fsr4.mp4, 2x (performance mode), HEVC on the GPU, audio kept
fsr4 upscale photo.png --mode quality       # 1.5x
fsr4 upscale clip.mkv --scale 3 --codec x265 -o out.mp4
fsr4 upscale clip.mp4 --flow-preset ultrafast   # faster motion vectors, slightly lower quality
fsr4 upscale clip.mp4 --flow zero           # no motion vectors (no OpenCV needed)
fsr4 info                                   # GPU, extensions, FSR versions offered by the DLL
fsr4 selftest --validate                    # Vulkan round trip under the validation layer
```
From the source tree, `python -m fsr4 ...` is the same (`python -m fsr4 ui` for the window). `start.bat` and
`upscale.bat` use the executables in `dist\fsr4\` when they exist, else Python. Before opening one, `start.bat`
runs `tools/dev_build.py`, which rebuilds them when a source is newer than the build: into `out\dev-build\`, then
in place of `dist\fsr4\` once no program runs from there (until then the window opens from the sources, and a
failed build does the same). `build_exe.bat` (after
`python -m pip install -r requirements-build.txt`) checks the FSR DLLs against their pinned hashes and builds
the executables with PyInstaller from `fsr4.spec`: `Coccinelle Player.exe` (window, no console) and `fsr4.exe`
(command line) share one folder of about 170 MB (85 MB zipped), mostly OpenCV and numpy, with the FSR DLLs and
AMD's license in `bin\` and the third-party notices in `licenses\`. Its two optional arguments build elsewhere:
the folder that receives `fsr4\` and PyInstaller's work folder.

Scale is 1x to 3x (FSR's limit); the modes are native 1x, quality 1.5x, balanced 1.7x, performance 2x,
ultra 3x. `--sharpness 0..1` enables FSR's RCAS pass (with `--upscaler hq`, the HQ scaler's own). Video encoders: `amf` (HEVC on the GPU, default),
`av1_amf`, `h264_amf`, `x264`, `x265`. `--upscaler 3.1` or `2.3` uses AMD's older upscalers instead of FSR 4,
`--upscaler hq` AMD's HQ scaler (ffmpeg 8.0 or later): faster, no motion vectors, each frame on its own.
`--upscaler realesrgan` Real-ESRGAN (installed apart, see Requirements): it redraws each frame, about 10 times slower
than FSR 4, with no motion vectors and no sharpening. `--upscaler realesrgan-x4plus` (general) and
`realesrgan-x4plus-anime` (drawings) are its slower models, for still images only: a video is refused.

**Settings, all of them:** FSR 4 itself takes the output size, the optional RCAS sharpening, and per frame a
colour image, depth, motion vectors, camera jitter, exposure and a reset flag. Depth, jitter and exposure have
no meaning for a video, so they are fixed (table above); the motion vectors are the one input worth a choice
(optical flow presets or off). The SDK's five tuning keys (velocity factor, reactiveness, shading change,
accumulation per frame, minimum disocclusion accumulation) change nothing in FSR 4: its output stayed identical
to the bit when they were tried here; they only act on FSR 3.1. The HQ scaler takes the output size and sharpening;
its algorithm stays Video SR 1.1, the best of the filter's five. Real-ESRGAN takes only its model, offered as the Upscaler setting: the
video model runs at the smallest of its scales (2, 3, 4) that reaches the output size, the two for still images at 4x, then the
picture is resized to the output size. The DLL also holds FSR 3.1.6 and 2.3.4, offered as the Upscaler setting
to compare. Measured on the panning fractal clip (PSNR against the original after a 2x downscale, DIS fast):

| Upscaler | PSNR | GPU time per frame (1600x900 to 3200x1800) |
|----------|------|---------------------------------------------|
| FSR 4.1.2 | 26.78 dB | 3.5 ms |
| FSR 2.3.4 | 26.54 dB | 2.4 ms |
| Lanczos | 26.37 dB | |
| FSR 3.1.6 | 25.92 dB (25.08 with velocity factor 0) | 2.5 ms |

## Files
Anything ffmpeg decodes: MP4, MKV, MOV, AVI, WMV/ASF, WebM, MPEG-1/2 (MPG, VOB, TS, M2TS), FLV, 3GP, Ogg
(OGG, OGV, OGM: Theora or VP8 with Vorbis or Opus), MXF, DV, RealMedia, Y4M; images PNG, JPEG, BMP, TIFF, WebP,
EXR, GIF (first frame). `tests/test_formats.py` round-trips a sample of each family. An .ogg with only sound
(music) is refused with that reason. What the tool does to make them come out right:
- **The real video stream** is upscaled, never a cover picture (WMV files from Windows Movie Maker and many
  MP4/MKV files carry a JPEG thumbnail as a video stream).
- **Audio**: every track is kept. AAC, MP3, AC-3, E-AC-3 and ALAC are copied into the MP4; WMA, PCM, Vorbis,
  Opus, FLAC, MP2 and the rest are converted to AAC (96 kb/s per channel), because MP4 cannot hold them or
  common players cannot play them there. Save as `.mkv` to copy every track as is. Subtitles are dropped.
- **Colour**: YUV is decoded with the file's own matrix, or BT.709 for untagged HD as players assume, and the
  output is encoded with BT.709 and tagged. Earlier outputs were BT.601 and untagged, which players showed
  with saturated colours off by up to 35 levels; the round trip is now within 2 levels.
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

## How playback works
The window never waits for a whole file. `fsr4/render.py` upscales into pieces (0.5 s at first after a seek so
playback can start soon, growing to 8 s): each piece is one encoder run, and a few frames before each new
stretch are upscaled but not kept so FSR's history and the optical flow are warm. `fsr4/timeline.py` keeps the
pieces and decides where to work: ahead of the playhead in preview mode (from when less than 12 s is ready to
30 s ahead), or everything, starting at the playhead, after Upscale. Saving joins the pieces with ffmpeg's
concat demuxer (stream copy) and adds the audio; `tests/test_render.py` checks that a joined video matches a
single pass frame for frame, with the audio in sync and the colour tags kept.

Each set of settings shown is a version with its own pieces. In preview mode the
renderer upscales every version shown, taking turns: the one with the least ready ahead of the playhead goes
first, and a run yields once it is ahead of another by as much as that one has (half a second at first, so
both show their frame soon after a seek, then up to 2 s). Two versions of 1080p60 at 4K share the renderer's
40 fps, about 20 each: after a seek the first shows its frame in 1.3 s and the second in 2.1 s, and playing
both waits more often. Versions no longer shown are kept (the four most recent) so switching back is instant.

`fsr4/player.py` plays the ready pieces: one decoder per upscaled version shown (cropped to the view by
ffmpeg), one for the source (the filters), the sound card's clock (waveOut, `fsr4/audio.py`, with ffmpeg's
atempo away from 1×) deciding which frame is due. Without sound (0.1×, faster than 2×, backwards) the system
clock runs; faster than 2× and backwards every frame is shown in order: a late decoder slows playback down
rather than making it skip to frames nobody decoded. Backwards, each decoder decodes a short chunk forward (one
seek) and hands it over last frame first while the chunk before it decodes; about 64 MB of frames per chunk,
so 8 frames of 1080p and 11 of an upscale shown in a 1600 × 900 view. `fsr4/ui_video.py` draws everything with GDI: Tk's image path takes 20 ms for a 1600×900 frame and
120 ms at 4K, a GDI blit 0.5 to 2 ms, so video plays smoothly in the window and full screen.

## Measured on an RX 9070 XT
Measured on the development PC (an RX 9070 XT and a 16-core processor) with clips that are not in the
repository: the commands repeat them on yours. End to end (decode, flow, upscale, encode), `--flow dis` with the
default `fast` preset, HEVC AMF output:

| Input                              | Output     | fps  | main thread per frame | flow (worker thread) |
|------------------------------------|------------|------|-----------------------|----------------------|
| 1280x720 60 fps screen recording   | 2560x1440  | 85   | 3.9 ms                | 9 ms                 |
| 1920x1080 60 fps screen recording  | 3840x2160  | 47   | 9.1 ms                | 14 ms                |
| same, `--flow-preset ultrafast`    | 3840x2160  | 55   | 8.8 ms                | 8 ms                 |
| same, `--flow-preset medium`       | 3840x2160  | 21   | 9.0 ms                | 41 ms                |
| same, `--flow zero`                | 3840x2160  | 60   | 8.5 ms                | 0                    |
| same, `--upscaler hq`              | 3840x2160  | 72   | in the encoder        | 0                    |
| same, `--upscaler realesrgan`      | 3840x2160  | 4.8  | in its program        | 0                    |

The 3840x2160 rows: `python -m fsr4 upscale <clip> --scale 2 --frames 600`, plus the option named.

One 1920x1080 picture to 3840x2160, the whole command (`python -m fsr4 upscale <image> --scale 2 --upscaler <name>`):
`realesrgan` 1.8 s, `realesrgan-x4plus-anime` 4.3 s, `realesrgan-x4plus` 10.1 s.

The FSR dispatch itself is about 2 ms at 1440p and 4.5 ms at 4K including the copies; optical flow on the
CPU is what limits the rate. So 1080p30 content upscales in real time, 1080p60 at about three quarters of real time.

In the window (`python tools/ui_smoke.py`, 1080p60 WMV to 4K): the first upscaled frame shows 1.6 s after the
file is opened, playback starts about 0.3 s after Play once a few seconds are ready, and plays at 60 fps.

**Encoders** (a natural-footage clip downscaled to 1440×1080 and a 1080p60 screen recording; upscaled 2×, then the same FSR frames encoded by each; frames taken per second, bitrate, PSNR against
the lossless FSR output):

| Encoder | Natural 2880×2160 | Screen recording 3840×2160 | Plays on |
|---------|-------------------|----------------------------|----------|
| HEVC GPU (`amf`, default) | 80 fps, 4.3 Mb/s, 44.53 dB | 64 fps, 10.8 Mb/s, 36.08 dB | most devices since about 2016 |
| AV1 GPU (`av1_amf`) | 86 fps, 3.4 Mb/s, 44.52 dB | 68 fps, 9.1 Mb/s, 36.06 dB | recent browsers, GPUs and TVs |
| H.264 GPU (`h264_amf`) | 97 fps, 5.2 Mb/s, 44.49 dB | 75 fps, 12.3 Mb/s, 36.07 dB | everything; at most 4096 wide |
| H.264 CPU (`x264`) | 63 fps, 7.8 Mb/s, 44.53 dB | 56 fps, 9.2 Mb/s, 36.03 dB | everything |
| HEVC CPU (`x265`) | 45 fps, 3.5 Mb/s, 44.53 dB | 39 fps, 9.9 Mb/s, 36.08 dB | most devices since about 2016 |

Every encoder lands on the same quality with these settings, so the choice is file size, speed and the devices
that must play the file. The GPU encoders take frames faster than FSR makes them and leave the processor free;
x265 at 4K is slower than FSR and holds the upscale back. AV1 uses quantizers 40/48, the setting that matches
HEVC's PSNR; 56/64 gives files about 30% smaller than HEVC for 0.1 dB less. The GPU encoders refuse frames
larger than 8192 × 4352 (HEVC, AV1) or 4096 per side (H.264), which the size check reports before starting.

**Processor settings** (1080p60 to 4K; processor time is the share of all 32 threads of a 16-core CPU; Task
Manager's CPU column shows "% Processor Utility", which counts boosted clock speed and reads much higher; the last
column is how late a 10 ms timer of the window fires, its responsiveness):

| Encoder | Threads, priority | Upscale | Processor time | Task Manager CPU | Window timer late (99th percentile) |
|---------|-------------------|---------|----------------|------------------|-------------------------------------|
| HEVC GPU | 32, Normal | 41 fps | 2.5% | 41% | 3.3 ms |
| HEVC GPU | 16, Low (default) | 41 fps | 2.4% | 42% | 2.9 ms |
| HEVC GPU | 8, Background | 42 fps | 2.2% | 41% | 2.9 ms |
| x265 | 32, Normal | 33 fps | 6.6% | 82% | 2.5 ms |
| x265 | 16, Low | 31 fps | 8.2% | 76% | 2.5 ms |
| x265 | 8, Background | 28 fps | 5.2% | 62% | 2.4 ms |

With the GPU encoder the upscale needs less than one core, whatever the setting; the thread count matters with
the CPU encoders, and Background runs only on time other programs leave idle (games keep priority). OpenCV's
optical flow is fastest with about 8 threads here (47 fps; 32 threads: 39 fps), so it never gets more than 8.

Quality lab (`python tools/lab_quality.py`, PSNR against the original after a 2x area downscale):

| Content                                   | Lanczos | AMD HQ scaler | Real-ESRGAN | FSR 4                      |
|-------------------------------------------|---------|---------------|-------------|----------------------------|
| Still image, smooth CGI wallpaper         | 42.06   | 43.25         | 36.02       | 42.32                      |
| Still image, aliased fractal              | 26.38   | 26.26         | 24.27       | 25.98                      |
| Panning fractal clip, no motion vectors   | 26.42   | 26.34         | 24.35       | 25.59                      |
| Panning fractal clip, DIS fast flow       | 26.42   | 26.34         | 24.35       | 26.82 (negated flow 25.63) |
| Panning fractal clip, DIS medium flow     | 26.42   | 26.34         | 24.35       | 27.10                      |
| Natural footage, DIS fast flow            | 44.90   | 44.32         | 34.36       | 44.46                      |
| Same, small clip through H.264 CRF 28     | 40.78   | 40.84         | 34.91       | 40.84                      |
| Screen recording, DIS fast flow           | 26.89   | 27.20         | 26.67       | 26.64                      |
| Drawn clip (flat colours, outlines)       | 35.31   | 36.76         | 33.42       | 33.20                      |
| Same, small clip through H.264 CRF 26     | 33.06   | 34.07         | 34.28       | 31.75                      |

Things the lab settled: motion vectors from optical flow help (+1.2 dB over zero flow, and the sign is
verified by the negated run); faking the camera jitter by resampling the input costs 2 to 8 dB and is not
done; feeding a still image several times does not help either, so images get one pass. On screen
recordings the FSR output is a touch crisper than Lanczos on text, with no visible artifacts.

AMD's HQ scaler (ffmpeg's `sr_amf`, Video SR 1.1, default sharpness; `--upscaler hq`) works on one frame at a time.
At 2880x2160 it takes 9 ms a frame against 17 ms for FSR 4 with DIS fast flow; a whole job at 4K runs 1.5 times as
fast as FSR 4 with motion and 1.2 times as fast without. It wins on the smooth still, is level with FSR 4 on natural footage, and loses 0.5 dB on the panning clip,
where FSR 4 gathers detail across frames. Lanczos leads on natural footage only because PSNR rewards softness: on
one frame it keeps 43% of the original's detail energy, the HQ scaler 99%, FSR 4 112%. Video SR 1.0 scores a little
higher on stills (43.81) but leaves stair steps on diagonal edges.

Real-ESRGAN (`realesrgan-ncnn-vulkan`, its video model; `--upscaler realesrgan`) is there for those who want its look.
It looks much crisper and removes compression artifacts, but it repaints the picture rather than recovering
it: on one frame of natural footage, 4.5 times the original's detail energy and SSIM 0.979 against 0.995 for FSR 4.
It comes out 2 to 3 levels brighter, and on small text it draws clean letters that are the wrong letters. It takes
160 ms a frame at 2880x2160 (FSR 4: 19 ms). On a drawn clip (made for the lab: flat colours, black outlines, a panning background) it scores below the HQ scaler when the file is clean, and first of the three once H.264 at CRF 26 has marked it up, because it wipes the artifacts; it also flickers least there (temporal PSNR 46.8, HQ scaler 45.9, FSR 4 41.0).
The program only takes files and each run costs 0.34 s to start, so frames go through it in chunks of up to 16 PNG
files in a temporary folder beside the output (about 400 MB at 4K), two runs overlapping. A seek waits for the chunk being read, up to
about 3 s at 4K.

Its two slower models are for still images only, since in a video a seek would wait minutes. They are there for their
look: on clean pictures they stay below FSR 4 and Lanczos, and neither leads the other. No real photograph was at hand
to measure, so nothing here says how they do on one.

| Still image (2x, PSNR)                | Video model | `x4plus` | `x4plus-anime` | FSR 4 | Lanczos |
|---------------------------------------|-------------|----------|----------------|-------|---------|
| Drawn frame (flat colours, outlines)  | 33.40       | 34.68    | 35.24          | 34.03 | 35.24   |
| Smooth CGI wallpaper                  | 34.85       | 36.09    | 35.72          | 39.59 | 39.05   |
| Rendered landscape                    | 39.46       | 38.49    | 40.14          | 44.13 | 44.99   |
| Soft frame of natural footage         | 35.06       | 34.06    | 32.64          | 47.39 | 49.41   |
| Aliased fractal                       | 24.27       | 24.06    | 23.54          | 25.98 | 26.38   |

On the drawn frame the anime model keeps the colours closest (at most 1.7 levels off on average, the video model 4.7,
`x4plus` 4.0), with a thin halo along the outlines that comes from the model, not from the resizing; `x4plus` leaves
blotches in flat colours. Both run at 4x, so their picture nearly always comes down to the output size: with an area
filter, which scores 0.1 to 0.5 dB above Lanczos for these two (the video model keeps Lanczos, 0.05 to 0.1 dB above area).

## Layout
- `fsr4/vk.py`, `fsr4/vk_types.py`, `fsr4/gpu.py`: ctypes Vulkan binding, generated structs, device and copies.
- `fsr4/ffx.py`: the FSR API (`amd_fidelityfx_loader_vk.dll`) and the upscaling context.
- `fsr4/media.py`: ffmpeg frame reader and writer. `fsr4/flow.py`: motion vectors. `fsr4/pipeline.py`: a frame.
  `fsr4/hq_scaler.py`: AMD's HQ scaler, run by the encoder's ffmpeg. `fsr4/realesrgan.py`: Real-ESRGAN, frames in
  chunks through its program. `fsr4/upscalers.py`: which pipeline an upscaler uses.
- `fsr4/job.py`: one pass over a file with progress and cancel, used by `fsr4/cli.py` (command line) and for
  images in the window; `fsr4/pipeline.py`'s `upscale_stream` is the frame loop both it and the renderer use.
- `fsr4/render.py` (versions upscaled into pieces around the playhead, saving), `fsr4/timeline.py` (pieces
  and scheduling), `fsr4/player.py` (real-time playback), `fsr4/audio.py` (waveOut sound), `fsr4/usage.py`
  (processor threads and priority), `fsr4/presets.py` (built-in and saved presets), `fsr4/preferences.py` (the
  window's preferences), `fsr4/temporary.py` (temporary files go beside the output), `fsr4/image_preview.py` (an image
  upscaled with every version shown).
- The window: `fsr4/ui.py` (layout and wiring), `fsr4/ui_video.py` (the GDI-drawn preview and comparison),
  `fsr4/ui_transport.py` (filmstrip, seek bar, playback buttons, speed), `fsr4/ui_window.py` (full screen on
  the window's monitor), `fsr4/system.py` (processor name, RAM and VRAM in use), `fsr4/ui_tooltip.py`,
  `fsr4/ui_theme.py`
  (palette, fonts, ttk styles made of anti-aliased images, dark title bar and dialogs), `fsr4/ui_art.py` (numpy
  drawing, icons, PNG/PPM/ICO encoding), `fsr4/ui_dnd.py` (files dropped from Explorer). Plain tkinter and
  ctypes, no extra package. `run_fsr4.py`: the entry point of both executables.
- `tools/gen_vk_types.py`: regenerates the Vulkan structs from `third_party/vulkan_core.h`.
- `tools/fetch_sdk.py`: downloads the FSR DLLs and checks their pinned hashes (`--check` only checks).
  `tools/collect_licenses.py`: the packaged app's third-party notices (run by `build_exe.bat`).
  `tools/dev_build.py`: keeps `dist\fsr4\` built from the current sources without touching a running copy
  (`--force` builds anyway; run by `start.bat`). `tools/lab_quality.py`: the PSNR lab.
- `tools/ui_smoke.py`: drives the window through every state (drop, first upscaled frame, tooltip, dropdown,
  file dialog, playback, the seek bar's hover label, buffering after a seek, zoom with original pixels, the
  zoom slider (a drag that must keep the picture still) and wheel, a side's dropdown opened from its label, a preset saved and two presets side by side
  paused and playing, a frame step, playing backwards, a side turned off, full screen on the window's own
  monitor (the second one when there is one) with its controls showing near the bottom and hiding at rest,
  then back where it was, folded sidebar, whole-file save, an image compared with a preset, error, a video opened
  without upscaling, minimum size) and screenshots each with `tools/window_capture.py`. Its presets and preferences
  go to scratch files, never yours.
- `tests/`: `python -m unittest discover -s tests`. Tests that need ffmpeg, OpenCV or FSR 4 itself (the card
  and the DLLs in `bin/`) skip without it, saying why; `tests/test_audio.py` also needs a sound device.

## Limits
- No jitter and no real depth means FSR 4 works mostly as a very good spatial upscaler with temporal
  stabilisation; it cannot reconstruct the sub-pixel detail it recovers in games.
- Input is processed as 8-bit sRGB; HDR video is not handled, and interlaced video is not deinterlaced.
- The player plays the first audio track; the saved file keeps every track. Pieces wait in a `fsr4-` folder
  beside the output (about 10 MB per second of 4K) until the file is closed, or in the system's temporary folder when
  the output's cannot take them; folders left by a crash are removed the next time that folder is used.

## License
MIT (`LICENSE`) for this project's code. What it builds on keeps its own license:
- `third_party/ffx/`: AMD's FSR API headers, MIT; `fsr4/ffx.py` transcribes them. `third_party/vulkan_core.h`:
  Khronos, Apache-2.0 or MIT, used here under MIT (`third_party/LICENSE-Khronos.txt`); `fsr4/vk_types.py` is
  generated from it.
- The FSR DLLs (downloaded into `bin/`, not in the repository): AMD's signed binaries under AMD's FSR SDK license
  (`bin/license.md`), redistributable in binary form with that notice, which the exe ships next to them.
- The packaged app also holds Python, Tcl/Tk, numpy and OpenCV with the libraries they bundle, frozen by
  PyInstaller: their notices are in its `licenses\` folder.
- ffmpeg is not included: it runs as a separate program under its own license (GPL for the common builds).
- The executables are not code-signed: check a download against the SHA-256 published with it.

Not affiliated with AMD or the Khronos Group. AMD, Radeon and FidelityFX are trademarks of Advanced Micro
Devices, Inc.; Vulkan is a registered trademark of the Khronos Group Inc.

## Credits
Written by Claude (Anthropic's AI model) in Claude Code: the code, the tests and this README, to the requirements
in `SPEC.md`. Directed and tested by TermWay.
