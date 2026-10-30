# FSR4 upscaler

Upscales an image or a video with **AMD FSR Upscaling 4** (the RDNA 4 machine-learning upscaler, FSR SDK 2.4,
upscaler 4.1.2) on a Radeon RX 9000 series card, from the command line, the packaged `fsr4.exe`, or by
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
- Windows, a Radeon RX 9000 (FSR 4 FP8 model) or RX 7000/6000 (INT8 model) with a current Adrenalin driver.
- ffmpeg and ffprobe on PATH (or next to `fsr4.exe`). All decoding and encoding goes through them.
- For the exe: nothing else. For the source: Python 3.12 with numpy, `opencv-python` for optical flow
  (without it, `--flow zero` still works), and the FSR SDK DLLs in `bin/` (`python tools/fetch_sdk.py`
  downloads them from AMD's GitHub, tag v2.4.0, hashes pinned).

## Use
**Window:** double-click `fsr4.exe` (or `start.bat`, the Play button). Pick a file, choose the scale, the
motion vectors and the encoder, press Upscale. The window shows the input, the progress with fps and time
left, the log, and when the job is done a 100% crop of the result next to the same crop enlarged without
FSR. Cancel keeps what was written so far. `fsr4.exe ui photo.png` opens it with a file preselected.

**Drag and drop:** drop a file on `upscale.bat` for a 2x upscale with no questions asked.

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
`upscale.bat` use `dist\fsr4\fsr4.exe` when it exists, else Python. `build_exe.bat` builds the exe into
`dist\fsr4\` with PyInstaller (about 205 MB, mostly OpenCV and numpy; the FSR DLLs are copied into `bin\`).

Scale is 1x to 3x (FSR's limit); the modes are native 1x, quality 1.5x, balanced 1.7x, performance 2x,
ultra 3x. `--sharpness 0..1` enables FSR's RCAS pass. Video encoders: `amf` (HEVC on the GPU, default),
`h264_amf`, `x264`, `x265`.

## Measured on an RX 9070 XT
End to end (decode, flow, upscale, encode), `--flow dis` with the default `fast` preset, HEVC AMF output:

| Input                              | Output     | fps  | main thread per frame | flow (worker thread) |
|------------------------------------|------------|------|-----------------------|----------------------|
| 1280x720 60 fps screen recording   | 2560x1440  | 85   | 3.9 ms                | 9 ms                 |
| 1920x1080 60 fps screen recording  | 3840x2160  | 41   | 10.6 ms               | 16 ms                |
| same, `--flow-preset ultrafast`    | 3840x2160  | 49   | 9.7 ms                | 11 ms                |
| 1920x1080, `--flow zero`           | 3840x2160  | 160+ | 6 ms                  | 0                    |

The FSR dispatch itself is about 2 ms at 1440p and 4.5 ms at 4K including the copies; optical flow on the
CPU is what limits the rate. So 1080p30 content upscales in real time, 1080p60 at about two thirds speed.

Quality lab (`python tools/lab_quality.py`, PSNR against the original after a 2x area downscale):

| Content                                   | Lanczos | FSR 4                      |
|-------------------------------------------|---------|----------------------------|
| Still image, smooth CGI wallpaper         | 42.06   | 42.32                      |
| Still image, aliased fractal              | 26.38   | 25.98                      |
| Panning fractal clip, no motion vectors   | 26.42   | 25.59                      |
| Panning fractal clip, DIS fast flow       | 26.42   | 26.82 (negated flow 25.63) |
| Panning fractal clip, DIS medium flow     | 26.42   | 27.09                      |

Things the lab settled: motion vectors from optical flow help (+1.2 dB over zero flow, and the sign is
verified by the negated run); faking the camera jitter by resampling the input costs 2 to 8 dB and is not
done; feeding a still image several times does not help either, so images get one pass. On screen
recordings the FSR output is a touch crisper than Lanczos on text, with no visible artifacts.

## Layout
- `fsr4/vk.py`, `fsr4/vk_types.py`, `fsr4/gpu.py`: ctypes Vulkan binding, generated structs, device and copies.
- `fsr4/ffx.py`: the FSR API (`amd_fidelityfx_loader_vk.dll`) and the upscaling context.
- `fsr4/media.py`: ffmpeg frame reader and writer. `fsr4/flow.py`: motion vectors. `fsr4/pipeline.py`: a frame.
- `fsr4/job.py`: one job with progress and cancel, shared by `fsr4/cli.py` (command line) and `fsr4/ui.py`
  (tkinter window); `run_fsr4.py`: the exe entry point (no arguments opens the window).
- `tools/gen_vk_types.py`: regenerates the Vulkan structs from `third_party/vulkan_core.h`.
- `tools/fetch_sdk.py`: downloads the FSR DLLs. `tools/lab_quality.py`: the PSNR lab.
- `tools/ui_smoke.py`: drives the window on a file and screenshots it before, during and after the job.
- `tests/`: `python -m unittest discover -s tests`.

## Limits
- No jitter and no real depth means FSR 4 works mostly as a very good spatial upscaler with temporal
  stabilisation; it cannot reconstruct the sub-pixel detail it recovers in games.
- Input is processed as 8-bit sRGB; HDR video is not handled.
- The FSR DLLs are AMD's signed binaries under the FSR SDK license (`bin/license.md`), redistributable in
  binary form with that notice.
