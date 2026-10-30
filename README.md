# FSR4 upscaler

Upscales an image or a video with **AMD FSR Upscaling 4** (the RDNA 4 machine-learning upscaler, FSR SDK 2.4,
upscaler 4.1.2) on a Radeon RX 9000 series card, from the command line or by dropping a file on `upscale.bat`.

FSR 4 is a game upscaler: it expects a low-resolution render plus depth, per-pixel motion vectors and a
sub-pixel camera jitter every frame. A video file has none of those, so this tool substitutes:

| FSR input      | What the tool feeds it                                                        |
|----------------|-------------------------------------------------------------------------------|
| Colour         | The decoded frame (sRGB 8-bit, declared as non-linear colour)                 |
| Motion vectors | DIS optical flow between consecutive frames (`--flow dis`), or zero            |
| Depth          | A constant plane                                                              |
| Jitter         | The frame resampled on FSR's Halton sequence (`--jitter fake`), or none        |
| Exposure       | FSR's auto exposure                                                           |

The FSR dispatch itself costs about 1.3 ms per 4K frame on an RX 9070 XT (AMD's figure); decoding, optical
flow on the CPU, the copies and encoding are what limit the frame rate.

## Requirements
- Windows, a Radeon RX 9000 (FSR 4 FP8 model) or RX 7000/6000 (INT8 model) with a current Adrenalin driver.
- Python 3.12 with numpy. `opencv-python` for optical flow and fake jitter (optional, strongly recommended).
- ffmpeg and ffprobe on PATH (or next to the exe).
- The FSR SDK DLLs in `bin/`: `python tools/fetch_sdk.py` downloads them from AMD's GitHub (tag v2.4.0).

## Use
```
python -m fsr4 upscale clip.mp4                 # clip.fsr4.mp4, 2x (performance mode), HEVC on the GPU
python -m fsr4 upscale photo.png --mode quality  # 1.5x
python -m fsr4 upscale clip.mkv --scale 3 --codec x265 -o out.mp4
python -m fsr4 upscale clip.mp4 --flow zero --jitter none   # without OpenCV
python -m fsr4 info                              # GPU, extensions, FSR versions offered by the DLL
python -m fsr4 selftest --validate               # Vulkan round trip under the validation layer
```
`start.bat` asks for a path; `upscale.bat` takes a dropped file. Scale is 1x to 3x (FSR's limit); the modes
are native 1x, quality 1.5x, balanced 1.7x, performance 2x, ultra 3x. Images are fed `--passes` times
(default 8) so the temporal accumulation converges.

## Layout
- `fsr4/vk.py`, `fsr4/vk_types.py`, `fsr4/gpu.py`: ctypes Vulkan binding, generated structs, device and copies.
- `fsr4/ffx.py`: the FSR API (`amd_fidelityfx_loader_vk.dll`) and the upscaling context.
- `fsr4/media.py`: ffmpeg frame reader and writer. `fsr4/flow.py`: motion vectors. `fsr4/pipeline.py`: a frame.
- `tools/gen_vk_types.py`: regenerates the Vulkan structs from `third_party/vulkan_core.h`.
- `tests/`: `python -m unittest discover -s tests`.

## Status
- Vulkan layer, FSR binding, ffmpeg I/O and the pipeline are written and unit-tested; the FSR DLLs and OpenCV
  are not installed yet, so the end-to-end run, the quality lab (PSNR per flow/jitter setting) and the
  packaged exe are pending.
