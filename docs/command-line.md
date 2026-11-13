# Command line

`fsr4.exe` sits next to `Coccinelle Player.exe`. From the source tree, `python -m fsr4 ...` is the same
(`python -m fsr4 ui` for the window).

```
fsr4 upscale clip.mp4                       # clip.fsr4.mp4, 2x (performance mode), HEVC on the GPU, audio kept
fsr4 upscale photo.png --mode quality       # 1.5x
fsr4 upscale clip.mkv --scale 3 --codec x265 -o out.mp4
fsr4 upscale clip.mp4 --flow-preset ultrafast   # faster motion vectors, slightly lower quality
fsr4 upscale clip.mp4 --flow zero           # no motion vectors (no OpenCV needed)
fsr4 info                                   # GPU, extensions, FSR versions offered by the DLL
fsr4 selftest --validate                    # Vulkan round trip under the validation layer
```

`fsr4 upscale --help` lists every option.

## Options
- **Size:** scale is 1x to 3x (FSR's limit). The modes are native 1x, quality 1.5x, balanced 1.7x,
  performance 2x, ultra 3x; `--scale`, `--width` and `--height` override them.
- **Sharpening:** `--sharpness 0..1` enables FSR's RCAS pass (with `--upscaler hq`, the HQ scaler's own).
- **Encoder:** `amf` (HEVC on the GPU, default), `av1_amf`, `h264_amf`, `x264`, `x265`. The GPU ones need an
  AMD card, and without one the default is `x264`.
- **Upscaler:**
  - `--upscaler 3.1` or `2.3` uses AMD's older upscalers instead of FSR 4.
  - `--upscaler hq` is AMD's HQ scaler (ffmpeg 8.0 or later): faster, no motion vectors, each frame on its own.
  - `--upscaler realesrgan` is Real-ESRGAN (installed apart, see the [README](../README.md#requirements)): it
    redraws each frame, about 10 times slower than FSR 4, with no motion vectors and no sharpening.
  - `--upscaler realesrgan-x4plus` (general) and `realesrgan-x4plus-anime` (drawings) are its slower models,
    for still images only: a video is refused.

## Drag and drop on a script
In the source tree, drop one or more files on `upscale.bat`: each is upscaled 2x with no questions asked. Other
settings go through `fsr4 upscale`. Windows hands a dropped name to a script unquoted when it has no space, so
cmd cuts one holding `&`, `^`, `,`, `;` or `=`: open those in the window or with `fsr4 upscale`.

## Settings, all of them
FSR 4 itself takes the output size, the optional RCAS sharpening, and per frame a colour image, depth, motion
vectors, camera jitter, exposure and a reset flag. Depth, jitter and exposure have no meaning for a video, so
they are fixed ([How it works](how-it-works.md)); the motion vectors are the one input worth a choice (optical
flow presets or off).

The SDK's five tuning keys (velocity factor, reactiveness, shading change, accumulation per frame, minimum
disocclusion accumulation) change nothing in FSR 4: its output stayed identical to the bit when they were tried
here; they only act on FSR 3.1.

The HQ scaler takes the output size and sharpening; its algorithm stays Video SR 1.1, the best of the filter's
five. Real-ESRGAN takes only its model, offered as the Upscaler setting: the video model runs at the smallest
of its scales (2, 3, 4) that reaches the output size, the two for still images at 4x, then the picture is
resized to the output size.

The DLL also holds FSR 3.1.6 and 2.3.4, offered as the Upscaler setting to compare. Measured on the panning
fractal clip (PSNR against the original after a 2x downscale, DIS fast):

| Upscaler | PSNR | GPU time per frame (1600x900 to 3200x1800) |
|----------|------|---------------------------------------------|
| FSR 4.1.2 | 26.78 dB | 3.5 ms |
| FSR 2.3.4 | 26.54 dB | 2.4 ms |
| Lanczos | 26.37 dB | |
| FSR 3.1.6 | 25.92 dB (25.08 with velocity factor 0) | 2.5 ms |
