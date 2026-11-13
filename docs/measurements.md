# Measurements

Measured on the development PC (an RX 9070 XT and a 16-core processor) with clips that are not in the
repository: the commands repeat them on yours.

## Speed
End to end (decode, flow, upscale, encode), `--flow dis` with the default `fast` preset, HEVC AMF output:

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

## Encoders
A natural-footage clip downscaled to 1440×1080 and a 1080p60 screen recording; upscaled 2×, then the same FSR
frames encoded by each; frames taken per second, bitrate, PSNR against the lossless FSR output:

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

## Processor settings
1080p60 to 4K. Processor time is the share of all 32 threads of a 16-core CPU; Task Manager's CPU column shows
"% Processor Utility", which counts boosted clock speed and reads much higher; the last column is how late a
10 ms timer of the window fires, its responsiveness.

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

## Quality
The lab (`python tools/lab_quality.py`) downscales a reference by 2x with an area filter, upscales it back and
reports PSNR against the original:

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

## AMD's HQ scaler
AMD's HQ scaler (ffmpeg's `sr_amf`, Video SR 1.1, default sharpness; `--upscaler hq`) works on one frame at a
time. At 2880x2160 it takes 9 ms a frame against 17 ms for FSR 4 with DIS fast flow; a whole job at 4K runs 1.5
times as fast as FSR 4 with motion and 1.2 times as fast without. It wins on the smooth still, is level with
FSR 4 on natural footage, and loses 0.5 dB on the panning clip, where FSR 4 gathers detail across frames.
Lanczos leads on natural footage only because PSNR rewards softness: on one frame it keeps 43% of the
original's detail energy, the HQ scaler 99%, FSR 4 112%. Video SR 1.0 scores a little higher on stills (43.81)
but leaves stair steps on diagonal edges.

## Real-ESRGAN
Real-ESRGAN (`realesrgan-ncnn-vulkan`, its video model; `--upscaler realesrgan`) is there for those who want
its look. It looks much crisper and removes compression artifacts, but it repaints the picture rather than
recovering it: on one frame of natural footage, 4.5 times the original's detail energy and SSIM 0.979 against
0.995 for FSR 4. It comes out 2 to 3 levels brighter, and on small text it draws clean letters that are the
wrong letters. It takes 160 ms a frame at 2880x2160 (FSR 4: 19 ms).

On a drawn clip (made for the lab: flat colours, black outlines, a panning background) it scores below the HQ
scaler when the file is clean, and first of the three once H.264 at CRF 26 has marked it up, because it wipes
the artifacts; it also flickers least there (temporal PSNR 46.8, HQ scaler 45.9, FSR 4 41.0).

The program only takes files and each run costs 0.34 s to start, so frames go through it in chunks of up to 16
PNG files in a temporary folder beside the output (about 400 MB at 4K), two runs overlapping. A seek waits for
the chunk being read, up to about 3 s at 4K.

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

## HDR and interlaced sources
An HDR video made from an SDR picture (a test pattern beside a fractal, its white at 203 nits as BT.2408 places
it, 10-bit HEVC), then decoded back to SDR, PSNR against the SDR picture:

| Decoding                                   | PQ       | HLG      | Colourfulness (SDR picture: 190) |
|--------------------------------------------|----------|----------|----------------------------------|
| As if it were SDR (before)                 | 9.99 dB  | 11.96 dB | 50 and 83: washed out            |
| Möbius curve, white at 100 nits (chosen)   | 20.97 dB | 21.14 dB | 183                              |
| Reinhard, white at 100 nits                | 20.80 dB | 20.97 dB | 173                              |
| Hable, white at 100 nits                   | 16.62 dB | 16.67 dB | 143: darker                      |

Tone mapping costs 1080p HEVC decoding 199 to 154 frames a second, 4K 21 to 19. Deinterlacing (bwdif) a test pattern
scrolling sideways, woven into interlaced MPEG-2 and H.264: 19.0 dB against the progressive pictures as decoded,
23.7 dB deinterlaced (`tests/test_formats.py` builds both kinds of file).
