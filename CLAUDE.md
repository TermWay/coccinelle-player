# FSR4 upscaler — project rules

A command line tool (and a packaged exe) that upscales an image or a video with AMD FSR Upscaling 4 on a
Radeon RX 9000 card, through Vulkan. The shared rules in `..\CLAUDE.md` apply; this file adds the specifics.

## Stack
- Python 3.12, stdlib `ctypes` for Vulkan and the FSR API, numpy for pixels, ffmpeg/ffprobe subprocesses for
  all decoding and encoding. OpenCV (`opencv-python`) only for optical flow and the fake jitter resampling;
  everything must still run with `--flow zero --jitter none` when it is absent. The window needs OpenCV (it
  resizes and filters every preview frame with it); the command line does not.
- No C++ toolchain on this machine: the Vulkan structs in `fsr4/vk_types.py` are generated from the vendored
  `third_party/vulkan_core.h` by `tools/gen_vk_types.py`. Regenerate instead of editing by hand.
- The FSR binaries (`bin/*.dll`) are AMD's signed DLLs from the FidelityFX SDK tag pinned in
  `tools/fetch_sdk.py`; they are not committed. `bin/license.md` must ship with them.
- Branch: `main`. Commit at the end of every prompt that changed files.

## Checks
- `python -m unittest discover -s tests` is the test suite; `python -m fsr4 selftest --validate` exercises the
  Vulkan layer under the validation layer (Vulkan SDK 1.4.313 is installed; it does not know
  `VK_EXT_shader_float8`, so warnings about structure type 1000567000 are expected).
- Quality is measured, not eyeballed: `tools/lab_quality.py` downscales a reference, upscales it back and
  reports PSNR against the reference for each flow/jitter setting. Run it before changing a default.
- Performance figures go in the README table with the command that produced them.
- The window is checked with `python tools/ui_smoke.py --video <clip> --image <png> --out out/ui`: it drives
  every state through the real widgets (a real WM_DROPFILES drop, tooltip, dropdown, file dialog, playback,
  buffering, full screen, save) and saves PrintWindow screenshots to look at. The window pops up on the desktop
  while it runs; `out/ui/sample.wmv` (1080p60 WMV with WMA audio) is the usual clip.
- `tests/test_render.py` (GPU) checks that pieces rendered around seeks join into exactly the single-pass video;
  `tests/test_audio.py` needs a sound device and skips without one.
- `tests/test_formats.py` builds one small file per container/codec family with ffmpeg and round-trips it;
  add a case there when a new kind of file misbehaves.

## Design notes
- FSR wants jittered, motion-vectored, depth-tested renders. For video the tool substitutes: depth is a
  constant plane, motion vectors come from DIS optical flow (current to previous, in pixels, which is FSR's
  convention, verified by the lab's negated run), jitter is zero. Faking jitter by resampling the input and
  feeding a still image several times were both measured and rejected (README). Keep the substitutions in
  `fsr4/pipeline.py` and `fsr4/flow.py`.
- Motion vectors are uploaded as 32-bit floats: converting to half floats on the CPU cost more than the flow.
- `fsr4/pipeline.py`'s `upscale_stream` is the one frame loop (flow on a worker thread while the current frame
  is on the GPU; the frame reader double-buffers for that). `fsr4/job.py` (CLI, images) and `fsr4/render.py`
  (the window's videos) both use it. Keep `Pipeline.process` synchronous and thread-free, and keep the window
  free of pipeline logic.
- Videos in the window: `Renderer` (one thread, owns the GPU) writes pieces, `Timeline` (pure, tested) decides
  where to work, `Player` (its own thread) plays ready pieces with the waveOut clock, `VideoView` draws. They talk
  through the window's queue; only `VideoView.present/show_still/set_spinner` may be called off the Tk thread.
- A settings change starts a new `Version` (settings + timeline); the previous one is kept as `before` (if it has
  a second or more) for the comparison layers, and settings equal to it swap the two back. Each side of the
  view shows a layer: "current", "before" or a filter of the source; the player decodes only the layers shown,
  and only "current" paces playback (buffering).
- FSR 4 ignores the SDK's upscale tuning keys (verified bit-identical); `ffx.Upscaler(provider=...)` selects the
  DLL's FSR 3.1.6 or 2.3.4 through `ffxOverrideVersion` for comparisons. The frame count from the probe is an
  estimate until a read reaches the end (`Timeline.exact`); never stop rendering at the estimate.
- Frame numbers must be exact across seeks: `FrameReader` reads on a constant-rate grid anchored at the file
  start (`setpts`, `fps`, `trim` with `-copyts`), verifies the first frame's timestamp, and needs a
  `KeyframeIndex` for containers without an index (TS, PS, FLV, Ogg). Never seek with plain `-ss` and a count.
- The preview is drawn with GDI (`SetDIBitsToDevice` on a background-less Tk frame), not Tk images, which are
  10-60 times slower. Everything on it (labels, divider, minimap, spinner) is composed in numpy from sprites
  (`Sketch`, GDI text); keep per-frame work allocation-light and under a few milliseconds.
- The window is plain tkinter (stdlib, DPI aware) with a custom dark ttk theme in `fsr4/ui_theme.py`: controls
  are 9-slice image elements drawn by `fsr4/ui_art.py`. Image elements composite over the style's background,
  so every style sets `background` (and maps it for disabled/active/pressed) to its container's colour, and
  sets the element `padding=0` so the style padding alone sizes it. Sizes go through `Theme.px` (DPI).
- All package output goes through `fsr4/log.py` so the window can show it.
- Background work (renderer, its ffmpeg processes, thumbnails) runs at the `fsr4/usage.py` level's priority and
  thread caps; playback never does. x265 takes its thread cap through `-x265-params pools=`, not `-threads`.
- Media rules (`fsr4/media.py`): pick the first video stream that is not an attached picture; decode untagged
  HD as BT.709; read video at a constant rate; encode with BT.709 tags via `setparams` (ffmpeg takes encoder
  tags from the frames); copy only MP4-safe audio (`MP4_AUDIO_COPY`), re-encode the rest to AAC; drain
  ffmpeg's stderr on a thread. GPU encoders have frame size limits (`ENCODER_LIMITS`, checked when planning).
- Resource states told to FSR must match the Vulkan layouts the images are really in (inputs
  `SHADER_READ_ONLY_OPTIMAL`, output `GENERAL`); `Gpu` tracks layouts, use its `cmd_*` helpers.
