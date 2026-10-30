# FSR4 upscaler — project rules

A command line tool (and a packaged exe) that upscales an image or a video with AMD FSR Upscaling 4 on a
Radeon RX 9000 card, through Vulkan. The shared rules in `..\CLAUDE.md` apply; this file adds the specifics.

## Stack
- Python 3.12, stdlib `ctypes` for Vulkan and the FSR API, numpy for pixels, ffmpeg/ffprobe subprocesses for
  all decoding and encoding. OpenCV (`opencv-python`) only for optical flow and the fake jitter resampling;
  everything must still run with `--flow zero --jitter none` when it is absent.
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
  every state through the real widgets (including a real WM_DROPFILES drop, the dropdown and the file dialog)
  and saves PrintWindow screenshots to look at. The window pops up on the desktop while it runs.
- `tests/test_formats.py` builds one small file per container/codec family with ffmpeg and round-trips it;
  add a case there when a new kind of file misbehaves.

## Design notes
- FSR wants jittered, motion-vectored, depth-tested renders. For video the tool substitutes: depth is a
  constant plane, motion vectors come from DIS optical flow (current to previous, in pixels, which is FSR's
  convention, verified by the lab's negated run), jitter is zero. Faking jitter by resampling the input and
  feeding a still image several times were both measured and rejected (README). Keep the substitutions in
  `fsr4/pipeline.py` and `fsr4/flow.py`.
- Motion vectors are uploaded as 32-bit floats: converting to half floats on the CPU cost more than the flow.
- `fsr4/job.py` owns the run loop (flow on a worker thread while the current frame is on the GPU; the frame
  reader double-buffers for that) and is the only thing the CLI and the window call. Keep `Pipeline.process`
  synchronous and thread-free, and keep the window free of pipeline logic.
- The window is plain tkinter (stdlib, DPI aware) with a custom dark ttk theme in `fsr4/ui_theme.py`: controls
  are 9-slice image elements drawn by `fsr4/ui_art.py`. Image elements composite over the style's background,
  so every style sets `background` (and maps it for disabled/active/pressed) to its container's colour, and
  sets the element `padding=0` so the style padding alone sizes it. Sizes go through `Theme.px` (DPI).
- All package output goes through `fsr4/log.py` so the window can show it. Previews come from the job's
  `preview` callback (copies of the input and output frames, twice a second), never from re-reading files.
- Media rules (`fsr4/media.py`): pick the first video stream that is not an attached picture; decode untagged
  HD as BT.709; read video at a constant rate; encode with BT.709 tags via `setparams` (ffmpeg takes encoder
  tags from the frames); copy only MP4-safe audio (`MP4_AUDIO_COPY`), re-encode the rest to AAC; drain
  ffmpeg's stderr on a thread.
- Resource states told to FSR must match the Vulkan layouts the images are really in (inputs
  `SHADER_READ_ONLY_OPTIMAL`, output `GENERAL`); `Gpu` tracks layouts, use its `cmd_*` helpers.
