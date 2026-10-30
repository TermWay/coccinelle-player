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

## Design notes
- FSR wants jittered, motion-vectored, depth-tested renders. For video the tool fakes them: depth is a constant
  plane, motion vectors come from DIS optical flow (current to previous, in pixels, which is FSR's convention),
  and "fake jitter" resamples the input on FSR's Halton offsets. Keep these three substitutions isolated in
  `fsr4/pipeline.py` and `fsr4/flow.py`.
- Resource states told to FSR must match the Vulkan layouts the images are really in (inputs
  `SHADER_READ_ONLY_OPTIMAL`, output `GENERAL`); `Gpu` tracks layouts, use its `cmd_*` helpers.
