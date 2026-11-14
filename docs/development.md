# Development

`SPEC.md` states what the tool must do, and `CLAUDE.md` holds the design notes for working on the code.

## Run from the source
- Python 3.12 and `python -m pip install -r requirements.txt`: numpy, and `opencv-python` for the window and
  the optical flow (without OpenCV the command line still runs with `--flow zero`).
- The FSR DLLs in `bin/`: `python tools/fetch_sdk.py` downloads them from AMD's GitHub (tag v2.4.0) and checks
  their pinned SHA-256 (`--check` only checks).
- ffmpeg and ffprobe on PATH, as for the packaged app.

`python -m fsr4 ui` opens the window and `python -m fsr4 upscale ...` is the command line. `start.bat` and
`upscale.bat` use the executables in `dist\fsr4\` when they exist, else Python.

## Build the executables
`build_exe.bat` (after `python -m pip install -r requirements-build.txt`) checks the FSR DLLs against their
pinned hashes and builds the executables with PyInstaller from `fsr4.spec`: `Coccinelle Player.exe` (window, no
console) and `fsr4.exe` (command line) share one folder of about 170 MB (85 MB zipped), mostly OpenCV and
numpy, with the FSR DLLs and AMD's license in `bin\`, the third-party notices in `licenses\`, these pages in
`docs\` and the install scripts for ffmpeg and Real-ESRGAN (`install-*.bat`). Its two optional arguments build elsewhere: the folder that receives `fsr4\` and PyInstaller's work
folder.

Before opening the window, `start.bat` runs `tools/dev_build.py`, which rebuilds the executables when a source
is newer than the build: into `out\dev-build\`, then in place of `dist\fsr4\` once no program runs from there
(until then the window opens from the sources, and a failed build does the same).

## Releases
Release zips are built by GitHub Actions, never on a personal computer: [release.yml](../.github/workflows/release.yml).

1. Set the version in `fsr4/__init__.py` and commit.
2. Push a tag of that version: `git tag -a v0.1.0 -m "Coccinelle Player 0.1.0"`, then `git push origin v0.1.0`.
3. The workflow refuses a tag that does not match the version, builds the executables on a Windows runner,
   checks that `fsr4.exe --version` runs, zips them with `tools/release_zip.py`, writes `SHA256SUMS.txt`, attests
   the zip's provenance and attaches both files to a draft release of the tag.
4. Read the draft, add the release notes and publish it.

"Run workflow" on the Actions tab builds the same files from any branch as an artifact of the run, with no
release. Attestations work only while the repository is public; a private one gets a warning instead.

## Tests and checks
- `python -m unittest discover -s tests`. Tests that need ffmpeg, OpenCV or FSR 4 itself (the card and the
  DLLs in `bin/`) skip without it, saying why; `tests/test_audio.py` also needs a sound device.
- `python -m fsr4 selftest --validate` exercises the Vulkan layer under the validation layer.
- `python tools/lab_quality.py` is the PSNR lab behind [Measurements](measurements.md#quality).
- `tools/ui_smoke.py` drives the window through every state (drop, first upscaled frame, tooltip, dropdown,
  file dialog, playback, the seek bar's hover label, buffering after a seek, zoom with original pixels, the
  zoom slider (a drag that must keep the picture still) and wheel, a side's dropdown opened from its label, a
  preset saved and two presets side by side paused and playing, a frame step, playing backwards, a side turned
  off, full screen on the window's own monitor (the second one when there is one) with its controls showing
  near the bottom and hiding at rest, then back where it was, folded sidebar, whole-file save, an image
  compared with a preset, error, a video opened without upscaling, minimum size) and screenshots each with
  `tools/window_capture.py`. Its presets and preferences go to scratch files, never yours.
- `tools/ui_record.py` records the window for the README's animation, from clips that may be published:
  `python tools/ui_record.py --video <first clip> --video <second clip> --out out/readme/demo.mkv`, then
  `ffmpeg -i out/readme/demo.mkv -vf fps=15 -c:v libwebp_anim -q:v 70 -compression_level 6 -loop 0 docs/demo.webp`.

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
- `tools/fetch_sdk.py`: downloads the FSR DLLs and checks their pinned hashes. `tools/release_zip.py`: the
  release zip and its `SHA256SUMS.txt` (run by the release workflow).
  `tools/collect_licenses.py`: the packaged app's third-party notices (run by `build_exe.bat`).
  `tools/dev_build.py`: keeps `dist\fsr4\` built from the current sources without touching a running copy
  (`--force` builds anyway; run by `start.bat`).
