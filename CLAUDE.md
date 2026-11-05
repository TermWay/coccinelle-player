# Coccinelle Player — project rules

A desktop window and a command line tool (packaged as two exes) that upscale an image or a video with AMD FSR
Upscaling 4 on a Radeon RX 9000 card, through Vulkan. The window is Coccinelle Player; the command line and the
Python package stay `fsr4`.

## Working here
- Ask before adding any package, runtime or dev-only; the tested versions are pinned in `requirements.txt`
  (`requirements-build.txt` adds PyInstaller).
- Measure and look before calling a change done: the tests, the lab, screenshots read back.
- KISS and DRY; no silent failures (log the error and fail early); descriptive names; docstrings on public
  classes and functions; no allocations in per-frame code.
- `SPEC.md` is the public statement of what the tool must do, standing in for the requests behind it: when a
  request changes a requirement (a behavior, a default, a limit, a format), update it in the same commit.

## Stack
- Python 3.12, stdlib `ctypes` for Vulkan and the FSR API, numpy for pixels, ffmpeg/ffprobe subprocesses for
  all decoding and encoding (ffmpeg 5.1 or later, 6.0 for `av1_amf`, 8.0 for `sr_amf`; not bundled: it is GPL). OpenCV
  (`opencv-python`) for optical flow; the command line must still run with `--flow zero` when it is absent.
  The window needs OpenCV (it resizes and filters every preview frame with it).
- No C++ toolchain on this machine: the Vulkan structs in `fsr4/vk_types.py` are generated from the vendored
  `third_party/vulkan_core.h` by `tools/gen_vk_types.py`. Regenerate instead of editing by hand.
- The FSR binaries (`bin/*.dll`) are AMD's signed DLLs from the FidelityFX SDK tag pinned in
  `tools/fetch_sdk.py`; they are not committed. `bin/license.md` must ship with them.
- Branch: `main`. Commit at the end of every prompt that changed files; never push unless asked. The
  repository is public, so commit messages carry no `Request:` line (this replaces the shared rule): once the
  commit exists, the request goes into `requests/log.md` (git-ignored, never published) under a
  `## <short hash> · <date> · <subject>` heading, quoted as the shared rule says.

## Checks
- `python -m unittest discover -s tests` is the test suite; `python -m fsr4 selftest --validate` exercises the
  Vulkan layer under the validation layer (Vulkan SDK 1.4.313 is installed; it does not know
  `VK_EXT_shader_float8`, so warnings about structure type 1000567000 are expected). Tests that need ffmpeg,
  OpenCV or FSR 4 itself skip without it, saying why (`tests/needs.py`).
- Quality is measured, not eyeballed: `tools/lab_quality.py` downscales a reference, upscales it back and
  reports PSNR against the reference for each flow/jitter setting. Run it before changing a default.
- Performance figures go in the README table with the command that produced them.
- The window is checked with `python tools/ui_smoke.py --video <clip> --image <png> --out out/ui`: it drives
  every state through the real widgets (a real WM_DROPFILES drop, tooltip, dropdown, file dialog, playback,
  buffering, presets side by side, zoom slider, full screen, save) and saves PrintWindow screenshots to look at.
  Its presets go to a scratch file in `--out`, never the user's `%APPDATA%` one. The window pops up on the desktop
  while it runs; the usual clip is a 1080p60 WMV with WMA audio. The saved outputs it makes are deleted once
  checked (a default output name never replaces a file, so each run would add one).
- `python tools/ui_flicker.py --video <clip> --frames out/ui/flicker` samples the screen on every composed frame
  while the sidebar folds, a repaint runs and the window changes size: stray pixels, blinks, how far what follows
  the right edge lags. PrintWindow never shows these. The window pops up and must be left alone: the run refuses if
  someone folds the sidebar or another window comes over it. It cannot drive a real mouse drag: Tk runs its events
  inside window messages then, and a ctypes call that triggers that crashes Python (no thread state to come back to).
- `tests/test_render.py` (GPU) checks that pieces rendered around seeks join into exactly the single-pass video;
  `tests/test_audio.py` needs a sound device and skips without one.
- `tests/test_formats.py` builds one small file per container/codec family with ffmpeg and round-trips it;
  add a case there when a new kind of file misbehaves. Never make Theora samples with this ffmpeg: its
  libtheora writes frames its own decoder rejects (Ogg samples carry VP8; real Theora files decode cleanly).

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
- Every distinct `RenderSettings` is a `Version` (settings + timeline). The current settings' version is what
  Upscale saves; `Renderer.show()` names the versions the view shows (the current settings or presets, see
  `fsr4/presets.py`), which preview mode upscales in turns (fewest ready ahead first, yield after
  `SWITCH_SECONDS`); batch mode does only the current one. Versions not shown are kept up to `KEPT_VERSIONS`.
  A preset holds only upscaler, motion and sharpening: size and encoder always come from the current settings.
- Each side of the view shows a filter of the source or an upscaled layer (keyed by its settings, so two sides
  showing the same version share it). The player decodes only the layers shown; layers being upscaled pace
  playback (buffering), the others (a preset during a whole-file upscale) show what they have. Images get an
  `ImagePreviewer` (own GPU device) for the layers shown; saving an image runs the job.
- The side dropdowns, the Preset dropdown and every other choice are the same themed ttk Combobox; no native
  `tk.Menu` (it cannot be styled to match). A label on the picture opens its side's dropdown on release (on
  press, the release would land in the list and pick an entry).
- Full screen goes through `fsr4/ui_window.py` (frame styles off, the window's own monitor covered, placement
  restored), never Tk's `-fullscreen`, which always takes the primary monitor and rebuilds the frame window.
  After either switch everything is repainted. The sidebar keeps its scrollbar column reserved: a canvas window
  does not always follow its canvas back to a narrower width.
- Playing speed: the sound card is the clock from 0.25x to 2x (atempo) and late frames are dropped; slower,
  the system clock runs and late frames are still dropped; faster than 2x and backwards every frame is shown
  in order, the clock re-anchored when a decoder is late. `ui.REVERSE_AHEAD_SECONDS` must stay above
  `player.BUFFER_MAX_SECONDS` (tested), or a reverse stall never ends. `_Stream.put_back` returns frames taken for a tick that cannot be shown yet: in
  frame-by-frame mode a released frame would never come back. Backwards, `ReverseFrameStream` decodes chunks
  and the window sets the renderer's target just behind the ready stretch behind the playhead.
- After full screen, or once the window has moved or changed size, it repaints every widget through Windows
  (`ui_window.repaint`) and through Tk (an Expose each), `REPAINT_SETTLE_MS` later (and at once for full screen).
  Maximizing used to lose what the widgets had drawn (black until redrawn, 1 to 2 s with slow element redraws, see
  the theme note); now it measures no black pixel. A whole repaint takes about 100 ms and shows no change. Measure
  repaints with a probe that counts black pixels on screen from another thread (keep no frames: Windows' maximize
  animation draws other things there).
- Tk registers its window classes with CS_HREDRAW and CS_VREDRAW: each size change repainted every widget once per
  level of nesting (390 Expose events, 360 ms a step). `ui_window.keep_contents_on_resize` takes them off (8 events,
  40 ms); the frame's class can only be reached once Tk has created the frame, at the first map. The root's Configure
  runs `update_idletasks`, or each level of nesting waits for its own idle pass and the right side trails the edge.
- Tk moves a parent window first and its children in later passes, and Windows shows every stage (the right side of
  the toolbar rode 340 px away with `main` for 5 frames). Folding the sidebar and toggling the log change the layout
  behind a `ui_window.Curtain`: a layered popup with the window's old picture, faded once the new layout is drawn
  (two idle passes later: the paints are idle callbacks too). Copying the screen needs CAPTUREBLT to see it.
- While playing, a layer whose decoder still works at the previous zoom or position is drawn with `reframe`
  (its picture resampled to where it belongs), never stretched: both sides of the divider must move together.
- `tools/window_capture.py` can capture what the screen shows (`screen=True`; `settle=False` to see paints Tk
  has not done yet) but it refuses when another program's window overlaps ours, and PrintWindow captures
  only this process's windows: never capture other windows' content.
  Probes that drive a window find it by the process they started, never by its title (the user may have the
  app open).
- AMD's HQ scaler (`fsr4/hq_scaler.py`) upscales inside the encoder's ffmpeg: its 4K frames through a pipe cost
  62 ms, more than FSR 4's whole frame. `HqPipeline` stands in for `Pipeline` in `upscale_stream` and hands frames on
  unchanged, so the renderer has no upscaled frame of it: no "live" event, thumbnails from the source. `sr_amf` and an
  AMF encoder deadlock at random on one shared device: `DEVICE_ARGUMENTS` declares a second, unused device, so ffmpeg
  gives the encoder none and it opens its own. Never reduce that to one device.
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
  Tk draws an image element as tiles, one GDI call each, plus a read back and a blend each when the image has
  partly transparent pixels: so images are opaque over their container's colour (`box`/`photo` take it) and
  9-slice middles are repeated to `SLICE_LENGTH`, the element keeping its small requested size (`box_extent`,
  `width=`). Only pieces drawn over another element (scale knob, progress bar, combobox arrow) stay
  transparent. A field went from 30 ms to 1.5 ms per redraw; keep new elements that way.
- All package output goes through `fsr4/log.py` so the window can show it.
- Background work (renderer, its ffmpeg processes, thumbnails) runs at the `fsr4/usage.py` level's priority and
  thread caps; playback never does. x265 takes its thread cap through `-x265-params pools=`, not `-threads`.
- Media rules (`fsr4/media.py`): pick the first video stream that is not an attached picture; decode untagged
  HD as BT.709; read video at a constant rate; encode with BT.709 tags via `setparams` (ffmpeg takes encoder
  tags from the frames); copy only MP4-safe audio (`MP4_AUDIO_COPY`), re-encode the rest to AAC; drain
  ffmpeg's stderr on a thread. GPU encoders have frame size limits (`ENCODER_LIMITS`, checked when planning).
  ffmpeg skips pictures it cannot decode and still succeeds (the timeline repeats the one before): readers run
  it with `-v level+...` and log the video decoder's first error once per file, for reads from the start
  (after a seek, the pictures before the keyframe fail by design).
- Anamorphic sources (a sample aspect ratio other than 1:1, as on DVDs) come out at the shape players show,
  with square pixels: FSR scales each axis on its own (`job.output_size`, measured as good as one scale), and
  the preview follows from the output size. Images are always taken as square; a tag outside 1:4 to 4:1 is
  ignored and logged.
- Resource states told to FSR must match the Vulkan layouts the images are really in (inputs
  `SHADER_READ_ONLY_OPTIMAL`, output `GENERAL`); `Gpu` tracks layouts, use its `cmd_*` helpers.
