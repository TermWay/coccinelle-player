# Coccinelle Player: specification

What the tool must do, written as requirements. The tool was built from a series of requests; this file states
where all of them led, so that a developer or a coding agent given it builds the same tool. Figures marked
*measured* come from the reference machine (§2) and are targets to meet. README.md is the user's guide and
CLAUDE.md the design notes for working on the code.

## 1. Goal
- Upscale a video or an image with AMD FSR Upscaling 4, the machine-learning upscaler AMD makes for games on
  Radeon RX 9000 cards, outside any game.
- Three ways in: a desktop window (the main one), a command line, and files dropped on a script.
- Saving need not be real time. Watching must be: the window plays the video upscaled while it is being made,
  with its sound, and waits like a streaming site when the upscaler is slower than the video.
- Everything must be comparable: any two versions side by side (FSR settings, AMD's older upscalers, the source
  enlarged by a classic filter), paused or playing, at any zoom down to single pixels.
- Every setting explains itself where it is; nothing needs the documentation to be understood.

## 2. Platforms and reference machine
- Windows, tested on Windows 11. A Radeon RX 9000 card (FSR 4's FP8 model) with a current Adrenalin driver. AMD
  also lists the RX 7000 series (FSR 4's INT8 model): not refused, not tested.
- Linux: not supported yet (AMD ships FSR 4 only as signed Windows DLLs, and the window draws, plays sound, takes
  dropped files and goes full screen through Win32). macOS: not supported. The README says both.
- Reference machine: Radeon RX 9070 XT, a 16-core (32-thread) processor, Windows 11.

## 3. Constraints
- Python 3.12. Vulkan and the FSR API are called through the standard library's `ctypes`; no compiled extension
  and no C or C++ toolchain. The Vulkan structures are generated from a vendored `vulkan_core.h` by a script and
  never edited by hand; tests pin them and the FSR API structures to the MSVC x64 layout.
- Vulkan 1.3, loaded on first use. The GPU is chosen automatically (a discrete card first, then AMD's, then the
  first) or by index; it needs a queue with graphics and compute. Cooperative matrices and 8-bit floats are
  enabled when offered.
- Packages: numpy, and opencv-python for the optical flow and the window's preview filters, at pinned versions;
  PyInstaller to package. Nothing else without the maintainer's agreement. Without OpenCV the command line still
  upscales images, and videos with the motion vectors off; the window requires it.
- FSR: AMD's signed DLLs from the FidelityFX SDK, tag v2.4.0 (FSR Upscaling 4.1.2), downloaded by a script from
  AMD's GitHub repository and checked against pinned SHA-256 hashes; never committed. AMD's license file always
  travels with them. They are looked for in the folder given (which must then hold them), else next to the
  packaged executable, else in the repository's `bin\`.
- ffmpeg and ffprobe, 5.1 or later (6.0 for the AV1 GPU encoder, 8.0 for AMD's HQ scaler), do all decoding and encoding as separate
  processes, found next to the executable or on PATH. They are never bundled (GPL). A missing one is named with
  how to install it (`winget install Gyan.FFmpeg.Essentials`); when ffmpeg rejects an option as unknown, the error
  says this ffmpeg is too old and which version is needed.
- Real-ESRGAN is optional and never bundled: `realesrgan-ncnn-vulkan.exe` with its `models` folder, tested with
  the release of 2022-04-24 (v0.2.5.0), looked for in `bin\realesrgan\` next to the packaged executable or in a
  folder above it (a build in the repository's `dist` finds the repository's), then in the repository's, then on
  PATH. Without it the error says where to get it and names the folder to unzip it into.
- The window is plain tkinter (DPI aware) with ctypes calls into Win32; no GUI package.
- Every message goes through one logging module, which the window shows. No log file is ever written.
- No silent failure: an error is logged and stops the work with a plain reason on one line.

## 4. Feeding FSR 4 a video
FSR expects a low-resolution render with depth, per-pixel motion vectors and a sub-pixel camera jitter. A video
has none of those; the tool substitutes, as the quality lab (§14) settled:
- Colour: the decoded frame, 8-bit sRGB, declared to FSR as non-linear.
- Motion vectors: DIS optical flow (OpenCV, on grayscale) from the current frame to the previous one, in input
  pixels, which is FSR's convention (the lab run with the flow negated must score lower); zero on the first frame
  of a run. Presets ultrafast, fast (default) and medium, optionally computed at 1/N of the size, or none.
  Uploaded as 32-bit floats: converting to half floats on the processor cost more than the flow. The next
  frame's flow runs on a worker thread while the current frame is on the GPU. It gets at most 8 threads, in the
  window and on the command line (47 fps against 39 with 32 threads, *measured*).
- Depth: a constant plane. Jitter: none. Exposure: FSR's auto exposure. Camera values: fixed. Reset: on the first
  frame of each run; there is no scene-cut detection.
- Not done, because the lab measured them worse: faking jitter by resampling the input (2 to 8 dB lower PSNR),
  and feeding a still image several times (no gain). An image gets one pass with no motion.
- FSR 4 ignores the SDK's five upscale tuning keys (its output stayed identical to the bit): they are not
  offered. The upscaler is FSR 4 by default; FSR 3.1.6 and 2.3.4, from the same DLL through its version override,
  are offered to compare against. A version the DLL lacks is an error naming those it has.
- AMD's HQ scaler (ffmpeg's `sr_amf`, Video SR 1.1) is offered beside them: faster, with no motion vectors and no
  history, each frame on its own. It runs in the encoder's ffmpeg, so a video's frames reach it at their own size
  and only the encoded file holds them upscaled. It needs an ffmpeg that has the filter (without one, an error
  says so), takes at least 32 × 32, and refuses `--debug-view`, `--debug-checking`, `--validate` and `--gpu`.
- Real-ESRGAN (its `realesr-animevideov3` model) is offered too: a neural network that redraws each frame on its own,
  crisper but not the same picture, and about 10 times slower than FSR 4. It runs at the smallest of its scales (2, 3,
  4) that reaches the output size, then the result is resized to it (Lanczos). Frames go through its program in
  chunks of PNG files in a temporary folder beside the output (§10), removed afterwards: 4, then 8, then 16 frames,
  two runs overlapping, so a picture comes about 2.5 s after a seek at 4K; a seek or a stop waits for the chunk
  being read, up to about 3 s. A program that fails or leaves frames out is an error quoting its output. It refuses
  `--sharpness`, `--debug-view`, `--debug-checking`, `--validate` and `--gpu`; it picks the discrete card itself.
- Two more Real-ESRGAN models are offered for still images only, since a 1080p picture takes them seconds (*measured*:
  `realesrgan-x4plus`, the general one, 10 s; `realesrgan-x4plus-anime`, for drawings, 4.3 s; the video model 1.8 s). A
  video is refused with a plain reason. They run at 4x and the picture comes down to the output size with an area
  filter. A picture being made stops at once when the job is cancelled, when no side shows it any more and when the
  file closes.
- Sharpening: strength 0 to 1, off by default: FSR's RCAS pass, or the HQ scaler's own, where off and 0 keep the
  source's detail and 1 is the filter's strongest. Real-ESRGAN has none.
- Each frame is uploaded, dispatched and read back through cached host memory (93 ms to 1.4 ms, *measured*). The
  resource states told to FSR match the layouts the images are really in; `selftest --validate` runs the Vulkan
  path under the validation layer without errors.

## 5. Files in
- Image or video is decided by the suffix. Images: `.png .jpg .jpeg .bmp .tif .tiff .webp .exr .gif` (a GIF
  gives its first frame). Anything else is probed as a video: anything ffmpeg decodes, at least MP4, MKV, MOV,
  AVI, WMV/ASF, WebM, MPEG-1/2 (MPG, VOB, TS, M2TS), FLV, 3GP, Ogg (Theora or VP8 with Vorbis or Opus), MXF, DV,
  RealMedia and Y4M. The window's open dialog lists `.mp4 .m4v .mov .mkv .webm .avi .wmv .asf .flv .f4v .mpg
  .mpeg .m2v .vob .ts .m2ts .mts .3gp .3g2 .ogg .ogv .ogm .mxf .dv .rm .rmvb .divx .y4m` and all files.
- The video upscaled is the first video stream that is not an attached picture (WMV files from Windows Movie
  Maker and many MP4 and MKV files carry a JPEG cover as a video stream). Refused, with the reason: a file
  without one ("no video stream in <file> (only sound)", "(only a cover picture)"), a file ffmpeg cannot read
  (with ffmpeg's own reason), a missing file.
- Colour: YUV is decoded with the file's own matrix, or BT.709 when untagged and at least 720 pixels both ways,
  as players assume.
- Frame rate: ffmpeg's real rate, or the average one when the real rate is implausible (above 210 with an average
  below 70); 30 when neither is known. Video is read at that constant rate on a grid anchored at the file's
  start, so variable-rate sources (phones, screen recorders) stay in sync with their audio; a video starting
  after its audio is padded at the start, as players show it.
- Exact frames: a read starting anywhere returns exactly the frames a read from the start numbers there, in every
  container. MP4/MOV, Matroska/WebM, ASF and AVI seek with their own index; the others (MPEG-TS, MPEG-PS, FLV,
  Ogg...) have their keyframes indexed first. A plain seek followed by a frame count is never used.
- Rotation: portrait phone videos are turned upright, the output size swapped accordingly.
- Anamorphic video (pixels stored wider or narrower than square: DVD, DV, HDV) comes out at the shape players
  show, with square pixels (§7). An aspect tag outside 1:4 to 4:1 is ignored and logged. Images are always taken
  as square.
- Damaged files: pictures ffmpeg cannot decode are reported once per file in the log; the picture before stands
  in for each, as players do.
- The frame count from the probe is an estimate until a read reaches the end; work never stops at the estimate.
- Out of scope: HDR and more than 8 bits (input is processed as 8-bit sRGB), deinterlacing. Decoding runs on the
  processor.

## 6. Files out
- Default name: `<name>.fsr4.mp4` for a video, `<name>.fsr4.<ext>` for an image in its own format, next to the
  source. The output is never the input, under any spelling of its path ("the output is the input file <name>;
  choose another output name").
- It is written as `<name>.partial<ext>` and moved into place once complete; a failed run deletes the partial
  file. If the move fails (the old file is open elsewhere), the error says so and the result is kept as the
  partial file.
- The command line replaces an existing output, as ffmpeg's `-y` does. The window never does unasked: its default
  name is numbered (`<name>.fsr4 (2).mp4`, `(3)`...) and it asks before replacing a name the user typed.
- Video: the container follows the output's suffix, MP4 by default (the window offers MP4, MKV and MOV). 4:2:0,
  TV range, the source's frame rate. Outputs 720 lines high or more are BT.709, smaller ones BT.601 (tagged
  smpte170m), tagged on the frames and the stream: a round trip stays within 2 levels, where untagged BT.601 had
  put players up to 35 levels off.
- Encoders, set so that all land on the same quality (within 0.05 dB of each other, *measured*); the choice is
  file size, speed and the devices that must play the file:

  | Choice | Encoder | Settings | Frame size accepted |
  |--------|---------|----------|---------------------|
  | `amf` (default) | HEVC on the GPU | constant QP 18/20, tagged `hvc1` | 384 × 128 to 8192 × 4352 |
  | `av1_amf` | AV1 on the GPU | constant QP 40/48, width a multiple of 8 | 320 × 128 to 8192 × 4352 |
  | `h264_amf` | H.264 on the GPU | constant QP 18/20 | 96 × 66 to 4096 × 4096 |
  | `x264` | H.264 on the processor | CRF 17, preset medium | any |
  | `x265` | HEVC on the processor | CRF 18, preset medium, tagged `hvc1` | any |
  | `png` | PNG frames | lossless | any |

  The GPU encoders' limits are checked before any work starts ("<encoder> encodes from a × b to c × d; W × H
  needs another encoder or another size").
- Audio: every track is kept. AAC, MP3, AC-3, E-AC-3 and ALAC are copied into an MP4; the rest (WMA, PCM, Vorbis,
  Opus, FLAC, MP2...) become AAC at 96 kb/s per channel (at least two). An MKV copies every track as is.
  Subtitles and data streams are dropped.
- Images: the format follows the output's suffix; every input image format can be written. PNG, TIFF and WebP
  keep the source's transparency, scaled up; JPEG is written at quality 2 (ffmpeg's scale) and WebP at 95.

## 7. Output size
- FSR's range: each axis 1x to 3x ("FSR supports scales from 1x to 3x; ...").
- Command line: `--mode` native 1x, quality 1.5x, balanced 1.7x, performance 2x (default) or ultra 3x; `--scale`
  any factor; `--width` and `--height` (given alone, the other follows the picture's shape).
- Window: chips 1.5×, 2× (default), 3×, 1440p and 4K (the short side becomes 1440 or 2160) and Custom, a slider
  from 1.00 to 3.00 in steps of 0.01 (2.00 at first). A chip that would leave the range is disabled (1440p and 4K
  while no file is open), and a selection that becomes disabled falls back to 2×.
- Rounding: both sides to the nearest even number (the width to a multiple of 8 for AV1 on the GPU), one step
  back when rounding left the range (853 × 480 at 3x gives 2558 × 1440).
- Anamorphic sources are scaled from the shape they are shown at (stored width × pixel aspect), each axis on its
  own and within its own 1x to 3x: a 16:9 DVD's 720 × 480 at 2x gives 1706 × 960 ("2.37x by 2.00x"), at 2.25x
  1920 × 1080, at 3x 2160 × 1216.

## 8. Command line
`fsr4 upscale|ui|info|selftest` (`python -m fsr4` from the source tree; `--version` prints the version and build).

| `fsr4 upscale <input>` option | Default | Meaning |
|-------------------------------|---------|---------|
| `-o, --output` | §6 | output file; its folder must exist |
| `--mode`, `--scale`, `--width`, `--height` | performance | §7; `--scale` overrides `--mode`, a width or height both |
| `--flow dis\|zero` | `dis` | optical flow or no motion vectors (images always use none) |
| `--flow-preset ultrafast\|fast\|medium` | `fast` | flow quality |
| `--flow-scale N` | 1 | compute the flow at 1/N of the size |
| `--sharpness S` | off | sharpening strength, 0 to 1 (§4) |
| `--codec` | `amf` | §6 |
| `--upscaler 4\|3.1\|2.3\|hq\|realesrgan\|realesrgan-x4plus\|realesrgan-x4plus-anime` | 4 | §4; `hq` and the `realesrgan` ones ignore the `--flow` options; the last two are for images |
| `--frames N` | all | stop after N frames (audio cut to match) |
| `--gpu I`, `--sdk FOLDER` | automatic, `bin\` | device index; folder holding both FSR DLLs |
| `--validate`, `--debug-checking`, `--debug-view` | off | validation layer, FSR's checks (logged), FSR's debug overlay |
| `-q, --quiet` | off | no progress or details; warnings and errors still print |

- `fsr4 ui [file]` opens the window. `fsr4 info` prints the GPU (Vulkan and driver versions, extensions), every
  device, and the FSR versions the DLL offers, by creating a 1080p-to-4K context. `fsr4 selftest [--validate]`
  uploads a 1920 × 1080 gradient, copies and reads it back, and passes with no mismatching byte; it needs
  neither FSR nor ffmpeg.
- Output: progress on stderr every 0.1 s on one line ("[fsr4] frame 120/1440, 41.3 fps, 32 s left"); details on
  stdout, tagged by part ([fsr4], [gpu], [fsr], [pipeline], [media]): the input and output sizes, each audio
  track's fate, the GPU, the FSR version and its memory, the timings. Errors: "fsr4: <reason>" on stderr, one
  line.
- Exit codes: 0 done; 2 for a wrong request (input missing, scale out of range, sharpening outside 0 to 1, size
  the encoder refuses, output is the input, upscaler not in the DLL); 1 for a failure (ffmpeg, Vulkan, FSR, missing DLLs or OpenCV) or no
  frame written. Ctrl+C stops the decoder and deletes the partial output.

## 9. The window

### Look and layout
- A dark theme throughout: a custom ttk theme of anti-aliased image elements, a dark title bar, dark file
  dialogs, the app's own icon (a ladybug seen from above, facing right, its head a play triangle under the
  shell) and the ladybug's red as the accent colour. Every choice is the same themed dropdown; there is no
  native menu.
- Title "Coccinelle Player" ("42% · Coccinelle Player" during a whole-file upscale). It opens centred at
  1280 × 860 (at most 92% × 88% of the screen), never smaller than 960 × 640, at the display's DPI.
- No header: the name and icon are only in the title bar, not repeated below it, and the sidebar and the toolbar
  start at the top of the window.
- Sidebar (340 px wide): the settings, top to bottom (below). The button that folds it away starts its first line,
  before "Source"; folded, the same button starts the toolbar, in the same place on the screen. It scrolls when
  the window is too short; the wheel anywhere over it scrolls it and never changes a control. Everything but
  Processor is locked during a whole-file upscale, a save or an image upscale.
- Folding or unfolding the sidebar and showing or hiding the log dissolve the old layout into the new one in
  0.12 s: no widget is ever seen in a wrong place. While the window changes size only what changes is redrawn, and
  what is anchored to the right follows the edge.
- Under the settings, always in view while the sidebar shows, refreshed every 2 s: "<processor> · RAM <this
  program> / <installed> GB" and "● <graphics card> · VRAM <this program> / <card> GB · FSR <version>" (VRAM as
  DXGI counts it, Vulkan memory included), whose tooltip also says the tool is independent, not made or endorsed
  by AMD, with AMD's trademarks; "Checking the GPU…" until the check ends, a red dot and "GPU unavailable" if it
  fails. Below, faint, which copy runs: "v<version> · <commit> · built <date> <time>" for the executables (written
  by the build), "· from the sources" when Python runs them; " + changes" follows the commit when tracked files
  differed from it. The log starts with the same line, after the program's name.
- Main area, top to bottom: the error banner; a toolbar with the Left and Right dropdowns, then the zoom (Fit
  button, slider and percentage); the picture; the playback controls (videos); the log.
- Sliders (zoom, volume, speed) look alike: a track filled up to a round knob that grows under the pointer, and
  marks under the track with their labels. A click or a drag goes anywhere, landing on a mark within 4 px of it;
  a click on a label goes to its mark. Where labels would touch, the less useful one gives way, and so do all the
  labels less useful than it, so the scale stays even; the one under the knob always shows and stands out.
- Footer: a thin progress bar, the status text, Show log / Hide log, Open and Show in folder (after a save), Stop
  (during a whole-file upscale) and the accent Upscale button.
- Every setting has a tooltip (after 450 ms, 320 px wide, kept on the pointer's monitor) saying in plain words
  what it does and what it trades, with measured numbers where there are some.
- Errors: one red banner at a time (the first line of the error, at most 220 characters; ✕ closes it; opening a
  file or pressing Upscale clears it; in full screen it waits for the window to come back). The log (8 lines,
  monospace, lines holding "error" in red) has the details and opens by itself when an upscale, a save or an
  image fails.
- Only presets persist. The sides, zoom, volume, threads and priority start at their defaults at each launch.

### Settings
1. **Source**: the file name ("No file yet"); for a video "<codec> · W × H · <fps> fps", then "[plays as W × H
   · ]3:20 · 12,000 frames · <audio codec, N audio tracks or no audio>"; for an image "<codec> image · W × H".
   Empty, it says "Drop a file anywhere on this window, or choose one."; while probing, "Reading…". The button
   reads "Choose file…", then "Choose another file…". Just after it on its line (12 px apart: they belong together),
   the switch "Auto upscale" (on by default; tooltip "Upscale as soon as a file opens"; item 6).
   Off, a file opens as in a plain player: the sides showing a preset are turned off, and when that leaves nothing
   the first of them shows the source bicubic. Their settings stay, so choosing a preset in a side's list starts
   upscaling with them.
2. **Size**: the chips (§7), each with a tooltip giving its ratio and FSR's name for it (Quality, Performance,
   Ultra Performance). Under them "1920 × 1080 → 3840 × 2160" and "2.00× per side · 4.0× the pixels" ("2.37× by
   2.00×" for anamorphic sources), or why that size cannot be made, or "The output size appears once a file is
   loaded."
3. **Upscaling**: the settings of one side: the upscaled one, or with both upscaled the one chosen by "Settings of
   Left side / Right side" (right by default, shown only then, even when both are the same), which is also the side
   Upscale saves. With no side upscaled, none of this shows.
   Preset: a dropdown naming the preset the settings match, or "Custom settings", + to save them as a preset and,
   for one of the user's own, a bin to delete it. Upscaler: "FSR 4 · machine learning", "FSR 3.1 · older, hand-tuned",
   "FSR 2.3 · older, hand-tuned", "AMD HQ scaler · faster, no motion", "Real-ESRGAN · slow, redraws detail" and, for
   images, "Real-ESRGAN x4plus · general, stills only" and "Real-ESRGAN anime · drawings, stills only"; a note says
   what it does and its tooltip quotes the lab's PSNR (FSR 4 26.8 dB, FSR 2.3 26.5, Lanczos 26.4, HQ scaler 26.3,
   FSR 3.1 25.9, Real-ESRGAN 24.4). Under it, only what that upscaler can change: FSR, Motion vectors ("Optical flow ·
   medium, best", "· fast", "· ultrafast", "Off"; not for images, with a note) and Sharpening; the HQ scaler,
   Sharpening; Real-ESRGAN, nothing. Sharpening is a switch (off), a slider from 0 to 1 (0.40) and its value: 0.2 to
   0.5 gives crisper edges, high values leave halos. Without Real-ESRGAN's program, a saved side showing it says how
   to install it under the size, and Upscale stays disabled.
4. **Output**: with no side upscaled, a note says so and Upscale stays disabled. Then
   the output path (§6; tooltip: .mp4 plays everywhere, .mkv keeps every audio track), "…" for a
   Save-as dialog (MP4, Matroska, QuickTime; PNG, JPEG, WebP for images), the encoder ("HEVC · GPU (AMF)"
   default, "AV1 · GPU (AMF)", "H.264 · GPU (AMF)", "H.264 · CPU (x264)", "HEVC · CPU (x265)") with a note on
   size, speed and the devices that play it, and what happens to the audio ("Audio: AAC copied as is; WMA
   converted to AAC (.mp4 cannot hold it)." or "No audio in this file."). Images disable the encoder.
5. **Processor**: threads from 1 to every logical processor (default half, shown as "8 of 16") and the priority:
   Normal, Low (default) or Background (quickest but other programs may stutter / other programs stay smooth /
   only spare time). They cap background work only (§10), stay usable while busy and reach the running upscale.
6. **Preferences**: "Auto upscale" (Source) is kept between runs in
   `%APPDATA%\Coccinelle Player\preferences.json` (written through a temporary file; one that cannot be read is
   reported in the banner, the defaults are used, and the next change keeps it as `preferences.unreadable`; fields
   of other versions are ignored). It applies from the next file opened.

A change applies 450 ms after the last one. Each combination of size, upscaler, motion, sharpening and encoder
is its own version (§10).

### Opening a file
- Ctrl+O, the Source button or a click on the empty picture opens "Choose a video or an image" (videos and
  images, videos, images, all files), in the open file's folder.
- A file dropped anywhere on the window opens, from Explorer even when the window runs elevated; of several, the
  first opens and the banner says "Opened <name>; drop one file at a time." A file dropped on the executable or
  on `start.bat` opens too.
- The empty picture is a drop zone: a rounded outline, an upload icon, "Drop a video or an image here", "or
  click to browse" and the main formats; outline and icon take the accent colour on hover, with a hand cursor.
- Refused with a banner: opening during a save or a whole-file upscale (saying to wait or to stop it first), a
  folder, a missing file, a file with no video ("Cannot open <name>: no video stream in <file> (only sound)").
- Opening closes the current file and deletes its upscaled pieces; the picture shows "Reading <name>…" with a
  spinner. A video then shows frame 0 paused while upscaling starts there, and the upscaled side appears once the
  first half second is ready (1.6 s after opening for 1080p60 to 4K, *measured*); status "Ready · Space plays it
  upscaled · Upscale saves the whole video". An image is upscaled at once with every version shown; status
  "Ready · press Upscale or Ctrl+Enter to save". With "Auto upscale" off, nothing is upscaled
  until a side shows a preset: status "Ready · nothing is upscaled · choose a preset in the Left or Right list to
  upscale".

### Comparing
- The divider splits the picture; the Left and Right dropdowns say what a side shows: the presets (FSR 4 on the right
  by default), the source enlarged by a filter (original pixels, bilinear, bicubic (left by default), Lanczos), or
  Off, so the other side fills the picture without the divider (turning both off brings the other back to its
  default). The list shows the preset the side's settings match, or else the settings described. Sides with the same
  settings share one version. Both are upscaled at the same size and encoder, paused or playing, so two FSR settings
  or FSR 4 and FSR 3.1 sit side by side. With no side upscaled, nothing is.
- The divider is a 2 px line with a round handle in the window's background colour, 4.5 px wider each side than
  Windows' own ⟷ pointer (the user's pointer scheme and size, at the window's DPI; 37 px at 100% with the standard
  pointer), carrying two small white chevrons, < >, drawn where the pointer's arrowheads are. Over the picture the
  pointer is that same ⟷. A left press anywhere on the picture moves the divider there and dragging follows, the
  handle centred on the pointer, whose arrowheads then cover the chevrons. The handle sits mid-height until the
  divider is pressed or dragged, then at the pointer's height (zoomed in, the middle is often far from where one
  looks), and stays there once released; merely passing over the picture leaves it where it is. It is always whole
  inside the picture.
- A label in each top corner names its side: the preset, the settings described ("FSR 3.1 · rough motion ·
  sharpening 0.80"; best motion goes unsaid) or the filter ("Source" at Fit while the source is not enlarged). Outside full screen it
  ends in ▾ and clicking it opens that side's dropdown, on release (on press, the release would pick an entry).
  Where a version is not upscaled yet, its side shows the bicubic source at 45% brightness and its label adds
  "· upscaling…" or "· not upscaled here".
- Zoom: a Fit button (the default, lit at Fit), then a slider (480 px when there is room, down to 160 px) on a log
  scale from the smallest of Fit, Original size and 100% to 1600%, marked Original (one screen pixel per source
  pixel, the upscaled side shrunk to match), 100%, 200%, 400%, 800% and 1600%; between marks it zooms to the whole
  percent, and its value follows it ("137%"). The wheel over the slider zooms a quarter octave per notch around
  the middle, over the picture around the pointer, pausing at Fit and the marks; a double-click goes from Fit to
  100% at the pointer and from any other zoom back to Fit. With nothing to compare the button and the slider are
  disabled. Away from Fit, a pill at the bottom left shows "Original size" or the percentage.
- Pan with the right or middle button. A minimap at the bottom right (at most 180 px, a third of the view) shows
  the visible part when some is hidden; a click or drag there moves to it.
- Pixels: at Fit, area averaging when shrinking and bicubic when enlarging; between Fit and 100%, area averaging;
  from 100%, every output pixel is a block, exactly as the file holds it, the source sides filtered to output
  pixels first. Paused and playing alike.

### Playing a video
- Controls under the picture: the filmstrip, the seek bar, then frame back, play/pause, frame forward, the time,
  mute, the volume, the playback state, the reverse toggle, the speed slider with its value, and full screen. In a
  narrow window the speed slider gives up width first (down to 84 px, labelling only 0, ¼, 1× and 5), then the
  volume slider (keeping the labels 0 and 100), then the volume's percentage, so full screen stays in reach at the
  minimum size.
- Play (Space) plays the upscaled video in real time with its sound. It stops on the last frame; Play there
  starts over (from the end when playing backwards). With nothing to play, the banner says why.
- The seek bar shows the played part in the accent colour, what is upscaled lighter (ready in every version
  shown that is being upscaled) and a tick where the upscaler works; its knob grows on hover. Click or drag to
  seek (during a drag the player follows at most every 0.12 s and the upscaler moves on release). Hovering the
  seek bar or the filmstrip shows a line and "frame 157 · 0:05.239".
- The filmstrip shows the video in tiles, bright where upscaled (thumbnails of the current version, one a second,
  or one per 1/240 of a longer video) and dimmed to 35% elsewhere (40 thumbnails of the source); click or drag
  to seek.
- Frame steps: the buttons or `,` and `.`, pausing first; holding a button repeats after 400 ms, then every
  70 ms. ← and → jump 5 s; Home and End go to the first and last frame.
- The time reads "0:05.239 / 3:20 · frame 157" (hours when needed, frames counted from 0), monospace and as wide
  as at the last frame, so nothing after it moves.
- Volume: the speaker button or M mutes (the fill and the percentage turn grey). The 150 px slider is the sound's
  amplitude in percent on a decibel scale, like the speed's: 0 at its left end, then 1% (−40 dB) to 100%, marked
  0, 1, 3, 10, 30 and 100 (about 10 dB apart), so equal distances sound like equal steps. Its percentage has two
  figures (1.3%, 80%); the wheel over it or over the speaker moves it 2 dB and ↑ ↓ 4 dB (from 0 up to 1%, below
  1% down to 0); raising it unmutes. The speaker shows the level: two waves from 25%, one below, none at 0,
  crossed when muted. 80% at launch, kept between files.
- Speed: a 176 px slider on a log scale from 0.05× to 5×, its track on the volume slider's line, marked .05, 0.1, ¼,
  ½, 1×, 2 and 5 under the track, with 0 at its left end and no speed between 0 and 0.05×. A click or a drag goes
  anywhere (two decimals below 1×, one above), landing on a mark within 4 px or on a clicked label; `[`, `]` and
  the wheel go from mark to mark. Backspace or a click on the value goes back to 1× forwards, and each new file starts there. At 0 playback
  pauses and the next speed plays on; Play at 0 plays at 1×.
- Backwards: the toggle beside the slider, or R, plays backwards at the slider's speed; the value then reads
  −0.25×, with a real minus sign. At 0 it sets the direction of the next speed.
- A new speed in the same direction leaves the decoders running: a change holds the picture less than 0.1 s, and a
  drag does not stop it (*measured*). The sound starts again 0.15 s after the last change. Turning round starts the
  decoders over.
- From 0.25× to 2× the sound plays at its own pitch and late frames are dropped; slower it is silent and late frames
  are still dropped. Above 2× it is silent and every second or third frame is shown; backwards it is silent and
  every frame is shown. In both, playback slows down when decoding cannot keep up (*measured* with 1080p60: 3.6× at
  5×, 2.6× with a 4K upscale shown; backwards 43 frames a second, 32 with a 4K upscale, so −0.5× to −0.7× from −1×
  on).
- Buffering: when a version being upscaled lacks the next frames, sound and clock stop, a spinner turns on the
  picture and the controls read "Buffering · upscaling ahead…". Playback resumes with 2 s ready ahead (or
  everything to the end); after catching up with an upscaler slower than the video it waits longer, clamp((1 − r)
  × 15 s, 2 s, 10 s) with r = upscaler fps ÷ versions being upscaled ÷ (video fps × |speed|), so it does not stop
  every few frames. Versions not being upscaled never hold playback up.

### Upscale
- Upscale (Ctrl+Enter) first checks the plan (a problem goes to the banner) and asks "<name> already exists.
  Replace it?" for an existing output the user typed.
- Video: only the saved side's version is upscaled, as it was when Upscale started, from the playhead to the end, then from the start;
  pieces already made with these settings are reused, so a finished version saves at once. While paused, the
  picture follows the upscaler's newest frame every 0.5 s until Play, a seek or a step (not with the HQ scaler,
  whose frames exist upscaled only in the encoder: the picture stays where it is). The status reads
  "Upscaling · 42% · 38 fps · 3:12 left", the bar shows the share done, the title the percentage. Then "Saving
  <name>…": the pieces are joined without re-encoding, with the source's audio, and "Saved <name> · 3:20 of
  video at 3840 × 2160". Open plays the file in the user's player and Show in folder selects it in Explorer; both
  go away at the next upscale or file.
- Stop or Esc ends a whole-file upscale: "Stopped at 42% upscaled · Upscale carries on from there"; what is done
  stays while the file is open. Saving itself cannot be stopped.
- Image: one pass with the saved side's settings, "Upscaling the image…" then "Saved <name> · 3840 × 2160 in 1.2 s".
  Stop or Esc cancels it: "Cancelled · nothing was saved", and no file is written.
- An upscaling error ends the whole-file upscale, pauses playback, shows "Upscaling stopped: <reason>" and opens
  the log. Each new piece needs 2 GB free in the pieces' folder; below that the reason says how much is free and
  that 4K takes about 10 MB per second.
- Closing the window during a whole-file upscale asks "The video is still being upscaled. Close the window and
  lose that work?"; during a save it closes once the video is saved.

### Presets
- A preset holds the upscaler, the motion vectors and the sharpening only; the size and the encoder stay the file's.
- Built in, always there and never changed: "FSR 4" (the best settings, *measured*: medium optical flow, 27.10 dB
  against 26.82 for fast at less than half the speed, no sharpening), "FSR 4 · fast motion", "FSR 4 · sharpened"
  (0.50), "FSR 4 · no motion", "FSR 3.1", "FSR 2.3", "AMD HQ scaler", "Real-ESRGAN", "Real-ESRGAN x4plus",
  "Real-ESRGAN anime". The user's own come first in the lists.
- Choosing one in the Preset dropdown, or in a side's list, loads it into that side's settings; changing them leaves
  the preset as it is, and the dropdown says "Custom settings". A video does not offer the models for still images,
  and a side with such a model when a video opens goes back to Real-ESRGAN's video model. + opens a row with a name
  (the settings' description, or "My preset"), Save and Cancel: Enter saves, Escape cancels. Refused: an empty name,
  more than 40 characters, an entry of the side lists (in any case). A name of the user's used again (in any case)
  replaces that preset. The bin deletes it, without asking; the side keeps its settings.
- Saved in `%APPDATA%\Coccinelle Player\presets.json` (`{"version": 1, "presets": [...]}`, written through a
  temporary file). A file that cannot be read is reported in the banner and never overwritten: the next save
  keeps it as `presets.unreadable`. Every save or delete reads the file again first, so two windows open at once
  keep each other's presets.

### Full screen
- F11, F or the button enters; the same or Esc leaves. Images too.
- It covers the monitor holding most of the window, without borders, and keeps the picture playing; leaving puts
  the window back exactly where and as it was (a maximized window is restored first and maximized again). If it
  fails, the window comes back with "Full screen is unavailable: <reason>". Tk's own full screen is not used: it
  always takes the primary monitor.
- Hidden in full screen: footer, toolbar, sidebar, banner, log and the picture's frame; the labels lose
  ▾ and open nothing.
- The playback controls (videos) sit over the bottom of the picture. They come up when the pointer moves within
  their height plus 80 px of the bottom, hide 600 ms after it moves away, and hide with the pointer after 2.5 s
  without movement, even at the bottom; never while a button is held or the seek bar dragged. They also show on
  entering, on pause, at the end and when a file loads, until the pointer rests.

### Keys and mouse
| Key | Action | Key | Action |
|-----|--------|-----|--------|
| Space | play / pause | ← → | 5 seconds back / forward |
| , . | one frame back / forward | Home End | first / last frame |
| F, F11 | full screen | Esc | leave full screen, else stop the upscale |
| M | mute | ↑ ↓ | volume ±4 dB |
| [ ] | slower / faster | Backspace | 1× forwards |
| R | backwards / forwards | | |
| Ctrl+O | open | Ctrl+Enter | upscale |
| Ctrl+L | log | Ctrl+B | fold the settings |

- Single-key shortcuts do nothing while typing in a field. A click outside a field, a pick in a dropdown, or
  Return or Escape in the output field gives the keys back to the window.
- Mouse: a click on the empty picture opens a file; on the picture, a left press or drag moves the divider, the
  wheel zooms, a right or middle drag pans and a double-click switches Fit and 100%; a label opens its side's
  dropdown; the minimap, the seek bar and the filmstrip take clicks and drags; the frame buttons repeat while
  held; the zoom, the volume and the speed slider take clicks, drags and the wheel (the speaker takes the wheel
  too); the Fit button fits the picture; a click on the speed value resets it.
  Dropdowns ignore the
  wheel, so scrolling the sidebar never changes a setting.

## 10. Behind the picture
- Pieces: the window never waits for a whole file. A renderer (one thread, owning the GPU) upscales into pieces
  of 0.5 s at first after a seek, so playback starts soon, then 1, 2, 4 and 8 s. Each piece is one encoder run
  with a keyframe every 2 s; 8 frames before each new run are upscaled and dropped so that FSR's history and the
  optical flow start warm.
- Scheduling: in preview mode the renderer works ahead of the playhead, from when less than 12 s is ready up to
  30 s ahead; after Upscale it does everything, starting at the playhead. Seeking moves the work there at once,
  and what was upscaled elsewhere stays. Backwards, it works in 1 s chunks just behind the ready stretch, until
  12 s behind the playhead are ready.
- Versions: every distinct set of settings is a version with its own pieces. In preview mode every version shown
  is upscaled, in turns: the one with the least ready ahead goes first, and a run yields once it is ahead of
  another by as much as that one has (half a second at first, so both show a frame soon after a seek, then up to
  2 s). Versions no longer shown are kept, the four most recent, so going back is instant. After Upscale only the
  saved version is worked on; another version shown then plays what it has.
- Saving joins the pieces with ffmpeg's concat demuxer (no re-encoding) and adds the source's audio. The joined
  video must equal a single pass frame for frame, with the audio in sync and the colour tags kept.
- Playback: one decoder per upscaled version shown (cropped to the view by ffmpeg) and one for the source. The
  sound card's clock (waveOut) decides which frame is due, with ffmpeg's atempo away from 1×; without sound the
  system clock runs. Up to 2× forward, late frames are dropped; faster and backwards, every frame is shown in
  order and a late decoder slows playback rather than skipping. Backwards, each decoder decodes a short chunk
  forward (one seek, about 64 MB of frames) and hands it over last frame first while the chunk before decodes.
  The player plays the first audio track; saved files keep every track.
- Drawing: the picture is drawn with GDI (`SetDIBitsToDevice`), never Tk images (10 to 60 times slower,
  *measured*). Everything on it (labels, divider, minimap, spinner) is composed in numpy; per-frame work stays
  under a few milliseconds and allocates almost nothing. While playing, a zoom or pan resamples the previous
  picture into place until the decoders restart, 0.15 s after the view settles.
- Images: an image previewer with its own GPU device upscales the image with every version shown.
- Temporary pieces live in a `fsr4-` folder beside the output (about 10 MB per second of 4K), until the file is
  closed; Real-ESRGAN's pictures go there too, in the window and on the command line. The folder is the output's as
  it is when the file opens (by default the source's own); choosing another output later keeps them where they are.
  An output folder that cannot take files (read-only, or not made yet) gives way to the user's temporary folder, and
  the log says so; no folder is ever made for them. Each `fsr4-` folder carries a marker while a window holds it;
  folders left by a crash are removed when the window starts (the user's temporary folder) and when it next puts
  pieces in that folder, and only this tool's, never one a running window holds.
- Processor setting: the renderer's ffmpeg processes get `-threads N` when N is below the processor count (x265
  `pools=N`), OpenCV at most N and never more than 8, and the processes and worker threads run at the chosen
  priority; thumbnails and the final join too. The GPU encoders are not capped, and playback never is.

## 11. Painting and responsiveness
- No widget may stay black or show an old layout after the sidebar folds or unfolds, full screen, the log, or a
  move, resize or maximize of the window: every widget is repainted through Windows and through Tk at once and
  again once the layout settled.
- A whole repaint takes about 20 ms, 70 ms at 3440 pixels wide, and a field about 1.5 ms (*measured*): theme
  images are opaque over their container's colour and 9-slice middles are pre-stretched, so Tk draws few tiles
  and blends nothing.
- While upscaling, the window's 10 ms timer fires at most about 3 ms late (99th percentile, *measured*).
- While zooming or panning during playback both sides of the divider move together: a picture whose decoder
  still works at the old zoom is resampled to where it belongs, never stretched.
- Dragging the zoom slider zooms around the view's middle and the picture stays put: the point there stays within
  half a screen pixel of it at every step, and back at the starting zoom the picture is exactly where it was
  (*measured* in the window: at most 0.88 px between steps of a drag to 600% and back, 0.00 px off at the end).

## 12. Packaging
- `build_exe.bat [output folder] [work folder]` (default `dist` and `build`) checks the FSR DLLs against their
  pinned hashes (downloading them if that fails), builds with PyInstaller from `fsr4.spec`, writes the
  third-party notices, and copies the FSR DLLs with AMD's license into `bin\`, the README, and the license as
  `LICENSE.txt`. Any failure prints one line, pauses and exits 1. It ends by reminding that ffmpeg and ffprobe
  must be on PATH or next to the executables.
- `dist\fsr4\` holds `Coccinelle Player.exe` (the window, no console) and `fsr4.exe` (the command line), built from
  one entry script: no arguments, or one existing file, opens the window (`fsr4.exe` then hides its console when
  it owns it); anything else is the command line. Both carry the icon and a version resource (product "Coccinelle
  Player", version 0.1.0, "Copyright (c) 2026 TermWay", "AMD, Radeon and FidelityFX are trademarks of Advanced
  Micro Devices, Inc. Not affiliated with AMD."). OpenSSL, asyncio and other unused modules, and OpenCV's own
  ffmpeg plugin (31 MB), are left out; the folder is about 170 MB, 85 MB zipped.
- `licenses\` holds one notice per component, from the installed packages and `third_party\`, and the build fails
  when one is missing: Python, a pointer to the notices of the software built into Python, Tcl, Tk, numpy,
  OpenCV and the libraries it bundles, AMD's FidelityFX API headers (MIT), and the Khronos Vulkan header (whose
  copyright line it must carry).
- `tools/fetch_sdk.py` downloads `amd_fidelityfx_loader_vk.dll`, `amd_fidelityfx_upscaler_vk.dll` and
  `license.md` from tag v2.4.0 of AMD's FidelityFX SDK repository, each through a `.part` file checked against
  its pinned SHA-256 before it takes its place; files that already match are kept, and `--check` only checks.
- `start.bat` opens the window, with a file dropped on it: the built executable when there is one, else Python
  after checking the DLLs (downloading them if needed). With Python there, it first runs `tools/dev_build.py`
  so it never opens a build older than the sources: when a build input (the package, the spec, the build
  script, the DLLs, the README, the licenses, the requirements) is newer than `dist\fsr4\`, it builds into
  `out\dev-build\` with `build_exe.bat` and puts that in place of `dist\fsr4\` when no program runs from there.
  A build that must wait for such a program stays in `out\dev-build\` for the next run, and meanwhile, or when
  the build fails, the window opens from the sources. `upscale.bat` upscales every file dropped on it with the
  defaults (2x, HEVC on the GPU, optical flow fast), with no questions, says when one failed, and pauses. Windows
  hands it a dropped name without quotes when it has no space, so names holding `&`, `^`, `,`, `;` or `=` are
  cut: the README says so.
- The executables are not code-signed: every release publishes the SHA-256 of what it ships.

## 13. License and credits
- The project's code is MIT, in the name of TermWay. Third-party parts keep their licenses: AMD's FSR API
  headers (MIT), the Khronos Vulkan header (Apache-2.0 or MIT, used under MIT), AMD's DLLs under AMD's FSR SDK
  license (redistributable in binary form with its notice), and in the packaged app Python, Tcl/Tk, numpy and
  OpenCV with the libraries they bundle. ffmpeg is not included.
- The tool is not affiliated with AMD: the README says so with the trademarks, as do the window title's tooltip
  and the version resource.
- The README credits Claude (Anthropic's AI model) as the author, directed and tested by TermWay.

## 14. Acceptance checks
- `python -m unittest discover -s tests` passes. The tests pin the Vulkan and FSR structure layouts, the flow's
  sign convention, one round trip per container and codec family (with exact frames after seeks), output sizes
  and limits, presets (including a broken file), the timeline's scheduling, the player's streams, the view's
  geometry, the theme images, the cleanup of temporary folders, and that pieces rendered around seeks join into
  exactly the single-pass video. Tests that need ffmpeg, OpenCV, FSR 4 (the card and the DLLs) or a sound device
  skip without it, saying why.
- `python -m fsr4 selftest --validate` passes under the validation layer.
- The quality lab (`tools/lab_quality.py`) downscales a reference 2x (area), upscales it back and reports PSNR
  against it for each setting, beside bicubic, Lanczos, AMD's HQ scaler when ffmpeg has `sr_amf`, and Real-ESRGAN
  when `--realesrgan` names its exe. Defaults change only on its numbers. *Measured* on a panning fractal clip:
  FSR 4 with DIS fast flow 26.82 dB (medium 27.10), Lanczos 26.42, HQ scaler 26.34, Real-ESRGAN 24.35, FSR 4
  without motion vectors 25.59, with the flow negated 25.63.
- Speed, end to end (decode, flow, upscale, encode), *measured*: 720p60 to 1440p at 85 fps, 1080p60 to 4K at
  47 fps, 60 fps without motion vectors, 72 fps with the HQ scaler, 4.8 fps with Real-ESRGAN. In the window a 1080p60 file plays at 60 fps once a few
  seconds are ready, about 0.3 s after Play.
- `tools/ui_smoke.py` drives the real window through every state (a real drop, the first upscaled frame, a
  tooltip, a dropdown, the file dialog, playback, the seek bar's hover, buffering after a seek, zoom with
  original pixels, the zoom slider (a drag that must keep the picture still) and wheel, a side's dropdown opened from its label, a preset saved and two
  presets side by side paused and playing, a frame step, playing backwards, a side turned off, full screen on the
  window's own monitor with its controls showing and hiding, the folded sidebar, a whole-file save, an image
  compared with a preset, an error, the minimum size) and saves screenshots, which are read before a window
  change is called done. It captures only its own window.
