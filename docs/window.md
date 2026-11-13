# Using the window

Everything the window does, control by control. For a first run, the [README](../README.md) is enough.

Double-click `Coccinelle Player.exe` (or `start.bat` in the source tree), or drop a file on its icon.

## Open a file
Drop a video or an image anywhere on the window, click the empty preview, or use Choose file (Ctrl+O).
Every setting explains itself when the pointer rests on it.

## Size
1.5×, 2×, 3×, 1440p or 4K (the short side; chips that would leave FSR's 1×–3× range are disabled), or Custom
with a slider. The sidebar shows the output size, what happens to each audio track and what the chosen encoder
trades.

## Watch it upscaled
Play (Space) plays the video in real time with sound. The upscaler follows the playhead: it keeps the next
seconds upscaled, jumps to wherever you seek, and keeps everything it made.

- When the upscaler is slower than the video, playback waits with a spinner, like a streaming site, then plays
  on. 1080p60 to 4K runs at about 44 fps with fast optical flow, and about 20 with the default preset, which
  uses the most accurate motion.
- The seek bar shows what is upscaled (lighter) and where the upscaler works (a tick). Hovering it gives the
  frame number and its time to the millisecond.
- The filmstrip above it shows the video, bright where it is upscaled.
- The buttons around Play step one frame back or forward (hold to keep stepping). The time shows the current
  frame's number and keeps one width, so the volume after it stays put.
- The volume slider is a decibel scale like the speed's, marked 0, 1, 3, 10, 30 and 100%, so equal distances
  sound like equal steps (click, drag, the wheel over it or the speaker, ↑ ↓); the speaker shows the level.

## Speed and playing backwards
The slider at the right of the controls is a log scale from .05× to 5×, with 0 at its left end: 0 pauses, and
the next speed plays on.

- A drag stops anywhere (0.35×) and lands on a mark when near one. [, ] and the wheel go from mark to mark;
  Backspace or a click on the value goes back to 1× forwards.
- The ◀ button beside it (R) plays backwards at the slider's speed.
- Dragging never interrupts the picture: the decoders keep running and the sound follows when you stop.
- The sound plays from 0.25× to 2× at its own pitch and is silent otherwise.
- Above 2× every second or third frame is shown, backwards every frame, and playback slows down when decoding
  cannot keep up. 1080p60 reaches about 3.6× at 5× here (2.6× with a 4K upscale shown), and backwards about 43
  frames a second (32 with a 4K upscale; the source decodes in short chunks, last frame first), so −0.5× and
  slower play at their speed and anything faster plays at −0.5× to −0.7×.
- Backwards, the upscaler works on the seconds just behind the playhead.

## Compare anything
The divider splits the picture. The Left and Right dropdowns above it (or a click on the label in either top
corner) choose what each side shows: a preset, or the source enlarged by a classic filter.

- The filters: Original pixels (each source pixel as a block), Bilinear (what most players do), Bicubic, or
  Lanczos.
- Two sides upscaled sit side by side, paused or playing (two versions share the GPU, so playing them waits
  more). The sidebar then asks which one its settings are for, which is also the one Upscale saves.
- Off turns a side off: the other one fills the picture, without the divider.
- Dragging the divider also takes its handle to the pointer's height.

## Presets
Presets cover the choices worth comparing: FSR 4 at its best (medium optical flow, no sharpening) with faster,
sharpened and motionless variants, FSR 3.1, FSR 2.3, AMD's HQ scaler and Real-ESRGAN.

The sidebar shows the preset, its upscaler and only what that upscaler can change: motion vectors and
sharpening for FSR, sharpening for the HQ scaler, nothing for Real-ESRGAN. + saves those settings as a preset
of your own (the size and the encoder stay the file's) in `%APPDATA%\Coccinelle Player\presets.json`, and the
bin deletes one.

## Zoom
The Fit button shows the whole picture; the slider beside it is marked like the speed's.

- Original shows the source at its own pixel size (the upscaled side shrunk to match); 100% to 1600% show
  upscaled pixels 1:1 or enlarged. Click or drag anywhere on it (137%), or click a mark.
- From 100% up every pixel is drawn as a block, exactly as the file holds it.
- The wheel zooms around the pointer in finer steps (four per doubling); right-drag or the minimap moves the
  visible area; double-click switches between Fit and 100%.
- All of it works while playing.

## Upscale and save
Upscale (Ctrl+Enter) does the rest of the file the way the saved side says, starting at the playhead, and saves
it: the pieces are joined without re-encoding and the source's audio is added.

- Stop (Esc) keeps what is done while the file stays open, and Upscale carries on from there.
- Changing a setting starts a new version from the playhead; the earlier ones are kept, so going back to them
  is instant.
- When it is saved, Open plays the file and Show in folder selects it.

## Images
Images are upscaled as soon as they open, with every version shown, so presets can be compared before Upscale
saves the picture.

## Auto upscale
The switch beside Choose file, kept between runs: off, files open as in a plain player (the source, bicubic)
and FSR starts only when you choose a preset in the Left or Right list.

## Full screen, sidebar and processor
- Full screen (F11, F or the button) covers the monitor the window is on and keeps the video; leaving it puts
  the window back exactly where and as it was. Its controls come up when the mouse moves near the bottom of
  the screen, and hide, with the pointer, once the mouse rests, wherever it is.
- The button at the top left of the window (Ctrl+B) folds the settings away.
- **Processor** in the settings sets how many threads upscaling may use (1 to all) and its priority (Normal,
  Low, Background).
- The window reopens as you left it: its place and size, the volume, the folded sidebar, the output size, the
  encoder, the threads and the priority.

## Log and status
- Show log (Ctrl+L) has the details; errors appear in a red banner.
- For a bug report, Copy log (beside Hide log) copies the log with the version, the graphics card and its
  driver, Windows and ffmpeg: read it before posting, it names your files.
- The foot of the sidebar names the processor and the graphics card, with the memory this program uses on each
  (RAM, VRAM) against what is installed. An amber dot and "no FSR 4" there mean the card cannot run FSR 4, and
  its presets say "not on this card".

## Keys

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| Space | play / pause | ← → | 5 seconds back / forward |
| , . | one frame back / forward | Home End | start / end |
| F, F11 | full screen | Esc | leave full screen, else stop the upscale |
| M | mute | ↑ ↓ | volume |
| [ ] | slower / faster | Backspace | normal speed, forwards |
| R | play backwards / forwards | | |
| Ctrl+B | settings sidebar | Ctrl+O, Ctrl+Enter, Ctrl+L | open, upscale, log |
