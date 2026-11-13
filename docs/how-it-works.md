# How it works

## Feeding a video to a game upscaler
FSR 4 is a game upscaler: it expects a low-resolution render plus depth, per-pixel motion vectors and a
sub-pixel camera jitter every frame. A video file has none of those, so this tool substitutes:

| FSR input      | What the tool feeds it                                                   |
|----------------|--------------------------------------------------------------------------|
| Colour         | The decoded frame (sRGB 8-bit, declared as non-linear colour)            |
| Motion vectors | DIS optical flow between consecutive frames (`--flow dis`), or zero      |
| Depth          | A constant plane                                                         |
| Jitter         | None (faking it by resampling the input only blurs, see the lab)         |
| Exposure       | FSR's auto exposure                                                      |

No jitter and no real depth means FSR 4 works mostly as a very good spatial upscaler with temporal
stabilisation; it cannot reconstruct the sub-pixel detail it recovers in games. What each substitution is
worth is in [Measurements](measurements.md#quality).

## How playback works
The window never waits for a whole file. `fsr4/render.py` upscales into pieces (0.5 s at first after a seek so
playback can start soon, growing to 8 s): each piece is one encoder run, and a few frames before each new
stretch are upscaled but not kept so FSR's history and the optical flow are warm. `fsr4/timeline.py` keeps the
pieces and decides where to work: ahead of the playhead in preview mode (from when less than 12 s is ready to
30 s ahead), or everything, starting at the playhead, after Upscale. Saving joins the pieces with ffmpeg's
concat demuxer (stream copy) and adds the audio; `tests/test_render.py` checks that a joined video matches a
single pass frame for frame, with the audio in sync and the colour tags kept.

Each set of settings shown is a version with its own pieces. In preview mode the
renderer upscales every version shown, taking turns: the one with the least ready ahead of the playhead goes
first, and a run yields once it is ahead of another by as much as that one has (half a second at first, so
both show their frame soon after a seek, then up to 2 s). Two versions of 1080p60 at 4K share the renderer's
40 fps, about 20 each: after a seek the first shows its frame in 1.3 s and the second in 2.1 s, and playing
both waits more often. Versions no longer shown are kept (the four most recent) so switching back is instant.

`fsr4/player.py` plays the ready pieces: one decoder per upscaled version shown (cropped to the view by
ffmpeg), one for the source (the filters), the sound card's clock (waveOut, `fsr4/audio.py`, with ffmpeg's
atempo away from 1×) deciding which frame is due. Without sound (0.1×, faster than 2×, backwards) the system
clock runs; faster than 2× and backwards every frame is shown in order: a late decoder slows playback down
rather than making it skip to frames nobody decoded. Backwards, each decoder decodes a short chunk forward (one
seek) and hands it over last frame first while the chunk before it decodes; about 64 MB of frames per chunk,
so 8 frames of 1080p and 11 of an upscale shown in a 1600 × 900 view.

`fsr4/ui_video.py` draws everything with GDI: Tk's image path takes 20 ms for a 1600×900 frame and 120 ms at
4K, a GDI blit 0.5 to 2 ms, so video plays smoothly in the window and full screen.

## Temporary files
Pieces wait in a `fsr4-` folder beside the output (about 10 MB per second of 4K) until the file is closed, or
in the system's temporary folder when the output's cannot take them; folders left by a crash are removed the
next time that folder is used.

The source code's layout is in [Development](development.md#layout).
