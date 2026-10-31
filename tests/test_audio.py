"""The player's sound: decoder arguments, start offsets, and the waveOut clock (skipped without a sound device)."""
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from fsr4 import media
from fsr4.audio import AudioPlayer, audio_command, leading_silence


def make_clip(path: Path, video_delay: float = 0.0) -> None:
    subprocess.run([media.tool_path("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "4", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)], check=True)


class AudioTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.clip = Path(self.directory.name) / "clip.mp4"
        make_clip(self.clip)
        self.info = media.probe(self.clip)

    def tearDown(self):
        self.directory.cleanup()

    def test_decoder_arguments(self):
        command = audio_command(self.clip, 0, 1.5)
        self.assertEqual(command[command.index("-ss") + 1], "1.500000")
        self.assertEqual(command[command.index("-map") + 1], "0:a:0")
        self.assertIn("s16le", command)
        self.assertNotIn("-ss", audio_command(self.clip, 0, 0.0))

    def test_silence_before_a_late_track(self):
        self.info.audio[0].start_time = self.info.start_time + 0.5
        self.assertEqual(leading_silence(self.info, 0, 0.0), 24000)
        self.assertEqual(leading_silence(self.info, 0, 0.25), 12000)
        self.assertEqual(leading_silence(self.info, 0, 1.0), 0)

    def test_clock_runs_after_pause_and_restart(self):
        try:
            player = AudioPlayer(self.clip, self.info)
        except OSError as error:
            self.skipTest(f"no sound device: {error}")
        try:
            player.set_volume(0.0)
            player.start(1.0)
            time.sleep(0.3)
            self.assertGreater(player.position(), 1.05)
            player.pause()
            paused = player.position()
            time.sleep(0.2)
            self.assertAlmostEqual(player.position(), paused, delta=0.01)
            player.start(2.0)  # a seek or the end of buffering: the device must play again
            time.sleep(0.4)
            self.assertGreater(player.position(), 2.15)
            self.assertIsNone(player.error)
        finally:
            player.close()


if __name__ == "__main__":
    unittest.main()
