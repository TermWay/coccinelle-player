"""The release zip: the build folder's contents at its root, and a checksums file that matches it."""
import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import release_zip  # noqa: E402


class ReleaseZipTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.build = self.root / "fsr4"
        (self.build / "bin").mkdir(parents=True)
        (self.build / "fsr4.exe").write_bytes(b"exe")
        (self.build / "bin" / "license.md").write_bytes(b"license\n")

    def tearDown(self):
        self.folder.cleanup()

    def test_zips_the_folder_contents_at_the_root_and_writes_its_hash(self):
        target, digest = release_zip.make_release_zip(self.build, self.root / "out", "1.2.3")
        self.assertEqual(target.name, "Coccinelle-Player-1.2.3-win-x64.zip")
        with zipfile.ZipFile(target) as archive:
            self.assertEqual(sorted(archive.namelist()), ["bin/license.md", "fsr4.exe"])
            self.assertEqual(archive.read("bin/license.md"), b"license\n")
        self.assertEqual(digest, hashlib.sha256(target.read_bytes()).hexdigest())
        checksums = (self.root / "out" / release_zip.CHECKSUMS_FILE).read_bytes()
        self.assertEqual(checksums, f"{digest}  {target.name}\n".encode("ascii"))

    def test_writes_how_to_check_the_hash_for_notepad(self):
        target, _ = release_zip.make_release_zip(self.build, self.root / "out", "1.2.3")
        instructions = (self.root / "out" / release_zip.INSTRUCTIONS_FILE).read_bytes()
        self.assertIn(f"Get-FileHash -Path {target.name} -Algorithm SHA256".encode("ascii"), instructions)
        self.assertIn(release_zip.CHECKSUMS_FILE.encode("ascii"), instructions)
        self.assertEqual(instructions.count(b"\n"), instructions.count(b"\r\n"), "Windows line ends throughout")

    def test_refuses_an_existing_zip(self):
        release_zip.make_release_zip(self.build, self.root, "1.2.3")
        with self.assertRaises(SystemExit):
            release_zip.make_release_zip(self.build, self.root, "1.2.3")

    def test_refuses_a_folder_without_a_build(self):
        (self.build / "fsr4.exe").unlink()
        with self.assertRaises(SystemExit):
            release_zip.make_release_zip(self.build, self.root, "1.2.3")


if __name__ == "__main__":
    unittest.main()
