"""Zip a build of the executables for a release and write its SHA-256 beside it.

    python tools/release_zip.py [build folder] [output folder]

The build folder defaults to dist\\fsr4 and the output folder to the build folder's parent. Writes
Coccinelle-Player-<version>-win-x64.zip (the build folder's contents at its root: Windows' Extract All already puts
them in a folder named after the zip, so a folder inside would nest twice) and SHA256SUMS.txt
("<hash>  <zip name>", as sha256sum writes it) with HOW-TO-VERIFY.txt (the PowerShell commands that check it),
then prints the zip's path and its hash. An existing zip of that
name is refused: a release asset is made once, from one build.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fsr4 import __version__  # noqa: E402

CHECKSUMS_FILE = "SHA256SUMS.txt"
INSTRUCTIONS_FILE = "HOW-TO-VERIFY.txt"
REPOSITORY = "TermWay/coccinelle-player"


def zip_name(version: str = __version__) -> str:
    """The release asset's file name for `version`."""
    return f"Coccinelle-Player-{version}-win-x64.zip"


def sha256(path: Path) -> str:
    """The file's SHA-256 as lower-case hex."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_instructions(name: str) -> str:
    """How a Windows user checks the zip called `name` against SHA256SUMS.txt, as plain text."""
    return f"""How to check the download
=========================

The SHA-256 of {name} is in {CHECKSUMS_FILE}, next to this file on the release page.
A matching hash means the zip arrived complete and unchanged.

1. Put {name} and {CHECKSUMS_FILE} in the same folder.
2. Open PowerShell in that folder: in Explorer, click the address bar, type powershell and press Enter.
3. Run:

       Get-FileHash -Path {name} -Algorithm SHA256

   The Hash it prints must be the one in {CHECKSUMS_FILE} (capital or small letters do not matter).

Or let PowerShell compare them; it prints True when they match:

       (Get-FileHash -Path {name} -Algorithm SHA256).Hash -eq (Get-Content {CHECKSUMS_FILE}).Split(" ")[0]

Where it was built
------------------
Both files come from the same page, so the hash alone does not prove who made the zip. Each zip is built by
GitHub Actions from the tagged commit, and GitHub signs a record of that. With the GitHub CLI
(https://cli.github.com/), this checks the zip against that record:

       gh attestation verify {name} --repo {REPOSITORY}
"""


def make_release_zip(build: Path, output: Path, version: str = __version__) -> tuple[Path, str]:
    """Zip `build` into `output` and write the checksums file there; returns the zip's path and its SHA-256."""
    if not (build / "fsr4.exe").is_file():
        raise SystemExit(f"release_zip: {build} holds no fsr4.exe: build it first (build_exe.bat)")
    target = output / zip_name(version)
    if target.exists():
        raise SystemExit(f"release_zip: {target} already exists")
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(build.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(build).as_posix())
    digest = sha256(target)
    (output / CHECKSUMS_FILE).write_text(f"{digest}  {target.name}\n", encoding="ascii", newline="\n")
    (output / INSTRUCTIONS_FILE).write_text(verify_instructions(target.name), encoding="ascii", newline="\r\n")
    return target, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("build", nargs="?", type=Path, default=ROOT / "dist" / "fsr4", help="the build folder")
    parser.add_argument("output", nargs="?", type=Path, help="where the zip goes (default: the build's parent)")
    args = parser.parse_args()
    build = args.build.resolve()
    target, digest = make_release_zip(build, (args.output or build.parent).resolve())
    print(target)
    print(f"{digest}  {target.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
