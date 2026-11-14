"""Zip a build of the executables for a release and write its SHA-256 beside it.

    python tools/release_zip.py [build folder] [output folder]

The build folder defaults to dist\\fsr4 and the output folder to the build folder's parent. Writes
Coccinelle-Player-<version>-win-x64.zip (the build folder inside it, under its own name) and SHA256SUMS.txt
("<hash>  <zip name>", as sha256sum writes it), then prints the zip's path and its hash. An existing zip of that
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
                archive.write(path, (Path(build.name) / path.relative_to(build)).as_posix())
    digest = sha256(target)
    (output / CHECKSUMS_FILE).write_text(f"{digest}  {target.name}\n", encoding="ascii", newline="\n")
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
