# PyInstaller build: two executables sharing one folder (dist/fsr4).
#   "Coccinelle Player.exe"  the window, no console (double-click it)
#   fsr4.exe             the command line (fsr4 upscale clip.mp4 ...); with no arguments it opens the window too
# Build with build_exe.bat: it also checks and copies the FSR DLLs next to the executables and writes the
# third-party notices into licenses\.
import re
import sys
from pathlib import Path

from PyInstaller.utils.win32.versioninfo import (FixedFileInfo, StringFileInfo, StringStruct, StringTable,
                                                 VarFileInfo, VarStruct, VSVersionInfo)

ROOT = Path(SPECPATH)
sys.path.insert(0, str(ROOT))
from fsr4.ui_art import ico_bytes  # noqa: E402

COPYRIGHT = "Copyright (c) 2026 TermWay"
TRADEMARKS = ("AMD, Radeon and FidelityFX are trademarks of Advanced Micro Devices, Inc. "
              "Not affiliated with AMD.")
# Pulled in by numpy's optional imports (numpy.testing, numpy.f2py, show_config) and by urllib.request; the app
# imports none of them. Without ssl and _hashlib no OpenSSL DLL ships (hashlib keeps Python's own hashes).
UNUSED_MODULES = ["psutil", "yaml", "charset_normalizer", "ssl", "_ssl", "_hashlib", "asyncio", "_asyncio"]


def source_constant(path: Path, name: str) -> str:
    """The value of a `NAME = "value"` line, read as text: importing fsr4.ui would load the GPU stack."""
    match = re.search(rf'^{name} = "([^"]+)"$', path.read_text(encoding="utf-8"), re.M)
    if match is None:
        raise SystemExit(f'fsr4.spec: no {name} = "..." line in {path}')
    return match.group(1)


APP_NAME = source_constant(ROOT / "fsr4" / "ui.py", "APP_NAME")
VERSION = source_constant(ROOT / "fsr4" / "__init__.py", "__version__")


def version_numbers(version: str) -> tuple[int, int, int, int]:
    """The four numbers of a Windows file version: "0.1.0" -> (0, 1, 0, 0)."""
    match = re.match(r"\d+(\.\d+){0,3}", version)
    if match is None:
        raise SystemExit(f"fsr4.spec: the version {version!r} does not start with a number")
    numbers = [int(part) for part in match.group(0).split(".")]
    return tuple(numbers + [0] * (4 - len(numbers)))


def version_info(file_name: str) -> VSVersionInfo:
    """An executable's version resource (Explorer's Details tab; Task Manager shows the FileDescription)."""
    numbers = version_numbers(VERSION)
    strings = [
        StringStruct("FileDescription", APP_NAME),
        StringStruct("FileVersion", VERSION),
        StringStruct("InternalName", Path(file_name).stem),
        StringStruct("LegalCopyright", COPYRIGHT),
        StringStruct("LegalTrademarks", TRADEMARKS),
        StringStruct("OriginalFilename", file_name),
        StringStruct("ProductName", APP_NAME),
        StringStruct("ProductVersion", VERSION),
    ]
    return VSVersionInfo(
        ffi=FixedFileInfo(filevers=numbers, prodvers=numbers),
        kids=[
            StringFileInfo([StringTable("040904B0", strings)]),
            VarFileInfo([VarStruct("Translation", [0x0409, 0x04B0])]),
        ],
    )


ICON = Path(workpath) / "fsr4.ico"
ICON.write_bytes(ico_bytes())

analysis = Analysis(
    [str(ROOT / "run_fsr4.py")],
    pathex=[str(ROOT)],
    excludes=["torch", "torchvision", "matplotlib", "PIL", "pytest", *UNUSED_MODULES],
    noarchive=False,
)
# OpenCV's hook bundles its FFmpeg video I/O plugin (31 MB, LGPL); the app only resizes, remaps and computes
# optical flow with cv2, and runs ffmpeg itself for all decoding.
analysis.binaries = [entry for entry in analysis.binaries
                     if not Path(entry[0]).name.lower().startswith("opencv_videoio_ffmpeg")]
archive = PYZ(analysis.pure)
command_line = EXE(archive, analysis.scripts, [], exclude_binaries=True, name="fsr4", console=True,
                   icon=str(ICON), version=version_info("fsr4.exe"))
window = EXE(archive, analysis.scripts, [], exclude_binaries=True, name="Coccinelle Player", console=False,
             icon=str(ICON), version=version_info("Coccinelle Player.exe"))
COLLECT(command_line, window, analysis.binaries, analysis.datas, name="fsr4")
