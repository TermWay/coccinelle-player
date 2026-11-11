"""Keep the development build in dist\\fsr4 up to date with the sources, without disturbing a running copy.

    python tools/dev_build.py          build when a source is newer than the build, then put it in place
    python tools/dev_build.py --force  build even when it is up to date

The build goes into out\\dev-build (build_exe.bat with that folder), so a window running from dist\\fsr4 neither
blocks nor breaks it; it then replaces dist\\fsr4 when no program runs from there. Otherwise it waits in
out\\dev-build and the next run puts it in place (start.bat runs this before it opens the window).
What was added to dist\\fsr4 by hand (`KEPT`: Real-ESRGAN, ffmpeg) moves into the new build.
Exit status: 0 when dist\\fsr4 is up to date, 2 when the new build waits because dist\\fsr4 is in use, 1 when the
build failed.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist" / "fsr4"
STAGING_ROOT = ROOT / "out" / "dev-build"
STAGING = STAGING_ROOT / "fsr4"
WORK = ROOT / "out" / "dev-build-work"  # PyInstaller's own, apart from build_exe.bat's default `build`
EXECUTABLE = "Coccinelle Player.exe"
# What the build is made from or holds: a newer one makes the build out of date.
INPUT_FOLDERS = ["fsr4", "third_party"]
INPUT_FILES = ["run_fsr4.py", "fsr4.spec", "build_exe.bat", "requirements.txt", "README.md", "LICENSE",
               "tools/collect_licenses.py", "bin/amd_fidelityfx_loader_vk.dll", "bin/amd_fidelityfx_upscaler_vk.dll",
               "bin/license.md"]
# What the README says to put into the build's folder by hand: never built, so a new build carries it over.
KEPT = ["bin/realesrgan", "ffmpeg.exe", "ffprobe.exe"]
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.K32EnumProcesses.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                ctypes.POINTER(wintypes.DWORD)]
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


def newest_input() -> float:
    """The latest modification time among the build's inputs (compiled caches left out)."""
    paths = [ROOT / name for name in INPUT_FILES if (ROOT / name).exists()]
    for folder in INPUT_FOLDERS:
        paths += [path for path in (ROOT / folder).rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    return max(path.stat().st_mtime for path in paths)


def built_at(folder: Path) -> float | None:
    """When the build in `folder` was started (its window executable's time), or None without one."""
    executable = folder / EXECUTABLE
    return executable.stat().st_mtime if executable.exists() else None


def processes_in(folder: Path) -> list[str]:
    """The programs running from `folder`, as "path (process id)"."""
    ids = (wintypes.DWORD * 8192)()
    returned = wintypes.DWORD()
    if not kernel32.K32EnumProcesses(ids, ctypes.sizeof(ids), ctypes.byref(returned)):
        raise ctypes.WinError(ctypes.get_last_error())
    prefix = str(folder.resolve()).lower() + os.sep
    found = []
    for process_id in ids[:returned.value // ctypes.sizeof(wintypes.DWORD)]:
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, process_id)
        if not handle:
            continue  # system and elevated processes refuse; none of them runs from the build
        try:
            name = ctypes.create_unicode_buffer(32768)
            size = wintypes.DWORD(len(name))
            if kernel32.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size)) \
                    and name.value.lower().startswith(prefix):
                found.append(f"{name.value} ({process_id})")
        finally:
            kernel32.CloseHandle(handle)
    return found


def build() -> bool:
    """Build into out\\dev-build with build_exe.bat; its executable is stamped with the time the build started, so
    a source saved meanwhile still counts as newer."""
    started = time.time()
    script = ROOT / "build_exe.bat"
    result = subprocess.run(["cmd", "/c", str(script), str(STAGING_ROOT), str(WORK)], cwd=ROOT,
                            stdin=subprocess.DEVNULL)  # no keyboard: the script's pauses return at once
    if result.returncode != 0 or built_at(STAGING) is None:
        print(f"dev_build: the build failed (exit {result.returncode}), see the messages above.", flush=True)
        return False
    os.utime(STAGING / EXECUTABLE, (started, started))
    return True


def install() -> bool:
    """Put the build waiting in out\\dev-build in place of dist\\fsr4; False when dist\\fsr4 is in use."""
    users = processes_in(DIST) if DIST.exists() else []
    if users:
        print("dev_build: the new build waits in out\\dev-build while these run from dist\\fsr4: "
              + ", ".join(users), flush=True)
        return False
    previous = DIST.with_name("fsr4.previous")
    if previous.exists():
        shutil.rmtree(previous)
    if DIST.exists():
        try:
            DIST.rename(previous)
        except OSError as error:
            print(f"dev_build: dist\\fsr4 is in use ({error}); the new build waits in out\\dev-build.", flush=True)
            return False
    DIST.parent.mkdir(exist_ok=True)
    try:
        STAGING.rename(DIST)
    except OSError:
        if previous.exists():
            previous.rename(DIST)
        raise
    if previous.exists():
        for name in KEPT:
            if (previous / name).exists() and not (DIST / name).exists():
                (DIST / name).parent.mkdir(parents=True, exist_ok=True)
                (previous / name).rename(DIST / name)
                print(f"dev_build: kept dist\\fsr4\\{Path(name)}", flush=True)
        shutil.rmtree(previous)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="build even when the build is up to date")
    args = parser.parse_args()
    newest = newest_input()
    current = [folder for folder in (DIST, STAGING) if (built_at(folder) or 0) >= newest]
    if args.force or not current:
        print("dev_build: " + ("building" if args.force else "the sources changed since the last build, building")
              + " dist\\fsr4 ...", flush=True)
        if not build():
            return 1
    staged, installed = built_at(STAGING), built_at(DIST)
    if staged is not None and staged > (installed or 0):
        if not install():
            return 2
        print("dev_build: dist\\fsr4 holds the new build.", flush=True)
    else:
        if STAGING.exists():
            shutil.rmtree(STAGING)  # an older or unfinished build: nothing to keep
        print("dev_build: dist\\fsr4 is up to date.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
