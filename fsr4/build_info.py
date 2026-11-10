"""Which build of the program runs, so a test can be sure it runs the code it means to.

The packaged executables carry `BUILD_FILE`, which `fsr4.spec` writes with `stamp()`: the commit they were built
from and when. Run from the sources, git is asked for the commit.
"""
from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path

from . import __version__, log

BUILD_FILE = "build.txt"
SOURCE_ROOT = Path(__file__).resolve().parent.parent
CHANGED_MARK = " + changes"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def commit(root: Path = SOURCE_ROOT) -> str | None:
    """The abbreviated commit `root` is at, followed by `CHANGED_MARK` when tracked files differ from it; None
    (logged) when git cannot tell."""
    command = ["git", "-C", str(root), "describe", "--always", f"--dirty={CHANGED_MARK}", "--exclude=*"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, creationflags=CREATE_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired) as error:
        log.emit(f"[build] the commit is unknown, git did not run: {error}")
        return None
    if result.returncode != 0:
        log.emit(f"[build] the commit is unknown: {result.stderr.strip()}")
        return None
    return result.stdout.strip()


def stamp(root: Path = SOURCE_ROOT, now: datetime.datetime | None = None) -> str:
    """What a build records in `BUILD_FILE`: "0e9ae93 · built 2026-11-11 14:32"."""
    when = (now or datetime.datetime.now()).strftime("%Y-%m-%d %H:%M")
    return f"{commit(root) or 'unknown commit'} · built {when}"


def describe() -> str:
    """The version and where it comes from: "0.1.0 · 0e9ae93 · built 2026-11-11 14:32" for the executables,
    "0.1.0 · 0e9ae93 · from the sources" otherwise (`CHANGED_MARK` after the commit: files changed since)."""
    if getattr(sys, "frozen", False):
        path = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / BUILD_FILE
        try:
            return f"{__version__} · {path.read_text(encoding='utf-8').strip()}"
        except OSError as error:
            log.emit(f"[build] error: the build stamp cannot be read: {error}")
            return f"{__version__} · unknown build"
    return f"{__version__} · {commit() or 'unknown commit'} · from the sources"
