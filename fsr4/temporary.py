"""Where temporary files go: the pieces of the window's videos and Real-ESRGAN's pictures.

By default the `temp` folder next to the executable (next to the `fsr4` package when run from source), so they stay
with the program instead of filling the system drive; the window's preferences can name another folder. When the
default folder cannot be made or written to (a read-only install), the system's temporary folder takes its place
and the log says so.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from . import log

DEFAULT_NAME = "temp"

_folder: Path | None = None


def program_folder() -> Path:
    """The folder of the executable, or of the repository when run from source."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def default_folder() -> Path:
    return program_folder() / DEFAULT_NAME


def use(path: Path | None) -> Path:
    """Put temporary files in `path` (None: `default_folder()`) from now on, and return it. The folder is made when
    needed and must take a file; raises OSError when it cannot, and the folder in use stays."""
    global _folder
    target = (Path(path) if path is not None else default_folder()).resolve()
    target.mkdir(parents=True, exist_ok=True)
    handle, probe = tempfile.mkstemp(dir=target, prefix="fsr4-check-")
    os.close(handle)
    os.unlink(probe)
    _folder = target
    return target


def folder() -> Path:
    """The folder in use. Until `use` names one, the default folder, or the system's when it cannot be used."""
    global _folder
    if _folder is None:
        try:
            use(None)
        except OSError as error:
            _folder = Path(tempfile.gettempdir())
            log.emit(f"[temp] error: cannot use {default_folder()} ({error}); temporary files go to {_folder}")
    return _folder


def folders_to_clean() -> list[Path]:
    """Where pieces left by a crash can be: the folder in use, the default one and the system's (earlier versions
    put them there)."""
    return list(dict.fromkeys((folder(), default_folder(), Path(tempfile.gettempdir()))))
