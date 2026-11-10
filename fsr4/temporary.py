"""Where temporary files go: the pieces of the window's videos and Real-ESRGAN's pictures.

They go in the folder of the file being written (the output), so they sit on the drive that receives the result and
need no setting of their own. Until an output is known, and when its folder cannot take files (read-only, or not
made yet), they go to the system's temporary folder, and the log says why.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from . import log

_folder: Path | None = None


def system_folder() -> Path:
    return Path(tempfile.gettempdir())


def use(path: Path | None) -> Path:
    """Put temporary files in `path` (None: the system's temporary folder) from now on, and return it. The folder
    must exist and take a file; raises OSError when it cannot, and the folder in use stays."""
    global _folder
    target = (Path(path) if path is not None else system_folder()).resolve()
    handle, probe = tempfile.mkstemp(dir=target, prefix="fsr4-check-")
    os.close(handle)
    os.unlink(probe)
    _folder = target
    return target


def use_beside(output: Path) -> Path:
    """Put temporary files in the folder of `output`, or in the system's temporary folder (logged) when that folder
    cannot take them; returns the folder in use."""
    wanted = Path(output).parent
    try:
        return use(wanted)
    except OSError as error:
        fallback = use(None)
        log.emit(f"[temp] error: cannot write to {wanted} ({error}); temporary files go to {fallback}")
        return fallback


def folder() -> Path:
    """The folder in use: the last one `use` took, else the system's temporary folder."""
    return _folder if _folder is not None else system_folder()


def folders_to_clean() -> list[Path]:
    """Where pieces left by a crash are looked for at start: the folder in use and the system's (earlier versions
    put them there). The window also cleans each output folder it starts using."""
    return list(dict.fromkeys((folder(), system_folder())))
