"""The window's preferences, kept between runs, and the folder where the window keeps its files.

Preferences: whether a file is upscaled as soon as it opens, and where the temporary files go (None: the default,
`temporary.default_folder()`). They are saved as JSON in the user's application data folder, next to the presets.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from . import log

FILE_VERSION = 1


def app_data_folder() -> Path:
    """The folder of the window's files (presets, preferences) in the user's profile."""
    base = os.environ.get("APPDATA")
    return Path(base) / "Coccinelle Player" if base else Path.home() / ".config" / "coccinelle-player"


def write_json(path: Path, data: object) -> None:
    """Write `data` to `path` whole or not at all (through a temporary file next to it); raises OSError."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class Preferences:
    upscale_on_open: bool = True
    temporary_folder: Path | None = None


def preferences_from_json(data: object) -> Preferences:
    """Preferences from their saved form; raises ValueError when a field makes no sense. Missing fields keep their
    default, so a file from an earlier version still reads."""
    if not isinstance(data, dict):
        raise ValueError("expected an object")
    upscale_on_open = data.get("upscale_on_open", True)
    if not isinstance(upscale_on_open, bool):
        raise ValueError(f"upscale_on_open should be true or false, not {upscale_on_open!r}")
    folder = data.get("temporary_folder")
    if folder is not None and (not isinstance(folder, str) or not folder.strip()):
        raise ValueError(f"temporary_folder should be a folder or null, not {folder!r}")
    return Preferences(upscale_on_open, Path(folder) if folder is not None else None)


class PreferenceStore:
    """The preferences, written to `path` after every change.

    A file that cannot be read is reported in `load_error`, the defaults are used, and it is renamed
    (".unreadable") on the next save, like the presets.
    """

    def __init__(self, path: Path | None = None):
        self.path = path or app_data_folder() / "preferences.json"
        self.preferences = Preferences()
        self.load_error: Exception | None = None
        if self.path.exists():
            try:
                self.preferences = preferences_from_json(json.loads(self.path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as error:
                self.load_error = error
                log.emit(f"[preferences] error: cannot read {self.path}: {error}")

    def change(self, **changes) -> Preferences:
        """Save the preferences with `changes` (field=value) and keep them; nothing changes when the write fails
        (OSError)."""
        preferences = replace(self.preferences, **changes)
        if self.load_error is not None and self.path.exists():
            self.path.replace(self.path.with_suffix(".unreadable"))
            self.load_error = None
        folder = preferences.temporary_folder
        write_json(self.path, {"version": FILE_VERSION, "upscale_on_open": preferences.upscale_on_open,
                               "temporary_folder": str(folder) if folder is not None else None})
        self.preferences = preferences
        return preferences
