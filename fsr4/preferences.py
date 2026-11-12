"""The window's preferences, kept between runs, and the folder where the window keeps its files.

Preferences: whether a file is upscaled as soon as it opens, and how the window was left (its place and size, the
volume, the sidebar folded or not). They are saved as JSON in the user's application data folder, next to the
presets.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
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
    """`window` is the window's place and size when not maximized (left, top, width, height: Tk's geometry, the
    frame's corner and the inside's size), None until the window was first closed."""
    upscale_on_open: bool = True
    window: tuple[int, int, int, int] | None = None
    maximized: bool = False
    volume: float = 0.8
    sidebar_folded: bool = False


def preferences_from_json(data: object) -> Preferences:
    """Preferences from their saved form; raises ValueError when a field makes no sense. Missing fields keep their
    default and fields this version no longer has are ignored, so a file from another version still reads."""
    if not isinstance(data, dict):
        raise ValueError("expected an object")
    defaults = Preferences()
    for name in ("upscale_on_open", "maximized", "sidebar_folded"):
        if not isinstance(data.get(name, False), bool):
            raise ValueError(f"{name} should be true or false, not {data[name]!r}")
    volume = data.get("volume", defaults.volume)
    if isinstance(volume, bool) or not isinstance(volume, (int, float)) or not 0 <= volume <= 1:
        raise ValueError(f"volume should be a number from 0 to 1, not {volume!r}")
    window = data.get("window")
    if window is not None:
        if (not isinstance(window, list) or len(window) != 4
                or not all(isinstance(value, int) and not isinstance(value, bool) for value in window)
                or window[2] <= 0 or window[3] <= 0):
            raise ValueError(f"window should be [left, top, width, height] in pixels, not {window!r}")
        window = tuple(window)
    return Preferences(upscale_on_open=data.get("upscale_on_open", defaults.upscale_on_open), window=window,
                       maximized=data.get("maximized", defaults.maximized), volume=float(volume),
                       sidebar_folded=data.get("sidebar_folded", defaults.sidebar_folded))


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
        write_json(self.path, {"version": FILE_VERSION, **asdict(preferences)})
        self.preferences = preferences
        return preferences
