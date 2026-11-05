"""What the sides of the window can show: the upscalers with their best settings, and the user's presets.

A preset holds how the video is upscaled: the upscaler, the motion vectors and the sharpening. The output size
and the encoder belong to the file being made, so every side is upscaled at the same size with the same encoder and
the comparison stays like for like. `FAMILIES` are the entries of the side lists that upscale, each starting from its
best settings; the user's own presets are saved as JSON in the user's application data folder (`default_path`).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from . import log
from .hq_scaler import HQ_SCALER
from .realesrgan import MODELS, REAL_ESRGAN
from .upscalers import EXTERNAL_UPSCALERS
from .render import UPSCALER_NAMES, RenderSettings

MAX_NAME_LENGTH = 40
FLOW_PRESETS = ("ultrafast", "fast", "medium")
FILE_VERSION = 1
SOURCE_LABELS = {"nearest": "Source · original pixels", "linear": "Source · bilinear", "cubic": "Source · bicubic",
                 "lanczos": "Source · Lanczos"}
OFF_LABEL = "Off"


@dataclass(frozen=True)
class Preset:
    """How to upscale; `flow` is "dis" (optical flow with `flow_preset`) or "zero" (no motion)."""
    name: str
    upscaler: str | None = None
    flow: str = "dis"
    flow_preset: str = "fast"
    sharpness: float | None = None

    def applied_to(self, settings: RenderSettings) -> RenderSettings:
        """`settings` (size, encoder) upscaled the way this preset says."""
        return replace(settings, upscaler=self.upscaler, flow=self.flow, flow_preset=self.flow_preset,
                       sharpness=self.sharpness)

    def matches(self, settings: RenderSettings) -> bool:
        return (self.upscaler, self.flow, self.flow_preset, self.sharpness) == (
            settings.upscaler, settings.flow, settings.flow_preset, settings.sharpness)


@dataclass(frozen=True)
class Family:
    """An upscaler of the side lists: the settings it starts from and what can be changed in them. `upscalers` are
    its versions or models, best first; with one, there is nothing to choose."""
    name: str
    best: Preset
    upscalers: tuple[str | None, ...]
    motion: bool = False
    sharpening: bool = False


FAMILIES = (
    Family("FSR", Preset("FSR", flow_preset="medium"), (None, "3.1.6", "2.3.4"), motion=True, sharpening=True),
    Family("AMD HQ scaler", Preset("AMD HQ scaler", upscaler=HQ_SCALER, flow="zero"), (HQ_SCALER,), sharpening=True),
    Family("Real-ESRGAN", Preset("Real-ESRGAN", upscaler=REAL_ESRGAN, flow="zero"), tuple(MODELS)),
)
FAMILY_BY_NAME = {family.name: family for family in FAMILIES}
LIST_NAMES = (*FAMILY_BY_NAME, *SOURCE_LABELS.values(), OFF_LABEL)


def family_of(upscaler: str | None) -> Family:
    """The family that offers `upscaler`."""
    return next(family for family in FAMILIES if upscaler in family.upscalers)


def default_path() -> Path:
    base = os.environ.get("APPDATA")
    folder = Path(base) / "Coccinelle Player" if base else Path.home() / ".config" / "coccinelle-player"
    return folder / "presets.json"


def preset_from_json(entry: dict) -> Preset:
    """A user preset from its saved form; raises ValueError when a field is missing or makes no sense."""
    if not isinstance(entry, dict):
        raise ValueError(f"a preset should be an object, not {entry!r}")
    name = entry.get("name")
    if not isinstance(name, str) or not name.strip() or len(name) > MAX_NAME_LENGTH:
        raise ValueError(f"bad preset name {name!r}")
    upscaler = entry.get("upscaler")
    if upscaler not in UPSCALER_NAMES:
        raise ValueError(f"preset {name}: unknown upscaler {upscaler!r}")
    flow, flow_preset = entry.get("flow"), entry.get("flow_preset")
    if flow not in ("dis", "zero") or flow_preset not in FLOW_PRESETS:
        raise ValueError(f"preset {name}: unknown motion {flow!r} {flow_preset!r}")
    if upscaler in EXTERNAL_UPSCALERS:
        flow = "zero"  # they take no motion
    if flow == "zero":
        flow_preset = "fast"  # unused without motion; the settings call "no motion" this way
    sharpness = entry.get("sharpness")
    if sharpness is not None and (not isinstance(sharpness, (int, float)) or not 0.0 <= sharpness <= 1.0):
        raise ValueError(f"preset {name}: sharpening {sharpness!r} is not between 0 and 1")
    if sharpness is None or upscaler in MODELS:  # it has no sharpening
        return Preset(name.strip(), upscaler, flow, flow_preset)
    return Preset(name.strip(), upscaler, flow, flow_preset, float(sharpness))


class PresetStore:
    """The user's presets, which are written to `path` after every change.

    A file that cannot be read is reported in `load_error` and renamed (".unreadable") on the next save, so
    nothing the user made is overwritten.
    """

    def __init__(self, path: Path | None = None):
        self.path = path or default_path()
        self.user: list[Preset] = []
        self.load_error: Exception | None = None
        if self.path.exists():
            self.user = self._read()

    def _read(self) -> list[Preset]:
        """The user's presets as the file has them now; when it cannot be read, `load_error` says why and the
        presets this store already has are kept."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("presets"), list):
                raise ValueError("expected an object with a list of presets")
            return [preset_from_json(entry) for entry in data["presets"]]
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.load_error = error
            log.emit(f"[presets] error: cannot read {self.path}: {error}")
            return self.user

    def _current(self) -> list[Preset]:
        """The user's presets to change: the file's as they are now, since another window may have saved some
        after this one read it."""
        if self.load_error is None and self.path.exists():
            return self._read()
        return self.user

    def find(self, name: str) -> Preset | None:
        return next((preset for preset in self.user if preset.name == name), None)

    def matching(self, settings: RenderSettings) -> Preset | None:
        """The first preset that upscales the way `settings` do."""
        return next((preset for preset in self.user if preset.matches(settings)), None)

    def check_name(self, name: str) -> str | None:
        """Why `name` cannot be used for a new preset, or None when it can."""
        name = name.strip()
        if not name:
            return "Give the preset a name."
        if len(name) > MAX_NAME_LENGTH:
            return f"Use at most {MAX_NAME_LENGTH} characters."
        if any(entry.casefold() == name.casefold() for entry in LIST_NAMES):
            return "The Left and Right lists already have an entry with that name."
        return None

    def save(self, name: str, settings: RenderSettings) -> Preset:
        """Save how `settings` upscale under `name`; a user preset with that name in any case is replaced and
        keeps its spelling."""
        problem = self.check_name(name)
        if problem is not None:
            raise ValueError(problem)
        user = self._current()
        same = [preset for preset in user if preset.name.casefold() == name.strip().casefold()]
        preset = Preset(same[0].name if same else name.strip(), settings.upscaler, settings.flow,
                        settings.flow_preset, settings.sharpness)
        self._write([existing for existing in user if existing not in same] + [preset])
        return preset

    def delete(self, name: str) -> None:
        self._write([preset for preset in self._current() if preset.name != name])

    def _write(self, user: list[Preset]) -> None:
        """Write `user` to the file, then keep it (nothing changes when the write fails)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.load_error is not None and self.path.exists():
            self.path.replace(self.path.with_suffix(".unreadable"))
            self.load_error = None
        entries = [{"name": preset.name, "upscaler": preset.upscaler, "flow": preset.flow,
                    "flow_preset": preset.flow_preset, "sharpness": preset.sharpness} for preset in user]
        temporary = self.path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps({"version": FILE_VERSION, "presets": entries}, indent=2),
                                 encoding="utf-8")
            temporary.replace(self.path)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise
        self.user = user
