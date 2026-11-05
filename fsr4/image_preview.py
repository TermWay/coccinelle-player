"""Upscales the image open in the window with every set of settings the view shows, on a worker thread.

Like a video, an image is upscaled as soon as it is open, with the current settings and with each preset a
side of the picture shows, so they can be compared before saving (saving runs the job, which gives the same
picture). The worker has its own GPU device and keeps one FSR context per size and upscaler while a shown
version needs it; results are kept for the settings shown and the KEPT_RESULTS most recent others.

Events reach the window through `on_event(kind, *values)`, from the worker thread: "ready" (settings) when a
picture is done, "error" (settings, exception) when one cannot be made, "failed" (exception) when the worker
cannot start (no GPU, unreadable image).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import log
from .flow import create_flow
from .upscalers import create_pipeline
from .media import read_image
from .render import RenderSettings

KEPT_RESULTS = 4


class ImagePreviewer:
    """See the module docstring. Thread-safe; `close()` stops the worker and frees the GPU."""

    def __init__(self, path: Path, sdk_dir: Path, on_event: Callable[..., None]):
        self.path = path
        self.sdk_dir = sdk_dir
        self.on_event = on_event
        self._condition = threading.Condition()
        self._wanted: list[RenderSettings] = []
        self._results: dict[RenderSettings, np.ndarray] = {}
        self._errors: dict[RenderSettings, Exception] = {}
        self._closing = False
        self._thread = threading.Thread(target=self._main, name="image-preview", daemon=True)
        self._thread.start()

    def request(self, wanted: list[RenderSettings]) -> None:
        """The settings the view shows now; asking again retries the ones that failed."""
        with self._condition:
            self._wanted = list(dict.fromkeys(wanted))
            for settings in self._wanted:
                self._errors.pop(settings, None)
            self._condition.notify_all()

    def result(self, settings: RenderSettings) -> np.ndarray | None:
        """The upscaled picture (BGRA) for `settings`, or None while it is not made."""
        with self._condition:
            return self._results.get(settings)

    @property
    def working(self) -> bool:
        """Whether a shown picture is still being made."""
        with self._condition:
            return any(settings not in self._results and settings not in self._errors for settings in self._wanted)

    def close(self) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join()

    def _next(self) -> tuple[RenderSettings, set[tuple]] | None:
        """The next shown picture to make and the pipelines the shown ones need; None once closing."""
        with self._condition:
            while not self._closing:
                missing = [settings for settings in self._wanted
                           if settings not in self._results and settings not in self._errors]
                if missing:
                    return missing[0], {(settings.output_size, settings.upscaler) for settings in self._wanted}
                self._condition.wait()
            return None

    def _keep(self, settings: RenderSettings, picture: np.ndarray) -> None:
        with self._condition:
            self._results[settings] = picture
            others = [kept for kept in self._results if kept not in self._wanted]
            for old in others[:max(0, len(others) - KEPT_RESULTS)]:
                del self._results[old]

    def _main(self) -> None:
        from .gpu import Gpu
        gpu = None
        pipelines: dict[tuple, object] = {}
        try:
            source = read_image(self.path)
            height, width = source.shape[:2]
            gpu = Gpu(verbose=False)
            while (job := self._next()) is not None:
                settings, needed = job
                for key in [key for key in pipelines if key not in needed]:
                    pipelines.pop(key).destroy()
                key = (settings.output_size, settings.upscaler)
                try:
                    pipeline = pipelines.get(key)
                    if pipeline is None:
                        pipeline = create_pipeline(gpu, (width, height), settings.output_size, self.sdk_dir,
                                                   create_flow("zero", width, height), None, settings.upscaler)
                        pipelines[key] = pipeline
                    pipeline.sharpness = settings.sharpness
                    picture = cv2.cvtColor(pipeline.picture(source), cv2.COLOR_RGBA2BGRA)
                except Exception as error:  # noqa: BLE001 - reported; the other pictures can still be made
                    with self._condition:
                        self._errors[settings] = error
                    log.emit(f"[image] error: {error}")
                    self.on_event("error", settings, error)
                    continue
                self._keep(settings, picture)
                self.on_event("ready", settings)
        except Exception as error:  # noqa: BLE001 - no GPU or an unreadable image: reported to the window
            log.emit(f"[image] error: {error}")
            self.on_event("failed", error)
        finally:
            for pipeline in pipelines.values():
                pipeline.destroy()
            if gpu is not None:
                gpu.destroy()
