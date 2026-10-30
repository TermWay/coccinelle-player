"""Motion vectors for FSR from plain video: optical flow between consecutive frames.

FSR wants, per render-resolution pixel, the 2D offset from that pixel to where the same content was in the
previous frame, in pixels. `DisFlow` computes exactly that with OpenCV's DIS optical flow (current -> previous).
`ZeroFlow` is the no-motion baseline. Results are float32 arrays of shape (height, width, 2), uploaded as an
R32G32_SFLOAT image (converting to half floats on the CPU costs more than the flow itself).
"""
from __future__ import annotations

import numpy as np

FLOW_METHODS = ("dis", "zero")


class ZeroFlow:
    """Motion vectors that are all zero: FSR behaves like a jittered spatial upscaler with history blending."""

    name = "zero"

    def __init__(self, width: int, height: int):
        self._zeros = np.zeros((height, width, 2), dtype=np.float32)

    def reset(self) -> None:
        """Forget the previous frame (the next frame starts a new sequence)."""

    def compute(self, current_rgba: np.ndarray) -> np.ndarray:
        return self._zeros


class DisFlow:
    """Dense inverse search optical flow from OpenCV, computed at `1/scale` resolution for speed."""

    name = "dis"

    def __init__(self, width: int, height: int, preset: str = "fast", scale: int = 1):
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("optical flow needs the opencv-python package (pip install opencv-python); "
                               "use --flow zero to run without it") from error
        self.cv2 = cv2
        presets = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST, "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
                   "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}
        if preset not in presets:
            raise ValueError(f"unknown DIS preset {preset}; choose from {', '.join(presets)}")
        if scale < 1:
            raise ValueError("flow scale must be at least 1")
        self.scale = scale
        self.width = width
        self.height = height
        self.flow = cv2.DISOpticalFlow_create(presets[preset])
        self.flow.setUseSpatialPropagation(True)
        self._previous_gray: np.ndarray | None = None
        self._zeros = np.zeros((height, width, 2), dtype=np.float32)

    def _gray(self, rgba: np.ndarray) -> np.ndarray:
        gray = self.cv2.cvtColor(rgba, self.cv2.COLOR_RGBA2GRAY)
        if self.scale > 1:
            gray = self.cv2.resize(gray, (self.width // self.scale, self.height // self.scale),
                                   interpolation=self.cv2.INTER_AREA)
        return gray

    def reset(self) -> None:
        """Forget the previous frame (the next frame starts a new sequence)."""
        self._previous_gray = None

    def compute(self, current_rgba: np.ndarray) -> np.ndarray:
        """Motion vectors from `current_rgba` to the frame given in the previous call (zeros on the first)."""
        current_gray = self._gray(current_rgba)
        if self._previous_gray is None:
            self._previous_gray = current_gray
            return self._zeros
        flow = self.flow.calc(current_gray, self._previous_gray, None)
        self._previous_gray = current_gray
        if self.scale > 1:
            flow = self.cv2.resize(flow, (self.width, self.height), interpolation=self.cv2.INTER_LINEAR)
            flow *= self.scale
        return flow


def create_flow(method: str, width: int, height: int, preset: str = "fast", scale: int = 1):
    """Build the flow provider named by `method` (one of FLOW_METHODS)."""
    if method == "zero":
        return ZeroFlow(width, height)
    if method == "dis":
        return DisFlow(width, height, preset=preset, scale=scale)
    raise ValueError(f"unknown flow method {method}; choose from {', '.join(FLOW_METHODS)}")
