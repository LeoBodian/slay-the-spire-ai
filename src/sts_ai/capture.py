"""Screen capture adapter — grabs live frames or loads from disk.

Uses ``mss`` for live screen capture and ``numpy`` for array handling.
``opencv-python`` is optional and only needed for ``imwrite`` saving.

Install the capture extras:  ``pip install -e .[capture]``
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .models import GameObservation, ScreenRegion
from .parser import parse_frame

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CaptureFrame:
    """One captured game frame with metadata."""

    pixels: np.ndarray  # BGR uint8 array, shape (H, W, 3)
    source: str  # "live", file path, or replay id
    timestamp: float = field(default_factory=time.time)

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class CaptureAdapter:
    """Grab a game frame from a live screen or a saved image."""

    def __init__(self, monitor_index: int = 0) -> None:
        self._monitor_index = monitor_index

    # -- live capture (requires ``mss``) ----------------------------------

    def grab_live(self) -> CaptureFrame:
        """Take a screenshot of the primary monitor and return it."""
        try:
            import mss  # noqa: WPS433
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Live capture requires the 'mss' package.  "
                "Install it with:  pip install -e .[capture]"
            ) from exc

        with mss.mss() as sct:
            monitor = sct.monitors[self._monitor_index]
            raw = sct.grab(monitor)
            pixels = np.frombuffer(raw.rgb, dtype=np.uint8).reshape(
                raw.height, raw.width, 3
            )
            # mss returns RGB; convert to BGR for OpenCV compatibility
            pixels = pixels[:, :, ::-1].copy()
        return CaptureFrame(pixels=pixels, source="live")

    # -- load from disk ---------------------------------------------------

    def load_image(self, path: str | Path) -> CaptureFrame:
        """Load a saved screenshot from *path*."""
        try:
            import cv2  # noqa: WPS433
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Loading images requires 'opencv-python'.  "
                "Install it with:  pip install -e .[capture]"
            ) from exc

        filepath = Path(path)
        img = cv2.imread(str(filepath))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {filepath}")
        return CaptureFrame(pixels=img, source=str(filepath))

    # -- convenience: capture → parse in one call -------------------------

    def capture_and_parse(
        self,
        source: str | Path | None = None,
        regions: dict[str, ScreenRegion] | None = None,
    ) -> GameObservation:
        """Grab or load a frame, then parse it into a GameObservation."""
        if source is None:
            frame = self.grab_live()
        else:
            frame = self.load_image(source)
        return parse_frame(frame.pixels, regions=regions)

    # -- save a frame to disk (debug / dataset building) ------------------

    @staticmethod
    def save_frame(frame: CaptureFrame, dest: str | Path) -> Path:
        """Write *frame* to *dest* as a PNG."""
        try:
            import cv2  # noqa: WPS433
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Saving images requires 'opencv-python'.  "
                "Install it with:  pip install -e .[capture]"
            ) from exc

        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest_path), frame.pixels)
        return dest_path
