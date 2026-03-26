"""Screen capture adapter — grabs live frames or loads from disk.

Uses ``mss`` for live screen capture and ``numpy`` for array handling.
``opencv-python`` is optional and only needed for ``imwrite`` saving.

Install the capture extras:  ``pip install -e .[capture]``
"""

from __future__ import annotations

import ctypes
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

    def __init__(
        self,
        monitor_index: int = 1,
        prefer_window_capture: bool = True,
    ) -> None:
        self._monitor_index = monitor_index
        self._prefer_window_capture = prefer_window_capture

    @staticmethod
    def _find_game_window() -> tuple[int, dict[str, int]] | None:
        """Find the Slay the Spire window handle and rectangle."""
        if not hasattr(ctypes, "windll"):
            return None

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        windows: list[tuple[int, str, str]] = []

        process_query_limited_info = 0x1000
        process_vm_read = 0x0010

        enum_proc_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def _get_process_name(hwnd: int) -> str:
            process_id = ctypes.c_ulong(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
            if process_id.value == 0:
                return ""

            handle = kernel32.OpenProcess(
                process_query_limited_info | process_vm_read,
                False,
                process_id.value,
            )
            if not handle:
                return ""

            try:
                max_path = 260
                buffer = ctypes.create_unicode_buffer(max_path)
                if psapi.GetModuleFileNameExW(handle, None, buffer, max_path) > 0:
                    return Path(buffer.value).name.casefold()
            finally:
                kernel32.CloseHandle(handle)
            return ""

        def _enum_proc(hwnd: int, _lparam: int) -> bool:
            if not user32.IsWindowVisible(hwnd):
                return True
            if user32.IsIconic(hwnd):
                return True

            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True

            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = buffer.value.strip()
            if title:
                process_name = _get_process_name(hwnd)
                windows.append((hwnd, title, process_name))
            return True

        user32.EnumWindows(enum_proc_type(_enum_proc), 0)

        process_keywords = ("slaythespire",)
        title_keywords = ("slay the spire", "slay the spire 2")
        excluded_title_tokens = (
            "visual studio code",
            "powershell",
            "terminal",
            "windows powershell",
        )
        excluded_processes = ("code.exe", "windowsterminal.exe", "pwsh.exe", "powershell.exe")
        target_hwnd = None

        for hwnd, title, process_name in windows:
            lowered = title.casefold()
            if process_name in excluded_processes:
                continue
            if any(token in lowered for token in excluded_title_tokens):
                continue
            if any(keyword in process_name for keyword in process_keywords):
                target_hwnd = hwnd
                break

        if target_hwnd is None:
            for hwnd, title, process_name in windows:
                lowered = title.casefold()
                if process_name in excluded_processes:
                    continue
                if any(token in lowered for token in excluded_title_tokens):
                    continue
                if any(keyword in lowered for keyword in title_keywords):
                    target_hwnd = hwnd
                    break

        if target_hwnd is None:
            return None

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        class POINT(ctypes.Structure):
            _fields_ = [
                ("x", ctypes.c_long),
                ("y", ctypes.c_long),
            ]

        client_rect = RECT()
        if user32.GetClientRect(target_hwnd, ctypes.byref(client_rect)):
            top_left = POINT(client_rect.left, client_rect.top)
            bottom_right = POINT(client_rect.right, client_rect.bottom)
            if user32.ClientToScreen(target_hwnd, ctypes.byref(top_left)) and user32.ClientToScreen(
                target_hwnd,
                ctypes.byref(bottom_right),
            ):
                client_width = int(bottom_right.x - top_left.x)
                client_height = int(bottom_right.y - top_left.y)
                if client_width > 0 and client_height > 0:
                    return target_hwnd, {
                        "left": int(top_left.x),
                        "top": int(top_left.y),
                        "width": client_width,
                        "height": client_height,
                    }

        rect = RECT()
        if not user32.GetWindowRect(target_hwnd, ctypes.byref(rect)):
            return None

        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width <= 0 or height <= 0:
            return None

        region = {
            "left": int(rect.left),
            "top": int(rect.top),
            "width": width,
            "height": height,
        }
        return target_hwnd, region

    @staticmethod
    def _find_game_window_region() -> dict[str, int] | None:
        """Find the Slay the Spire window rectangle for focused capture."""
        found = CaptureAdapter._find_game_window()
        if found is None:
            return None
        _, region = found
        return region

    @staticmethod
    def focus_game_window() -> None:
        """Bring the game window to foreground when available."""
        if not hasattr(ctypes, "windll"):
            return

        found = CaptureAdapter._find_game_window()
        if found is None:
            return

        hwnd, _ = found
        user32 = ctypes.windll.user32
        sw_restore = 9
        user32.ShowWindow(hwnd, sw_restore)
        user32.SetForegroundWindow(hwnd)

    # -- live capture (requires ``mss``) ----------------------------------

    def grab_live(self) -> CaptureFrame:
        """Take a screenshot of the game window (or fallback monitor) and return it."""
        try:
            import mss  # noqa: WPS433
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Live capture requires the 'mss' package.  "
                "Install it with:  pip install -e .[capture]"
            ) from exc

        with mss.mss() as sct:
            monitor: dict[str, int] | None = None
            if self._prefer_window_capture:
                monitor = self._find_game_window_region()

            if monitor is None:
                max_index = len(sct.monitors) - 1
                clamped_index = min(max(self._monitor_index, 1), max_index)
                monitor = sct.monitors[clamped_index]

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
