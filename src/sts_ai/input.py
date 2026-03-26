"""Input automation helpers for interacting with Slay the Spire.

This module intentionally supports a dry-run mode so the rest of the agent
loop can be tested without sending real clicks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .capture import CaptureAdapter
from .regions import REGIONS


@dataclass(slots=True)
class InputAdapter:
    """Translate high-level actions into game clicks."""

    click_delay: float = 0.3
    dry_run: bool = False
    base_width: int = 1920
    base_height: int = 1080

    def _ensure_not_aborted(self) -> None:
        """Abort automation when Escape is currently pressed."""
        try:
            import ctypes  # noqa: WPS433

            if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:
                raise RuntimeError("Escape pressed; aborting input automation")
        except (AttributeError, OSError):
            # Non-Windows environments may not expose user32.
            return

    def _to_screen_coords(self, x: int, y: int) -> tuple[int, int]:
        """Convert logical game-space coordinates to real screen coordinates."""
        region = CaptureAdapter._find_game_window_region()
        if not region:
            return x, y

        width = max(region["width"], 1)
        height = max(region["height"], 1)
        scaled_x = region["left"] + int((x / self.base_width) * width)
        scaled_y = region["top"] + int((y / self.base_height) * height)
        return scaled_x, scaled_y

    def click_position(self, x: int, y: int) -> None:
        """Click a specific screen coordinate."""
        self._ensure_not_aborted()
        if self.dry_run:
            return

        CaptureAdapter.focus_game_window()
        screen_x, screen_y = self._to_screen_coords(x, y)

        try:
            import ctypes  # noqa: WPS433

            user32 = ctypes.windll.user32
            # Move first, then emit low-level left-button down/up.
            if user32.SetCursorPos(screen_x, screen_y):
                mouseeventf_leftdown = 0x0002
                mouseeventf_leftup = 0x0004
                user32.mouse_event(mouseeventf_leftdown, 0, 0, 0, 0)
                time.sleep(0.03)
                user32.mouse_event(mouseeventf_leftup, 0, 0, 0, 0)
                time.sleep(self.click_delay)
                return
        except (AttributeError, OSError):
            pass

        try:
            import pyautogui  # noqa: WPS433
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Input automation requires 'pyautogui'. "
                "Install it with: pip install -e .[capture]"
            ) from exc

        pyautogui.click(x=screen_x, y=screen_y)
        time.sleep(self.click_delay)

    def hover_position(self, x: int, y: int) -> None:
        """Move cursor to a specific coordinate without clicking."""
        self._ensure_not_aborted()
        if self.dry_run:
            return

        CaptureAdapter.focus_game_window()
        screen_x, screen_y = self._to_screen_coords(x, y)

        try:
            import ctypes  # noqa: WPS433

            user32 = ctypes.windll.user32
            if user32.SetCursorPos(screen_x, screen_y):
                time.sleep(min(self.click_delay, 0.15))
                return
        except (AttributeError, OSError):
            pass

        try:
            import pyautogui  # noqa: WPS433
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Cursor movement requires native Windows input or 'pyautogui'. "
                "Install it with: pip install -e .[capture]"
            ) from exc

        pyautogui.moveTo(x=screen_x, y=screen_y)
        time.sleep(min(self.click_delay, 0.15))

    def _press_virtual_key(self, vk_code: int) -> bool:
        """Press and release one Windows virtual key; return True on success."""
        try:
            import ctypes  # noqa: WPS433

            user32 = ctypes.windll.user32
            keyeventf_keyup = 0x0002
            user32.keybd_event(vk_code, 0, 0, 0)
            time.sleep(0.03)
            user32.keybd_event(vk_code, 0, keyeventf_keyup, 0)
            time.sleep(self.click_delay)
            return True
        except (AttributeError, OSError):
            return False

    def click_card(self, index: int, hand_size: int) -> None:
        """Click the card at *index* in the current hand."""
        region = REGIONS["hand_area"]
        slots = max(hand_size, 1)
        slot_width = region.w / slots
        x = int(region.x + slot_width * (index + 0.5))
        y = int(region.y + region.h * 0.5)
        self.click_position(x, y)

    def click_enemy(self, index: int, enemy_count: int) -> None:
        """Click the enemy at *index* in the enemy area."""
        region = REGIONS["enemy_area"]
        slots = max(enemy_count, 1)
        slot_width = region.w / slots
        x = int(region.x + slot_width * (index + 0.5))
        y = int(region.y + region.h * 0.5)
        self.click_position(x, y)

    def click_end_turn(self) -> None:
        """Click the End Turn button."""
        region = REGIONS["end_turn_btn"]
        x = int(region.x + region.w * 0.5)
        y = int(region.y + region.h * 0.5)
        self.click_position(x, y)

    def click_proceed_button(self) -> None:
        """Click the Proceed/Continue button."""
        region = REGIONS["proceed_btn"]
        x = int(region.x + region.w * 0.5)
        y = int(region.y + region.h * 0.5)
        self.click_position(x, y)

    def highlight_proceed_button(self) -> None:
        """Hover the Proceed/Continue button so it highlights before confirm."""
        region = REGIONS["proceed_btn"]
        x = int(region.x + region.w * 0.5)
        y = int(region.y + region.h * 0.5)
        self.hover_position(x, y)

    def click_map_node(self, index: int, node_count: int) -> None:
        """Click one map node by index within the map area."""
        region = REGIONS["map_area"]
        slots = max(node_count, 1)
        slot_width = region.w / slots
        x = int(region.x + slot_width * (index + 0.5))
        y = int(region.y + region.h * 0.5)
        self.click_position(x, y)

    def click_reward_option(self, index: int, option_count: int) -> None:
        """Click one reward option within the reward panel."""
        region = REGIONS["reward_area"]
        slots = max(option_count, 1)
        slot_height = region.h / slots
        x = int(region.x + region.w * 0.5)
        y = int(region.y + slot_height * (index + 0.5))
        self.click_position(x, y)

    def click_rest_action(self, action: str) -> None:
        """Click either rest or smith within the reward/rest area."""
        region = REGIONS["reward_area"]
        is_rest = action.casefold() == "rest"
        x = int(region.x + region.w * (0.3 if is_rest else 0.7))
        y = int(region.y + region.h * 0.7)
        self.click_position(x, y)

    def click_neow_option(self, index: int, option_count: int) -> None:
        """Click one Neow choice by vertical slot index."""
        region = REGIONS.get("neow_area", REGIONS["reward_area"])
        slots = max(option_count, 3)
        # In STS2 Neow, clickable rows sit left of center under the speech bubble.
        x = int(region.x + region.w * 0.18)
        top_ratio = 0.14
        lane_height = 0.22
        clamped_index = max(0, min(index, slots - 1))
        y_ratio = top_ratio + (lane_height * clamped_index)
        y = int(region.y + region.h * y_ratio)
        self.click_position(x, y)

    def click_neow_continue(self) -> None:
        """Click the initial Neow continue/awaken button area."""
        region = REGIONS.get("neow_area", REGIONS["reward_area"])
        x = int(region.x + region.w * 0.5)
        y = int(region.y + region.h * 0.82)
        self.click_position(x, y)

    def press_neow_option_hotkey(self, index: int) -> None:
        """Select a Neow option with keyboard hotkeys 1/2/3."""
        CaptureAdapter.focus_game_window()
        clamped = max(0, min(index, 2))
        vk_1 = 0x31
        if self._press_virtual_key(vk_1 + clamped):
            return

        try:
            import pyautogui  # noqa: WPS433
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Neow hotkeys require native Windows input or 'pyautogui'. "
                "Install it with: pip install -e .[capture]"
            ) from exc

        pyautogui.press(str(clamped + 1))
        time.sleep(self.click_delay)

    def highlight_neow_option_with_arrows(self, index: int) -> None:
        """Move Neow highlight to one of three options using arrow keys."""
        CaptureAdapter.focus_game_window()
        clamped = max(0, min(index, 2))

        # Prime focus: STS2 can start keyboard focus on top HUD elements.
        # A few down/up sweeps pulls focus into the Neow option list reliably.
        vk_up = 0x26
        vk_down = 0x28
        for _ in range(3):
            self._press_virtual_key(vk_down)
        for _ in range(3):
            self._press_virtual_key(vk_up)

        # Reset to the top option, then step down to the target option.
        vk_home = 0x24
        self._press_virtual_key(vk_home)
        for _ in range(clamped):
            self._press_virtual_key(vk_down)

    def confirm_with_enter(self) -> None:
        """Press Enter to confirm the currently highlighted option."""
        CaptureAdapter.focus_game_window()
        vk_enter = 0x0D
        self._press_virtual_key(vk_enter)
