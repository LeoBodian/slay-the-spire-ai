"""Input automation helpers for interacting with Slay the Spire.

This module intentionally supports a dry-run mode so the rest of the agent
loop can be tested without sending real clicks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .regions import REGIONS


@dataclass(slots=True)
class InputAdapter:
    """Translate high-level actions into game clicks."""

    click_delay: float = 0.3
    dry_run: bool = False

    def _ensure_not_aborted(self) -> None:
        """Abort automation when Escape is currently pressed."""
        try:
            import ctypes  # noqa: WPS433

            if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:
                raise RuntimeError("Escape pressed; aborting input automation")
        except (AttributeError, OSError):
            # Non-Windows environments may not expose user32.
            return

    def click_position(self, x: int, y: int) -> None:
        """Click a specific screen coordinate."""
        self._ensure_not_aborted()
        if self.dry_run:
            return

        try:
            import pyautogui  # noqa: WPS433
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Input automation requires 'pyautogui'. "
                "Install it with: pip install -e .[capture]"
            ) from exc

        pyautogui.click(x=x, y=y)
        time.sleep(self.click_delay)

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
