"""Default screen-region definitions for a 1920×1080 Slay the Spire 2 window.

These are starting-point rectangles.  Adjust them for your actual resolution
and UI layout by editing the REGIONS dict or by loading from a JSON config.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import ScreenRegion

# (x, y, width, height) — calibrated on a 1920×1080 window.
# Tweak these or load overrides from a config file for your setup.
REGIONS: dict[str, ScreenRegion] = {
    "player_hp": ScreenRegion(name="player_hp", x=350, y=860, w=130, h=36),
    "player_block": ScreenRegion(
        name="player_block", x=310, y=810, w=60, h=36
    ),
    "energy": ScreenRegion(name="energy", x=152, y=870, w=70, h=50),
    "hand_area": ScreenRegion(name="hand_area", x=250, y=920, w=1420, h=160),
    "draw_pile": ScreenRegion(name="draw_pile", x=50, y=960, w=80, h=40),
    "discard_pile": ScreenRegion(
        name="discard_pile", x=1790, y=960, w=80, h=40
    ),
    "enemy_area": ScreenRegion(
        name="enemy_area", x=900, y=200, w=900, h=500
    ),
    "enemy_intent_area": ScreenRegion(
        name="enemy_intent_area", x=900, y=140, w=900, h=80
    ),
    "gold": ScreenRegion(name="gold", x=1700, y=60, w=120, h=36),
    "floor": ScreenRegion(name="floor", x=20, y=60, w=100, h=36),
    "map_area": ScreenRegion(name="map_area", x=200, y=100, w=1520, h=880),
    "reward_area": ScreenRegion(
        name="reward_area", x=560, y=250, w=800, h=500
    ),
    "end_turn_btn": ScreenRegion(
        name="end_turn_btn", x=1630, y=860, w=190, h=60
    ),
}


def get_region(name: str) -> ScreenRegion:
    """Return a region by name, raising KeyError if unknown."""
    return REGIONS[name]


def load_regions(path: str | Path) -> dict[str, ScreenRegion]:
    """Load region overrides from JSON and merge onto defaults.

    Expected JSON structure:
    {
      "energy": {"x": 100, "y": 850, "w": 70, "h": 50},
      "player_hp": {"x": 330, "y": 850, "w": 150, "h": 40}
    }
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    merged = {name: region.model_copy() for name, region in REGIONS.items()}
    for name, payload in raw.items():
        if name not in merged:
            continue
        merged[name] = ScreenRegion(name=name, **payload)
    return merged
