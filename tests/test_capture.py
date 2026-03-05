"""Tests for the capture adapter, parser, and region definitions."""

import json

import numpy as np

from sts_ai.capture import CaptureFrame
from sts_ai.models import GameObservation, GamePhase, ScreenRegion
from sts_ai.parser import _parse_fraction, _parse_int, detect_game_phase, parse_frame
from sts_ai.regions import REGIONS, get_region, load_regions

# ---------------------------------------------------------------------------
# Region definitions
# ---------------------------------------------------------------------------


def test_regions_are_all_valid_screen_regions() -> None:
    for name, region in REGIONS.items():
        assert isinstance(region, ScreenRegion)
        assert region.name == name
        assert region.w > 0 and region.h > 0


def test_get_region_returns_known_region() -> None:
    region = get_region("energy")
    assert region.name == "energy"


def test_load_regions_overrides_known_regions(tmp_path) -> None:
    config = {
        "energy": {"x": 99, "y": 88, "w": 77, "h": 66},
        "unknown_region": {"x": 1, "y": 1, "w": 1, "h": 1},
    }
    config_path = tmp_path / "regions.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_regions(config_path)

    assert loaded["energy"].x == 99
    assert "unknown_region" not in loaded
    assert loaded["player_hp"].name == "player_hp"


# ---------------------------------------------------------------------------
# Text-parsing helpers
# ---------------------------------------------------------------------------


def test_parse_int_extracts_number() -> None:
    assert _parse_int("42 gold") == 42
    assert _parse_int("no digits") == 0
    assert _parse_int("") == 0


def test_parse_fraction_extracts_hp() -> None:
    assert _parse_fraction("51/80") == (51, 80)
    assert _parse_fraction("  12 / 99  ") == (12, 99)
    assert _parse_fraction("nothing") == (0, 0)


# ---------------------------------------------------------------------------
# Frame parser
# ---------------------------------------------------------------------------


def _dummy_frame(width: int = 1920, height: int = 1080) -> np.ndarray:
    """Create a blank BGR frame for testing."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_parse_frame_returns_game_observation() -> None:
    frame = _dummy_frame()
    obs = parse_frame(frame)
    assert isinstance(obs, GameObservation)


def test_parse_frame_defaults_to_combat_phase() -> None:
    obs = parse_frame(_dummy_frame())
    assert obs.phase == GamePhase.COMBAT
    assert obs.combat is not None


def test_parse_frame_combat_has_player() -> None:
    obs = parse_frame(_dummy_frame())
    assert obs.combat is not None
    assert obs.combat.player.hp >= 1
    assert obs.combat.player.max_hp >= 1


def test_parse_frame_hand_and_enemies_are_lists() -> None:
    obs = parse_frame(_dummy_frame())
    assert obs.combat is not None
    assert isinstance(obs.combat.hand, list)
    assert isinstance(obs.combat.enemies, list)


def test_detect_game_phase_combat_from_bright_end_turn() -> None:
    frame = _dummy_frame()
    region = REGIONS["end_turn_btn"]
    frame[region.y : region.y + region.h, region.x : region.x + region.w] = 255

    phase = detect_game_phase(frame, REGIONS)

    assert phase == GamePhase.COMBAT


def test_detect_game_phase_reward_from_bright_panel() -> None:
    frame = _dummy_frame()
    region = REGIONS["reward_area"]
    frame[region.y : region.y + region.h, region.x : region.x + region.w] = 180

    phase = detect_game_phase(frame, REGIONS)

    assert phase == GamePhase.REWARD


def test_detect_game_phase_map_from_high_variance_area() -> None:
    frame = _dummy_frame()
    region = REGIONS["map_area"]

    checker = np.indices((region.h, region.w)).sum(axis=0) % 2
    checker = (checker * 255).astype(np.uint8)
    checker_bgr = np.stack([checker, checker, checker], axis=2)
    frame[region.y : region.y + region.h, region.x : region.x + region.w] = checker_bgr

    phase = detect_game_phase(frame, REGIONS)

    assert phase == GamePhase.MAP


# ---------------------------------------------------------------------------
# CaptureFrame container
# ---------------------------------------------------------------------------


def test_capture_frame_dimensions() -> None:
    pixels = _dummy_frame(800, 600)
    frame = CaptureFrame(pixels=pixels, source="test")
    assert frame.width == 800
    assert frame.height == 600
