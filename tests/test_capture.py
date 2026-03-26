"""Tests for the capture adapter, parser, and region definitions."""

import json

import numpy as np

import sts_ai.parser as parser_module
from sts_ai.capture import CaptureFrame
from sts_ai.models import GameObservation, GamePhase, ScreenRegion
from sts_ai.parser import (
    _parse_fraction,
    _parse_int,
    _scale_regions_for_frame,
    detect_game_phase,
    extract_neow_highlight_index,
    parse_frame,
)
from sts_ai.policy import HeuristicPolicy
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


def test_detect_game_phase_neow_from_ocr_keywords(monkeypatch) -> None:
    frame = _dummy_frame()

    def _fake_ocr(_frame, _region) -> str:
        return "Neow offers your blessing"

    monkeypatch.setattr(parser_module, "_ocr_region", _fake_ocr)

    phase = detect_game_phase(frame, REGIONS)

    assert phase == GamePhase.NEOW


def test_detect_game_phase_neow_from_architect_phrase(monkeypatch) -> None:
    frame = _dummy_frame()

    def _fake_ocr(_frame, _region) -> str:
        return "...kill... the... ARCHITECT"

    monkeypatch.setattr(parser_module, "_ocr_region", _fake_ocr)

    phase = detect_game_phase(frame, REGIONS)

    assert phase == GamePhase.NEOW


def test_extract_neow_highlight_index_prefers_bright_row() -> None:
    frame = _dummy_frame()
    neow = REGIONS["neow_area"]

    x0 = neow.x + int(neow.w * 0.55)
    x1 = neow.x + int(neow.w * 0.80)
    rows = [
        (neow.y + int(neow.h * 0.20), neow.y + int(neow.h * 0.36)),
        (neow.y + int(neow.h * 0.42), neow.y + int(neow.h * 0.58)),
        (neow.y + int(neow.h * 0.64), neow.y + int(neow.h * 0.80)),
    ]

    for idx, (y0, y1) in enumerate(rows):
        value = 90 if idx != 1 else 210
        frame[y0:y1, x0:x1] = value

    assert extract_neow_highlight_index(frame, REGIONS) == 1


def test_detect_game_phase_proceed_from_ocr_keywords(monkeypatch) -> None:
    frame = _dummy_frame()

    def _fake_ocr(_frame, region) -> str:
        if region.name == "proceed_btn":
            return "Proceed"
        return ""

    monkeypatch.setattr(parser_module, "_ocr_region", _fake_ocr)

    phase = detect_game_phase(frame, REGIONS)

    assert phase == GamePhase.PROCEED


def test_scale_regions_for_frame_scales_dimensions() -> None:
    scaled = _scale_regions_for_frame(REGIONS, frame_width=960, frame_height=540)
    assert scaled["player_hp"].x == REGIONS["player_hp"].x // 2
    assert scaled["player_hp"].y == REGIONS["player_hp"].y // 2


# ---------------------------------------------------------------------------
# CaptureFrame container
# ---------------------------------------------------------------------------


def test_capture_frame_dimensions() -> None:
    pixels = _dummy_frame(800, 600)
    frame = CaptureFrame(pixels=pixels, source="test")
    assert frame.width == 800
    assert frame.height == 600


def test_heuristic_policy_map_fallback_without_nodes() -> None:
    policy = HeuristicPolicy()
    observation = GameObservation(phase=GamePhase.MAP)

    assert policy.choose_map_path(observation) == 3
