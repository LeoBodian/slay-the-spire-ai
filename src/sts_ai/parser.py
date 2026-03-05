"""Convert a captured frame into a structured GameObservation.

The parser is intentionally split into small, testable extraction functions
that each operate on a cropped region of the screen.  Right now they return
stub / placeholder values.  Replace each stub with real OCR, template
matching, or pixel heuristics as you calibrate against the live game.
"""

from __future__ import annotations

import re

import numpy as np

from .models import (
    CardState,
    CombatState,
    EnemyState,
    GameObservation,
    GamePhase,
    PlayerState,
    ScreenRegion,
)
from .regions import REGIONS

# ---------------------------------------------------------------------------
# Low-level helpers (stubs — swap in real OCR later)
# ---------------------------------------------------------------------------


def _safe_crop(frame: np.ndarray, region: ScreenRegion) -> np.ndarray:
    """Crop a region from a frame, clamping coordinates to bounds."""
    height, width = frame.shape[:2]
    x0 = max(0, min(region.x, width))
    y0 = max(0, min(region.y, height))
    x1 = max(0, min(region.x + region.w, width))
    y1 = max(0, min(region.y + region.h, height))
    return frame[y0:y1, x0:x1]


def _ocr_region(
    frame: np.ndarray, region: ScreenRegion
) -> str:
    """Run OCR on a cropped region and return the raw text.

    If OCR dependencies are not installed, this returns an empty string so
    other parser functions can still run in stub mode.
    """
    crop = _safe_crop(frame, region)
    if crop.size == 0:
        return ""

    try:
        import cv2  # noqa: WPS433
        import pytesseract  # noqa: WPS433
    except ModuleNotFoundError:
        return ""

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(
        resized,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    try:
        text = pytesseract.image_to_string(thresh, config="--oem 3 --psm 7")
    except Exception:
        return ""

    return text.strip()


def _parse_int(text: str, default: int = 0) -> int:
    """Extract the first integer found in *text*."""
    match = re.search(r"\d+", text)
    return int(match.group()) if match else default


def _parse_fraction(text: str) -> tuple[int, int]:
    """Parse '51/80' style HP text into (current, max)."""
    match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


# ---------------------------------------------------------------------------
# Per-field extractors
# ---------------------------------------------------------------------------


def extract_player_hp(
    frame: np.ndarray,
    regions: dict[str, ScreenRegion],
) -> tuple[int, int]:
    text = _ocr_region(frame, regions["player_hp"])
    return _parse_fraction(text)


def extract_player_block(frame: np.ndarray, regions: dict[str, ScreenRegion]) -> int:
    text = _ocr_region(frame, regions["player_block"])
    return _parse_int(text)


def extract_energy(frame: np.ndarray, regions: dict[str, ScreenRegion]) -> int:
    text = _ocr_region(frame, regions["energy"])
    return _parse_int(text, default=3)


def extract_gold(frame: np.ndarray, regions: dict[str, ScreenRegion]) -> int:
    text = _ocr_region(frame, regions["gold"])
    return _parse_int(text)


def extract_floor(frame: np.ndarray, regions: dict[str, ScreenRegion]) -> int:
    text = _ocr_region(frame, regions["floor"])
    return _parse_int(text)


def extract_draw_pile(frame: np.ndarray, regions: dict[str, ScreenRegion]) -> int:
    text = _ocr_region(frame, regions["draw_pile"])
    return _parse_int(text)


def extract_discard_pile(frame: np.ndarray, regions: dict[str, ScreenRegion]) -> int:
    text = _ocr_region(frame, regions["discard_pile"])
    return _parse_int(text)


def detect_game_phase(frame: np.ndarray, regions: dict[str, ScreenRegion]) -> GamePhase:
    """Detect the current game phase from broad pixel heuristics."""
    end_turn = _safe_crop(frame, regions["end_turn_btn"])
    reward = _safe_crop(frame, regions["reward_area"])
    map_area = _safe_crop(frame, regions["map_area"])

    if end_turn.size > 0 and float(end_turn.mean()) > 140.0:
        return GamePhase.COMBAT

    if reward.size > 0 and float(reward.mean()) > 120.0 and float(reward.std()) < 70.0:
        return GamePhase.REWARD

    if map_area.size > 0 and float(map_area.std()) > 60.0:
        return GamePhase.MAP

    # Keep COMBAT as fallback so legacy tests and behavior remain stable.
    return GamePhase.COMBAT


def extract_hand(frame: np.ndarray) -> list[CardState]:
    """Extract the list of cards in the player's hand.

    Stub: returns an empty hand.  Replace with card-region
    segmentation + per-card OCR / template matching.
    """
    return []


def extract_enemies(frame: np.ndarray) -> list[EnemyState]:
    """Extract enemy names, HP, block, and intents.

    Stub: returns an empty list.  Replace with enemy-region
    detection + HP bar parsing + intent icon classification.
    """
    return []


# ---------------------------------------------------------------------------
# Top-level parser
# ---------------------------------------------------------------------------


def parse_frame(
    frame: np.ndarray,
    regions: dict[str, ScreenRegion] | None = None,
) -> GameObservation:
    """Parse a full game screenshot into a structured observation.

    Call this with a BGR numpy array (as returned by ``cv2.imread``
    or ``mss`` + ``numpy``).  Each sub-extractor is a stub that you
    replace one at a time as you calibrate.
    """
    active_regions = regions or REGIONS
    phase = detect_game_phase(frame, active_regions)

    hp, max_hp = extract_player_hp(frame, active_regions)
    block = extract_player_block(frame, active_regions)
    energy = extract_energy(frame, active_regions)
    gold = extract_gold(frame, active_regions)
    floor = extract_floor(frame, active_regions)

    combat: CombatState | None = None
    if phase == GamePhase.COMBAT:
        player = PlayerState(
            hp=max(hp, 1),
            max_hp=max(max_hp, 1),
            block=block,
            energy=energy,
        )
        combat = CombatState(
            player=player,
            enemies=extract_enemies(frame),
            hand=extract_hand(frame),
            draw_pile=extract_draw_pile(frame, active_regions),
            discard_pile=extract_discard_pile(frame, active_regions),
        )

    return GameObservation(
        phase=phase,
        combat=combat,
        floor=floor,
        gold=gold,
    )
