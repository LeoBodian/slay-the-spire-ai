"""Convert a captured frame into a structured GameObservation.

The parser is intentionally split into small, testable extraction functions
that each operate on a cropped region of the screen.  Right now they return
stub / placeholder values.  Replace each stub with real OCR, template
matching, or pixel heuristics as you calibrate against the live game.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

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

_TESSERACT_CONFIGURED = False
_BASE_WIDTH = 1920
_BASE_HEIGHT = 1080

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

    global _TESSERACT_CONFIGURED
    if not _TESSERACT_CONFIGURED:
        candidate = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if candidate.exists() and not os.environ.get("TESSERACT_CMD"):
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
        _TESSERACT_CONFIGURED = True

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(
        resized,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    psm = 6 if (region.w * region.h) >= 120000 else 7

    try:
        text = pytesseract.image_to_string(thresh, config=f"--oem 3 --psm {psm}")
    except Exception:
        return ""

    return text.strip()


def _ocr_numeric_region(
    frame: np.ndarray,
    region: ScreenRegion,
    *,
    allow_slash: bool = False,
) -> str:
    """Run OCR tuned for numeric UI text fields."""
    crop = _safe_crop(frame, region)
    if crop.size == 0:
        return ""

    try:
        import cv2  # noqa: WPS433
        import pytesseract  # noqa: WPS433
    except ModuleNotFoundError:
        return ""

    global _TESSERACT_CONFIGURED
    if not _TESSERACT_CONFIGURED:
        candidate = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if candidate.exists() and not os.environ.get("TESSERACT_CMD"):
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
        _TESSERACT_CONFIGURED = True

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(
        enlarged,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    whitelist = "0123456789/" if allow_slash else "0123456789"
    config = f"--oem 3 --psm 7 -c tessedit_char_whitelist={whitelist}"
    try:
        text = pytesseract.image_to_string(thresh, config=config)
    except Exception:
        return ""
    return text.strip()


def _parse_int(text: str, default: int = 0) -> int:
    """Extract the first integer found in *text*."""
    normalized = re.sub(r"[^\d]", "", text)
    match = re.search(r"\d+", normalized)
    return int(match.group()) if match else default


def _parse_fraction(text: str) -> tuple[int, int]:
    """Parse '51/80' style HP text into (current, max)."""
    cleaned = re.sub(r"[^\d/]", "", text)
    match = re.search(r"(\d+)\s*/\s*(\d+)", cleaned)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


def _sanitize_hp_values(hp: int, max_hp: int) -> tuple[int, int]:
    """Clamp implausible OCR HP values to safe in-game ranges."""
    safe_max = max(max_hp, 1)
    safe_hp = max(hp, 1)
    if safe_hp > safe_max:
        safe_hp = safe_max
    return safe_hp, safe_max


def _contains_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(keyword in lowered for keyword in keywords)


def _scale_regions_for_frame(
    regions: dict[str, ScreenRegion],
    frame_width: int,
    frame_height: int,
) -> dict[str, ScreenRegion]:
    """Scale baseline-calibrated regions to the current frame size."""
    if frame_width <= 0 or frame_height <= 0:
        return regions

    if frame_width == _BASE_WIDTH and frame_height == _BASE_HEIGHT:
        return regions

    scale_x = frame_width / _BASE_WIDTH
    scale_y = frame_height / _BASE_HEIGHT

    return {
        name: ScreenRegion(
            name=region.name,
            x=max(0, int(region.x * scale_x)),
            y=max(0, int(region.y * scale_y)),
            w=max(1, int(region.w * scale_x)),
            h=max(1, int(region.h * scale_y)),
        )
        for name, region in regions.items()
    }


# ---------------------------------------------------------------------------
# Per-field extractors
# ---------------------------------------------------------------------------


def extract_player_hp(
    frame: np.ndarray,
    regions: dict[str, ScreenRegion],
) -> tuple[int, int]:
    text = _ocr_numeric_region(frame, regions["player_hp"], allow_slash=True)
    return _parse_fraction(text)


def extract_player_block(frame: np.ndarray, regions: dict[str, ScreenRegion]) -> int:
    text = _ocr_numeric_region(frame, regions["player_block"])
    return _parse_int(text)


def extract_energy(frame: np.ndarray, regions: dict[str, ScreenRegion]) -> int:
    text = _ocr_numeric_region(frame, regions["energy"])
    return _parse_int(text, default=3)


def extract_gold(frame: np.ndarray, regions: dict[str, ScreenRegion]) -> int:
    base = regions["gold"]
    candidates = [
        base,
        ScreenRegion(
            name="gold_right",
            x=base.x + int(base.w * 0.20),
            y=base.y,
            w=max(1, int(base.w * 0.75)),
            h=base.h,
        ),
        ScreenRegion(
            name="gold_tight",
            x=base.x + int(base.w * 0.30),
            y=base.y + int(base.h * 0.05),
            w=max(1, int(base.w * 0.60)),
            h=max(1, int(base.h * 0.90)),
        ),
    ]

    values: list[int] = []
    for candidate in candidates:
        text = _ocr_numeric_region(frame, candidate)
        value = _parse_int(text, default=0)
        if value > 0:
            values.append(value)

    if not values:
        return 0

    non_outliers = [value for value in values if value <= 300]
    if non_outliers:
        return max(non_outliers)
    return min(values)


def extract_floor(frame: np.ndarray, regions: dict[str, ScreenRegion]) -> int:
    text = _ocr_numeric_region(frame, regions["floor"])
    return _parse_int(text)


def extract_draw_pile(frame: np.ndarray, regions: dict[str, ScreenRegion]) -> int:
    text = _ocr_numeric_region(frame, regions["draw_pile"])
    return _parse_int(text)


def extract_discard_pile(frame: np.ndarray, regions: dict[str, ScreenRegion]) -> int:
    text = _ocr_numeric_region(frame, regions["discard_pile"])
    return _parse_int(text)


def extract_neow_highlight_index(
    frame: np.ndarray,
    regions: dict[str, ScreenRegion],
) -> int | None:
    """Estimate which Neow option row is highlighted (0, 1, or 2)."""
    region = regions.get("neow_area")
    if region is None:
        return None

    # Sample a text-light strip on the right side of each row so OCR text color
    # does not dominate the signal.
    x0 = region.x + int(region.w * 0.55)
    x1 = region.x + int(region.w * 0.80)
    row_top = 0.20
    row_height = 0.16
    row_step = 0.22

    scores: list[float] = []
    for row_index in range(3):
        y0 = region.y + int(region.h * (row_top + row_step * row_index))
        y1 = region.y + int(region.h * (row_top + row_height + row_step * row_index))

        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        row_region = ScreenRegion(name=f"neow_row_{row_index}", x=x0, y=y0, w=width, h=height)
        crop = _safe_crop(frame, row_region)
        if crop.size == 0:
            scores.append(float("-inf"))
            continue

        # Brightness proxy resilient to hue shifts in row artwork.
        scores.append(float(np.max(crop, axis=2).mean()))

    if not scores or all(score == float("-inf") for score in scores):
        return None
    return int(np.argmax(scores))


def detect_game_phase(frame: np.ndarray, regions: dict[str, ScreenRegion]) -> GamePhase:
    """Detect the current game phase from broad pixel heuristics."""
    end_turn = _safe_crop(frame, regions["end_turn_btn"])
    reward = _safe_crop(frame, regions["reward_area"])
    map_area = _safe_crop(frame, regions["map_area"])
    proceed_region = regions.get("proceed_btn")

    neow_region = regions.get("neow_area", regions["reward_area"])
    neow_text = _ocr_region(frame, neow_region)
    if _contains_keyword(
        neow_text,
        (
            "neow",
            "blessing",
            "your boon",
            "choose",
            "lose your",
            "awaken",
            "architect",
            "kill.. the",
            "kill... the",
        ),
    ):
        return GamePhase.NEOW

    if end_turn.size > 0 and float(end_turn.mean()) > 140.0:
        return GamePhase.COMBAT

    if proceed_region is not None:
        proceed_text = _ocr_region(frame, proceed_region)
        if _contains_keyword(proceed_text, ("proceed", "continue", "next")):
            return GamePhase.PROCEED

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
    base_regions = regions or REGIONS
    frame_height, frame_width = frame.shape[:2]
    active_regions = _scale_regions_for_frame(base_regions, frame_width, frame_height)
    phase = detect_game_phase(frame, active_regions)

    hp, max_hp = extract_player_hp(frame, active_regions)
    hp, max_hp = _sanitize_hp_values(hp, max_hp)
    block = extract_player_block(frame, active_regions)
    energy = extract_energy(frame, active_regions)
    gold = extract_gold(frame, active_regions)
    floor = extract_floor(frame, active_regions)

    raw_texts: dict[str, str] = {}
    if phase == GamePhase.NEOW:
        neow_region = active_regions.get("neow_area", active_regions["reward_area"])
        raw_texts["neow"] = _ocr_region(frame, neow_region)
        highlight_index = extract_neow_highlight_index(frame, active_regions)
        if highlight_index is not None:
            raw_texts["neow_highlight_index"] = str(highlight_index)

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
        raw_texts=raw_texts,
    )
