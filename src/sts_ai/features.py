"""Feature encoding for combat states and action indices."""

from __future__ import annotations

import numpy as np

from .models import CardType, CombatState, IntentType
from .policy import PlannedAction

MAX_ENEMIES = 5
MAX_HAND_CARDS = 10
ACTION_SPACE = MAX_HAND_CARDS + 1  # card slots + end-turn


def _one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def encode_combat_state(state: CombatState) -> np.ndarray:
    """Encode one combat state to a fixed-size float32 feature vector."""
    features: list[float] = []

    # Global player/combat features.
    features.extend(
        [
            float(state.player.hp),
            float(state.player.max_hp),
            float(state.player.block),
            float(state.player.energy),
            float(state.incoming_damage),
            float(len(state.alive_enemies)),
            float(state.draw_pile),
            float(state.discard_pile),
            float(state.turn),
        ]
    )

    # Enemy slots.
    for idx in range(MAX_ENEMIES):
        if idx < len(state.enemies):
            enemy = state.enemies[idx]
            intent_idx = {
                IntentType.ATTACK: 0,
                IntentType.BLOCK: 1,
                IntentType.BUFF: 2,
                IntentType.DEBUFF: 3,
                IntentType.UNKNOWN: 4,
            }[enemy.intent]
            features.extend(
                [
                    float(enemy.hp),
                    float(enemy.block),
                    float(enemy.intent_value),
                    float(enemy.effective_hp),
                ]
            )
            features.extend(_one_hot(intent_idx, 5))
        else:
            features.extend([0.0] * 9)

    # Hand card slots.
    for idx in range(MAX_HAND_CARDS):
        if idx < len(state.hand):
            card = state.hand[idx]
            card_type_idx = {
                CardType.ATTACK: 0,
                CardType.SKILL: 1,
                CardType.POWER: 2,
            }[card.card_type]
            features.extend(
                [
                    float(card.cost),
                    float(card.damage),
                    float(card.block),
                    float(int(card.exhausts)),
                    float(int(card.upgraded)),
                ]
            )
            features.extend(_one_hot(card_type_idx, 3))
        else:
            features.extend([0.0] * 8)

    return np.asarray(features, dtype=np.float32)


def encode_action(action: PlannedAction | None, state: CombatState) -> int:
    """Map an action to a fixed action index for supervised targets.

    Returns index in [0, MAX_HAND_CARDS] where MAX_HAND_CARDS means end-turn.
    """
    if action is None:
        return MAX_HAND_CARDS

    for idx, card in enumerate(state.hand[:MAX_HAND_CARDS]):
        if card.name.casefold() == action.card_name.casefold():
            return idx
    return MAX_HAND_CARDS
