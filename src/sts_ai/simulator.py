"""Approximate combat forward model for search experiments."""

from __future__ import annotations

from .features import MAX_HAND_CARDS
from .models import CombatState

END_TURN_ACTION = MAX_HAND_CARDS


def legal_action_indices(state: CombatState) -> list[int]:
    """Return legal action indices for one decision point."""
    legal: list[int] = []
    for idx, card in enumerate(state.hand[:MAX_HAND_CARDS]):
        if card.cost <= state.player.energy:
            legal.append(idx)
    legal.append(END_TURN_ACTION)
    return legal


def _pick_target_index(state: CombatState) -> int | None:
    if not state.alive_enemies:
        return None
    target = max(state.alive_enemies, key=lambda enemy: (enemy.intent_value, enemy.effective_hp))
    for idx, enemy in enumerate(state.enemies):
        if enemy.name == target.name and enemy.hp == target.hp and enemy.block == target.block:
            return idx
    return None


def _apply_damage(enemy, damage: int) -> None:
    blocked = min(enemy.block, damage)
    enemy.block -= blocked
    remaining = damage - blocked
    enemy.hp = max(enemy.hp - remaining, 0)


def simulate_action(state: CombatState, action_index: int) -> CombatState:
    """Simulate one card play (or end-turn) and return next state."""
    next_state = state.model_copy(deep=True)

    if action_index == END_TURN_ACTION:
        next_state.player.block = 0
        next_state.turn += 1
        return next_state

    if action_index < 0 or action_index >= len(next_state.hand):
        return next_state

    card = next_state.hand[action_index]
    if card.cost > next_state.player.energy:
        return next_state

    next_state.player.energy -= card.cost

    if card.damage > 0:
        target_index = _pick_target_index(next_state)
        if target_index is not None:
            _apply_damage(next_state.enemies[target_index], card.damage)
            next_state.enemies = [enemy for enemy in next_state.enemies if enemy.hp > 0]

    if card.block > 0:
        next_state.player.block += card.block

    next_state.hand.pop(action_index)
    return next_state
