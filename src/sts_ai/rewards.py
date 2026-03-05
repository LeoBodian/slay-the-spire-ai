"""Reward shaping helpers for online data collection."""

from __future__ import annotations

from .models import GameObservation, GamePhase


def combat_win_reward(obs: GameObservation, prev_obs: GameObservation | None) -> float:
    """Reward successful transition out of combat."""
    if prev_obs is None:
        return 0.0
    if prev_obs.phase == GamePhase.COMBAT and obs.phase != GamePhase.COMBAT:
        return 10.0
    return 0.0


def hp_preserved_reward(obs: GameObservation, prev_obs: GameObservation | None) -> float:
    """Reward preserving HP; penalize HP loss."""
    if prev_obs is None or prev_obs.combat is None or obs.combat is None:
        return 0.0
    delta = obs.combat.player.hp - prev_obs.combat.player.hp
    return delta * 0.2


def damage_prevented_reward(obs: GameObservation) -> float:
    """Reward useful block under incoming threat."""
    if obs.combat is None:
        return 0.0
    blocked = min(obs.combat.player.block, obs.combat.incoming_damage)
    return blocked * 0.05


def floor_progress_reward(obs: GameObservation, prev_obs: GameObservation | None) -> float:
    """Reward progression through the run."""
    if prev_obs is None:
        return 0.0
    return max(obs.floor - prev_obs.floor, 0) * 1.0


def compute_reward(
    obs: GameObservation,
    prev_obs: GameObservation | None,
    done: bool,
) -> float:
    """Aggregate shaped rewards and terminal outcomes."""
    reward = 0.0
    reward += combat_win_reward(obs, prev_obs)
    reward += hp_preserved_reward(obs, prev_obs)
    reward += damage_prevented_reward(obs)
    reward += floor_progress_reward(obs, prev_obs)

    if done:
        reward += 50.0 if obs.phase != GamePhase.GAME_OVER else -50.0
    return reward
