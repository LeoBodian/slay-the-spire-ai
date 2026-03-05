import pytest

from sts_ai.models import GameObservation, GamePhase
from sts_ai.rewards import (
    combat_win_reward,
    compute_reward,
    damage_prevented_reward,
    floor_progress_reward,
    hp_preserved_reward,
)


def _combat_obs(
    hp: int,
    max_hp: int = 80,
    block: int = 0,
    incoming: int = 0,
    floor: int = 0,
) -> GameObservation:
    from sts_ai.models import CardState, CardType, CombatState, EnemyState, IntentType, PlayerState

    combat = CombatState(
        player=PlayerState(hp=hp, max_hp=max_hp, block=block, energy=1),
        enemies=[
            EnemyState(
                name="Dummy",
                hp=30,
                block=0,
                intent=IntentType.ATTACK,
                intent_value=incoming,
            )
        ],
        hand=[CardState(name="Strike", cost=1, card_type=CardType.ATTACK, damage=6)],
    )
    return GameObservation(phase=GamePhase.COMBAT, combat=combat, floor=floor)


def test_floor_progress_reward_positive() -> None:
    prev_obs = _combat_obs(hp=50, floor=1)
    obs = _combat_obs(hp=50, floor=3)
    assert floor_progress_reward(obs, prev_obs) == 2.0


def test_hp_preserved_reward_penalizes_damage() -> None:
    prev_obs = _combat_obs(hp=50)
    obs = _combat_obs(hp=45)
    assert hp_preserved_reward(obs, prev_obs) < 0


def test_damage_prevented_reward_uses_blocked_damage() -> None:
    obs = _combat_obs(hp=50, block=7, incoming=10)
    assert damage_prevented_reward(obs) == pytest.approx(0.35)


def test_combat_win_reward_on_phase_change() -> None:
    prev_obs = _combat_obs(hp=50)
    obs = GameObservation(phase=GamePhase.REWARD, combat=None)
    assert combat_win_reward(obs, prev_obs) == 10.0


def test_compute_reward_terminal_game_over_negative() -> None:
    prev_obs = _combat_obs(hp=40, floor=10)
    obs = GameObservation(phase=GamePhase.GAME_OVER, combat=None, floor=10)
    assert compute_reward(obs, prev_obs, done=True) <= -40.0
