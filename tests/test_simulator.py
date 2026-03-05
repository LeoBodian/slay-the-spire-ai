import numpy as np

from sts_ai.features import ACTION_SPACE
from sts_ai.models import CardState, CardType, CombatState, EnemyState, IntentType, PlayerState
from sts_ai.search import SearchPolicy, beam_search
from sts_ai.simulator import END_TURN_ACTION, legal_action_indices, simulate_action


class DummyScorer:
    def action_scores(self, state: CombatState) -> np.ndarray:
        scores = np.zeros(ACTION_SPACE, dtype=np.float32)
        for idx, card in enumerate(state.hand):
            scores[idx] = float(card.damage + card.block)
        scores[END_TURN_ACTION] = 0.01
        return scores

    def evaluate_state(self, state: CombatState) -> float:
        # Prefer states with lower enemy effective HP and more player HP.
        enemy_total = sum(enemy.effective_hp for enemy in state.enemies)
        return float(state.player.hp - enemy_total)


def _state() -> CombatState:
    return CombatState(
        player=PlayerState(hp=50, max_hp=80, block=0, energy=1),
        enemies=[
            EnemyState(
                name="Cultist",
                hp=8,
                block=0,
                intent=IntentType.ATTACK,
                intent_value=6,
            )
        ],
        hand=[
            CardState(name="Strike", cost=1, card_type=CardType.ATTACK, damage=6),
            CardState(name="Defend", cost=1, card_type=CardType.SKILL, block=5),
        ],
    )


def test_legal_action_indices_contains_end_turn() -> None:
    state = _state()
    legal = legal_action_indices(state)
    assert END_TURN_ACTION in legal
    assert 0 in legal
    assert 1 in legal


def test_simulate_action_applies_damage_and_removes_card() -> None:
    next_state = simulate_action(_state(), 0)
    assert len(next_state.hand) == 1
    assert next_state.player.energy == 0
    assert next_state.enemies[0].hp == 2


def test_simulate_action_end_turn_resets_block_and_advances_turn() -> None:
    state = _state()
    state.player.block = 9
    state.turn = 3

    next_state = simulate_action(state, END_TURN_ACTION)
    assert next_state.player.block == 0
    assert next_state.turn == 4


def test_beam_search_returns_attack_index() -> None:
    state = _state()
    scorer = DummyScorer()
    action_index = beam_search(state, scorer=scorer, beam_width=3, depth=2)
    assert action_index == 0


def test_search_policy_returns_planned_action() -> None:
    state = _state()
    policy = SearchPolicy(scorer=DummyScorer(), beam_width=3, depth=2)
    action = policy.choose_action(state)
    assert action is not None
    assert action.card_name == "Strike"
