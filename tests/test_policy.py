from sts_ai.evaluator import evaluate_policy
from sts_ai.models import CardState, CardType, CombatState, EnemyState, IntentType, PlayerState
from sts_ai.policy import HeuristicPolicy


def test_policy_prefers_lethal_attack_when_available() -> None:
    state = CombatState(
        player=PlayerState(hp=50, max_hp=80, block=0, energy=1),
        enemies=[
            EnemyState(
                name="Cultist",
                hp=6,
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

    action = HeuristicPolicy().choose_action(state)

    assert action is not None
    assert action.card_name == "Strike"
    assert action.target == "Cultist"


def test_policy_blocks_when_under_high_pressure() -> None:
    state = CombatState(
        player=PlayerState(hp=18, max_hp=80, block=0, energy=1),
        enemies=[
            EnemyState(
                name="Jaw Worm",
                hp=40,
                block=0,
                intent=IntentType.ATTACK,
                intent_value=11,
            )
        ],
        hand=[
            CardState(name="Strike", cost=1, card_type=CardType.ATTACK, damage=6),
            CardState(name="Defend", cost=1, card_type=CardType.SKILL, block=5),
        ],
    )

    action = HeuristicPolicy().choose_action(state)

    assert action is not None
    assert action.card_name == "Defend"


def test_evaluator_returns_named_scenarios() -> None:
    results = evaluate_policy()

    assert [result.scenario_name for result in results] == ["scenario_1", "scenario_2"]
    assert all(result.chosen_action is not None for result in results)
