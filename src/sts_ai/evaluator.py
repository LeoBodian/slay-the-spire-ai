from __future__ import annotations

from dataclasses import dataclass

from .models import CardState, CardType, CombatState, EnemyState, IntentType, PlayerState
from .policy import HeuristicPolicy, PlannedAction


@dataclass(slots=True)
class EvaluationResult:
    scenario_name: str
    chosen_action: PlannedAction | None


def sample_scenarios() -> list[CombatState]:
    return [
        CombatState(
            player=PlayerState(hp=51, max_hp=80, block=0, energy=1),
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
                CardState(name="Bash", cost=2, card_type=CardType.ATTACK, damage=8),
            ],
            turn=1,
        ),
        CombatState(
            player=PlayerState(hp=22, max_hp=80, block=0, energy=1),
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
            turn=4,
        ),
    ]


def evaluate_policy(policy: HeuristicPolicy | None = None) -> list[EvaluationResult]:
    active_policy = policy or HeuristicPolicy()
    results: list[EvaluationResult] = []
    for index, state in enumerate(sample_scenarios(), start=1):
        action = active_policy.choose_action(state)
        results.append(EvaluationResult(scenario_name=f"scenario_{index}", chosen_action=action))
    return results
