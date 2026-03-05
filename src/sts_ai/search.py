"""Beam search policy over approximate combat simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .models import CombatState
from .policy import PlannedAction, PolicyBase
from .simulator import END_TURN_ACTION, legal_action_indices, simulate_action


class SearchScorer(Protocol):
    """Scores actions and states for beam-search expansion."""

    def action_scores(self, state: CombatState) -> np.ndarray:
        ...

    def evaluate_state(self, state: CombatState) -> float:
        ...


@dataclass(slots=True)
class _BeamNode:
    state: CombatState
    score: float
    first_action: int


def beam_search(
    state: CombatState,
    scorer: SearchScorer,
    beam_width: int = 5,
    depth: int = 3,
) -> int:
    """Return best first action index from beam search."""
    action_scores = scorer.action_scores(state)
    legal = legal_action_indices(state)
    if not legal:
        return END_TURN_ACTION

    nodes: list[_BeamNode] = []
    for action_idx in legal:
        next_state = simulate_action(state, action_idx)
        prior = float(action_scores[action_idx]) if action_idx < len(action_scores) else 0.0
        value = float(scorer.evaluate_state(next_state))
        nodes.append(
            _BeamNode(
                state=next_state,
                score=0.7 * prior + 0.3 * value,
                first_action=action_idx,
            )
        )

    nodes.sort(key=lambda node: node.score, reverse=True)
    nodes = nodes[:beam_width]

    for _ in range(max(depth - 1, 0)):
        expanded: list[_BeamNode] = []
        for node in nodes:
            next_scores = scorer.action_scores(node.state)
            for action_idx in legal_action_indices(node.state):
                next_state = simulate_action(node.state, action_idx)
                prior = float(next_scores[action_idx]) if action_idx < len(next_scores) else 0.0
                value = float(scorer.evaluate_state(next_state))
                expanded.append(
                    _BeamNode(
                        state=next_state,
                        score=node.score + 0.7 * prior + 0.3 * value,
                        first_action=node.first_action,
                    )
                )

        expanded.sort(key=lambda node: node.score, reverse=True)
        nodes = expanded[:beam_width]

    if not nodes:
        return END_TURN_ACTION
    return nodes[0].first_action


class SearchPolicy(PolicyBase):
    """Search policy that chooses the first action from beam search."""

    def __init__(self, scorer: SearchScorer, beam_width: int = 5, depth: int = 3) -> None:
        self._scorer = scorer
        self._beam_width = beam_width
        self._depth = depth

    def choose_action(self, state: CombatState) -> PlannedAction | None:
        action_idx = beam_search(
            state=state,
            scorer=self._scorer,
            beam_width=self._beam_width,
            depth=self._depth,
        )
        if action_idx == END_TURN_ACTION or action_idx >= len(state.hand):
            return None

        card = state.hand[action_idx]
        target = None
        if state.alive_enemies and card.damage > 0:
            target = max(
                state.alive_enemies,
                key=lambda enemy: (enemy.intent_value, enemy.effective_hp),
            ).name

        return PlannedAction(
            card_name=card.name,
            target=target,
            score=0.0,
            rationale="beam_search",
        )
