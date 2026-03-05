from __future__ import annotations

from dataclasses import dataclass

from sts_ai.agent import GameAgent
from sts_ai.models import (
    CardState,
    CardType,
    CombatState,
    EnemyState,
    GameObservation,
    GamePhase,
    IntentType,
    PlayerState,
)
from sts_ai.policy import PlannedAction


@dataclass
class FakeCaptureAdapter:
    observations: list[GameObservation]

    def __post_init__(self) -> None:
        self._index = 0

    def capture_and_parse(self) -> GameObservation:
        obs = self.observations[min(self._index, len(self.observations) - 1)]
        self._index += 1
        return obs


@dataclass
class FakeInputAdapter:
    calls: list[tuple[str, int, int]]

    def click_card(self, index: int, hand_size: int) -> None:
        self.calls.append(("card", index, hand_size))

    def click_enemy(self, index: int, enemy_count: int) -> None:
        self.calls.append(("enemy", index, enemy_count))

    def click_end_turn(self) -> None:
        self.calls.append(("end_turn", 0, 0))

    def click_map_node(self, index: int, node_count: int) -> None:
        self.calls.append(("map", index, node_count))

    def click_reward_option(self, index: int, option_count: int) -> None:
        self.calls.append(("reward", index, option_count))

    def click_rest_action(self, action: str) -> None:
        self.calls.append(("rest", 1 if action == "rest" else 2, 0))


@dataclass
class FakePolicy:
    action: PlannedAction | None

    def choose_action(self, state: CombatState) -> PlannedAction | None:  # noqa: ARG002
        return self.action


def _combat_observation() -> GameObservation:
    combat = CombatState(
        player=PlayerState(hp=60, max_hp=80, block=0, energy=1),
        enemies=[
            EnemyState(
                name="Cultist",
                hp=20,
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
    return GameObservation(phase=GamePhase.COMBAT, combat=combat)


def test_game_agent_executes_card_and_target_clicks() -> None:
    observation = _combat_observation()
    capture = FakeCaptureAdapter([observation])
    input_adapter = FakeInputAdapter(calls=[])
    policy = FakePolicy(
        action=PlannedAction(
            card_name="Strike",
            target="Cultist",
            score=10.0,
            rationale="test",
        )
    )

    agent = GameAgent(
        policy=policy,
        capture_adapter=capture,
        input_adapter=input_adapter,
        loop_delay=0.0,
    )
    agent.step()

    assert ("card", 0, 2) in input_adapter.calls
    assert ("enemy", 0, 1) in input_adapter.calls


def test_game_agent_ends_turn_when_policy_returns_none() -> None:
    observation = _combat_observation()
    capture = FakeCaptureAdapter([observation])
    input_adapter = FakeInputAdapter(calls=[])
    policy = FakePolicy(action=None)

    agent = GameAgent(
        policy=policy,
        capture_adapter=capture,
        input_adapter=input_adapter,
        loop_delay=0.0,
    )
    agent.step()

    assert input_adapter.calls == [("end_turn", 0, 0)]


def test_game_agent_skips_actions_outside_combat() -> None:
    capture = FakeCaptureAdapter([GameObservation(phase=GamePhase.MAP, combat=None)])
    input_adapter = FakeInputAdapter(calls=[])
    policy = FakePolicy(action=None)

    agent = GameAgent(
        policy=policy,
        capture_adapter=capture,
        input_adapter=input_adapter,
        loop_delay=0.0,
    )
    agent.step()

    assert input_adapter.calls == []


def test_game_agent_clicks_map_node_when_available() -> None:
    observation = GameObservation(
        phase=GamePhase.MAP,
        map_nodes=[
            {"x": 0, "y": 1, "node_type": "monster", "connected_to": []},
            {"x": 1, "y": 2, "node_type": "elite", "connected_to": []},
        ],
    )
    capture = FakeCaptureAdapter([observation])
    input_adapter = FakeInputAdapter(calls=[])
    policy = FakePolicy(action=None)

    agent = GameAgent(
        policy=policy,
        capture_adapter=capture,
        input_adapter=input_adapter,
        loop_delay=0.0,
    )
    agent.step()

    assert any(call[0] == "map" for call in input_adapter.calls)


def test_game_agent_clicks_reward_option_when_available() -> None:
    observation = GameObservation(
        phase=GamePhase.REWARD,
        rewards=[
            {"kind": "card", "label": "Strike+"},
            {"kind": "gold", "label": "100 gold"},
        ],
    )
    capture = FakeCaptureAdapter([observation])
    input_adapter = FakeInputAdapter(calls=[])
    policy = FakePolicy(action=None)

    agent = GameAgent(
        policy=policy,
        capture_adapter=capture,
        input_adapter=input_adapter,
        loop_delay=0.0,
    )
    agent.step()

    assert any(call[0] == "reward" for call in input_adapter.calls)


def test_game_agent_clicks_rest_action() -> None:
    observation = GameObservation(phase=GamePhase.REST)
    capture = FakeCaptureAdapter([observation])
    input_adapter = FakeInputAdapter(calls=[])
    policy = FakePolicy(action=None)

    agent = GameAgent(
        policy=policy,
        capture_adapter=capture,
        input_adapter=input_adapter,
        loop_delay=0.0,
    )
    agent.step()

    assert any(call[0] == "rest" for call in input_adapter.calls)
