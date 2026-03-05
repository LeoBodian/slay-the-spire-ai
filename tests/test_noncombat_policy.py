from sts_ai.models import GameObservation, GamePhase, MapNode, RewardOption
from sts_ai.policy import HeuristicPolicy


def test_choose_map_path_prefers_elite_when_healthy() -> None:
    policy = HeuristicPolicy()
    observation = GameObservation(
        phase=GamePhase.MAP,
        map_nodes=[
            MapNode(x=0, y=2, node_type="monster", connected_to=[]),
            MapNode(x=1, y=3, node_type="elite", connected_to=[]),
        ],
    )

    assert policy.choose_map_path(observation) == 1


def test_choose_card_reward_prefers_non_curse_card() -> None:
    policy = HeuristicPolicy()
    observation = GameObservation(
        phase=GamePhase.REWARD,
        rewards=[
            RewardOption(kind="card", label="Cursed Blade"),
            RewardOption(kind="card", label="Pommel Strike"),
        ],
    )

    assert policy.choose_card_reward(observation) == 1


def test_choose_rest_action_defaults_to_rest_without_combat_state() -> None:
    policy = HeuristicPolicy()
    observation = GameObservation(phase=GamePhase.REST)

    assert policy.choose_rest_action(observation) == "rest"
