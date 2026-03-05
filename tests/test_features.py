from sts_ai.features import ACTION_SPACE, MAX_HAND_CARDS, encode_action, encode_combat_state
from sts_ai.models import CardState, CardType, CombatState, EnemyState, IntentType, PlayerState
from sts_ai.policy import PlannedAction


def _state() -> CombatState:
    return CombatState(
        player=PlayerState(hp=40, max_hp=80, block=3, energy=1),
        enemies=[
            EnemyState(
                name="Louse",
                hp=12,
                block=2,
                intent=IntentType.ATTACK,
                intent_value=6,
            )
        ],
        hand=[
            CardState(name="Strike", cost=1, card_type=CardType.ATTACK, damage=6),
            CardState(name="Defend", cost=1, card_type=CardType.SKILL, block=5),
        ],
        draw_pile=10,
        discard_pile=3,
        turn=2,
    )


def test_encode_combat_state_shape_is_fixed() -> None:
    vec = encode_combat_state(_state())
    assert vec.ndim == 1
    # 9 globals + (5 enemies * 9 features) + (10 cards * 8 features)
    assert vec.shape[0] == 9 + (5 * 9) + (10 * 8)


def test_encode_action_maps_card_name_to_index() -> None:
    state = _state()
    action = PlannedAction(card_name="Defend", target=None, score=1.0, rationale="test")
    assert encode_action(action, state) == 1


def test_encode_action_uses_end_turn_for_none_or_unknown() -> None:
    state = _state()
    assert encode_action(None, state) == MAX_HAND_CARDS

    missing = PlannedAction(card_name="Bash", target=None, score=1.0, rationale="test")
    assert encode_action(missing, state) == MAX_HAND_CARDS


def test_action_space_matches_hand_plus_end_turn() -> None:
    assert ACTION_SPACE == MAX_HAND_CARDS + 1
