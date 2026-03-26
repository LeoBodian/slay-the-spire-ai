from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GamePhase(str, Enum):
    MAIN_MENU = "main_menu"
    NEOW = "neow"
    PROCEED = "proceed"
    MAP = "map"
    COMBAT = "combat"
    REWARD = "reward"
    SHOP = "shop"
    EVENT = "event"
    REST = "rest"
    GAME_OVER = "game_over"
    UNKNOWN = "unknown"


class CardType(str, Enum):
    ATTACK = "attack"
    SKILL = "skill"
    POWER = "power"


class IntentType(str, Enum):
    ATTACK = "attack"
    BLOCK = "block"
    BUFF = "buff"
    DEBUFF = "debuff"
    UNKNOWN = "unknown"


class CardState(BaseModel):
    name: str
    cost: int = Field(ge=0)
    card_type: CardType
    damage: int = Field(default=0, ge=0)
    block: int = Field(default=0, ge=0)
    exhausts: bool = False
    upgraded: bool = False


class EnemyState(BaseModel):
    name: str
    hp: int = Field(gt=0)
    block: int = Field(default=0, ge=0)
    intent: IntentType = IntentType.UNKNOWN
    intent_value: int = Field(default=0, ge=0)

    @property
    def effective_hp(self) -> int:
        return self.hp + self.block


class PlayerState(BaseModel):
    hp: int = Field(gt=0)
    max_hp: int = Field(gt=0)
    block: int = Field(default=0, ge=0)
    energy: int = Field(default=3, ge=0)

    @property
    def missing_hp(self) -> int:
        return self.max_hp - self.hp


class CombatState(BaseModel):
    player: PlayerState
    enemies: list[EnemyState]
    hand: list[CardState]
    draw_pile: int = Field(default=0, ge=0)
    discard_pile: int = Field(default=0, ge=0)
    turn: int = Field(default=1, ge=1)

    @property
    def incoming_damage(self) -> int:
        return sum(
            enemy.intent_value
            for enemy in self.enemies
            if enemy.intent == IntentType.ATTACK
        )

    @property
    def alive_enemies(self) -> list[EnemyState]:
        return [enemy for enemy in self.enemies if enemy.hp > 0]


# ---------------------------------------------------------------------------
# Status effects
# ---------------------------------------------------------------------------


class StatusEffect(BaseModel):
    name: str
    amount: int = 0


# ---------------------------------------------------------------------------
# Non-combat game phases
# ---------------------------------------------------------------------------


class MapNode(BaseModel):
    x: int
    y: int
    node_type: str  # "monster", "elite", "rest", "shop", "event", "boss"
    connected_to: list[int] = Field(default_factory=list)


class RewardOption(BaseModel):
    kind: str  # "card", "gold", "potion", "relic"
    label: str


# ---------------------------------------------------------------------------
# Screen region descriptor (used by capture / parser)
# ---------------------------------------------------------------------------


class ScreenRegion(BaseModel):
    """A named rectangular region of the game window, in pixel coords."""

    name: str
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)


# ---------------------------------------------------------------------------
# Top-level observation produced by the parser
# ---------------------------------------------------------------------------


class GameObservation(BaseModel):
    """Everything the AI knows after parsing one frame."""

    phase: GamePhase = GamePhase.UNKNOWN
    combat: CombatState | None = None
    player_statuses: list[StatusEffect] = Field(default_factory=list)
    floor: int = Field(default=0, ge=0)
    gold: int = Field(default=0, ge=0)
    map_nodes: list[MapNode] = Field(default_factory=list)
    rewards: list[RewardOption] = Field(default_factory=list)
    raw_texts: dict[str, str] = Field(default_factory=dict)
