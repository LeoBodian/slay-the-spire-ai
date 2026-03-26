from __future__ import annotations

from dataclasses import dataclass

from .models import CardState, CardType, CombatState, EnemyState, GameObservation


@dataclass(slots=True)
class PlannedAction:
    card_name: str
    target: str | None
    score: float
    rationale: str


class PolicyBase:
    """Shared policy contract with safe defaults for non-combat decisions."""

    def choose_action(self, state: CombatState) -> PlannedAction | None:  # noqa: ARG002
        raise NotImplementedError()

    def choose_map_path(self, observation: GameObservation) -> int | None:  # noqa: ARG002
        return 0 if observation.map_nodes else 3

    def choose_card_reward(self, observation: GameObservation) -> int | None:  # noqa: ARG002
        return 0 if observation.rewards else None

    def choose_rest_action(self, observation: GameObservation) -> str:  # noqa: ARG002
        return "rest"

    def choose_neow_blessing(self, observation: GameObservation) -> int | None:  # noqa: ARG002
        return 0


class HeuristicPolicy(PolicyBase):
    """Simple baseline policy for early offline experiments."""

    def choose_action(self, state: CombatState) -> PlannedAction | None:
        affordable_cards = [card for card in state.hand if card.cost <= state.player.energy]
        if not affordable_cards:
            return None

        scored_actions = [self._score_card(card, state) for card in affordable_cards]
        scored_actions.sort(key=lambda action: action.score, reverse=True)
        return scored_actions[0]

    def _score_card(self, card: CardState, state: CombatState) -> PlannedAction:
        if card.card_type == CardType.ATTACK:
            target = self._select_attack_target(card, state.alive_enemies)
            score = self._score_attack(card, target)
            rationale = f"attack target={target.name}"
            return PlannedAction(
                card_name=card.name,
                target=target.name,
                score=score,
                rationale=rationale,
            )

        if card.card_type == CardType.SKILL:
            score = self._score_skill(card, state)
            return PlannedAction(
                card_name=card.name,
                target=None,
                score=score,
                rationale="defensive utility",
            )

        score = 1.0 - (0.4 * card.cost)
        return PlannedAction(
            card_name=card.name,
            target=None,
            score=score,
            rationale="low-priority power",
        )

    def _select_attack_target(self, card: CardState, enemies: list[EnemyState]) -> EnemyState:
        lethal_targets = [enemy for enemy in enemies if card.damage >= enemy.effective_hp]
        if lethal_targets:
            return min(lethal_targets, key=lambda enemy: enemy.effective_hp)
        return max(enemies, key=lambda enemy: (enemy.intent_value, enemy.effective_hp))

    def _score_attack(self, card: CardState, target: EnemyState) -> float:
        lethal_bonus = 100.0 if card.damage >= target.effective_hp else 0.0
        efficiency = (card.damage / max(card.cost, 1)) * 4.0
        pressure_bonus = target.intent_value * 1.5
        exhaust_penalty = 2.0 if card.exhausts else 0.0
        return lethal_bonus + efficiency + pressure_bonus - exhaust_penalty

    def _score_skill(self, card: CardState, state: CombatState) -> float:
        incoming_after_block = max(state.incoming_damage - state.player.block, 0)
        covered_damage = min(card.block, incoming_after_block)
        emergency_bonus = 8.0 if incoming_after_block > 0 else 0.0
        hp_pressure_bonus = min(state.player.missing_hp, 20) * 0.1
        danger_ratio = incoming_after_block / max(state.player.hp, 1)
        survival_bonus = danger_ratio * 25.0
        if incoming_after_block >= state.player.hp:
            survival_bonus += 15.0
        efficiency = (covered_damage / max(card.cost, 1)) * 3.5
        exhaust_penalty = 1.0 if card.exhausts else 0.0
        return (
            emergency_bonus
            + hp_pressure_bonus
            + survival_bonus
            + efficiency
            - exhaust_penalty
        )

    def choose_map_path(self, observation: GameObservation) -> int | None:
        """Pick a map node index from currently available nodes."""
        if not observation.map_nodes:
            return 3

        hp_ratio = None
        if observation.combat is not None:
            hp_ratio = observation.combat.player.hp / max(observation.combat.player.max_hp, 1)

        priorities = {
            "rest": 3.0 if hp_ratio is not None and hp_ratio < 0.50 else 1.0,
            "elite": 0.4 if hp_ratio is not None and hp_ratio < 0.50 else 1.8,
            "monster": 1.5,
            "event": 1.2,
            "shop": 1.0,
            "boss": 1.3,
        }

        best_idx = 0
        best_score = float("-inf")
        for idx, node in enumerate(observation.map_nodes):
            base = priorities.get(node.node_type, 0.8)
            depth_bonus = node.y * 0.01
            score = base + depth_bonus
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def choose_card_reward(self, observation: GameObservation) -> int | None:
        """Pick the best reward card index, or None to skip."""
        if not observation.rewards:
            return None

        # Basic deck-agnostic scoring: prefer cards over non-cards for now.
        best_idx = None
        best_score = float("-inf")
        for idx, reward in enumerate(observation.rewards):
            score = 0.0
            if reward.kind == "card":
                score += 2.0
            if "upgrade" in reward.label.casefold():
                score += 1.0
            if "curse" in reward.label.casefold():
                score -= 3.0
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def choose_rest_action(self, observation: GameObservation) -> str:
        """Choose between 'rest' and 'smith'."""
        combat = observation.combat
        if combat is None:
            return "rest"

        hp_ratio = combat.player.hp / max(combat.player.max_hp, 1)
        return "rest" if hp_ratio < 0.60 else "smith"
