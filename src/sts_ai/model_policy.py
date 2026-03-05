"""Model-backed policy wrapper with legal-action masking."""

from __future__ import annotations

import numpy as np
import torch

from .features import MAX_HAND_CARDS, encode_combat_state
from .models import CombatState
from .policy import PlannedAction, PolicyBase
from .trainer import Trainer


class ModelPolicy(PolicyBase):
    def __init__(self, checkpoint: str) -> None:
        self.net = Trainer.load_checkpoint(checkpoint)

    def action_scores(self, state: CombatState) -> np.ndarray:
        """Return masked action probabilities across fixed action space."""
        features = encode_combat_state(state)
        x = torch.tensor(features[np.newaxis, :], dtype=torch.float32)

        with torch.no_grad():
            log_probs, _ = self.net(x)
        probs = log_probs.exp().squeeze(0).cpu().numpy()

        legal_mask = np.zeros_like(probs, dtype=np.float32)
        for idx, card in enumerate(state.hand[:MAX_HAND_CARDS]):
            if card.cost <= state.player.energy:
                legal_mask[idx] = 1.0
        legal_mask[MAX_HAND_CARDS] = 1.0
        return probs * legal_mask

    def evaluate_state(self, state: CombatState) -> float:
        """Estimate scalar value for a combat state."""
        features = encode_combat_state(state)
        x = torch.tensor(features[np.newaxis, :], dtype=torch.float32)
        with torch.no_grad():
            _, value = self.net(x)
        return float(value.squeeze(0).cpu().item())

    def choose_action(self, state: CombatState) -> PlannedAction | None:
        masked = self.action_scores(state)
        if float(masked.sum()) <= 0:
            return None

        action_index = int(masked.argmax())
        if action_index >= len(state.hand):
            return None

        card = state.hand[action_index]
        target = None
        if state.alive_enemies and card.damage > 0:
            target = max(
                state.alive_enemies,
                key=lambda enemy: (enemy.intent_value, enemy.effective_hp),
            ).name

        return PlannedAction(
            card_name=card.name,
            target=target,
            score=float(masked[action_index]),
            rationale="model_policy",
        )
