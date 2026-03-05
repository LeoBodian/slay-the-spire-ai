"""PyTorch policy/value model."""

from __future__ import annotations

import torch
from torch import nn

from .features import ACTION_SPACE


class PolicyValueNet(nn.Module):
    """Shared trunk network with policy and value heads."""

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.policy_head = nn.Linear(hidden_dim, ACTION_SPACE)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(state_features)
        log_probs = torch.log_softmax(self.policy_head(hidden), dim=-1)
        value = torch.tanh(self.value_head(hidden)).squeeze(-1)
        return log_probs, value
