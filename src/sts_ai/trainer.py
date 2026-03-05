"""Training utilities for behavior cloning and value regression."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dataset import Episode
from .features import encode_action, encode_combat_state
from .network import PolicyValueNet


class Trainer:
    def __init__(self, net: PolicyValueNet, lr: float = 1e-3, gamma: float = 0.99) -> None:
        self.net = net
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def _build_policy_batch(self, episodes: list[Episode]) -> tuple[torch.Tensor, torch.Tensor]:
        states: list[np.ndarray] = []
        targets: list[int] = []
        for episode in episodes:
            for transition in episode.transitions:
                combat = transition.observation.combat
                if combat is None:
                    continue
                states.append(encode_combat_state(combat))
                targets.append(encode_action(transition.action, combat))

        if not states:
            raise ValueError("No combat transitions available for policy training")

        x = torch.tensor(np.vstack(states), dtype=torch.float32)
        y = torch.tensor(np.asarray(targets), dtype=torch.long)
        return x, y

    def _build_value_batch(self, episodes: list[Episode]) -> tuple[torch.Tensor, torch.Tensor]:
        states: list[np.ndarray] = []
        returns: list[float] = []

        for episode in episodes:
            discounted: list[float] = []
            running = 0.0
            for transition in reversed(episode.transitions):
                running = transition.reward + self.gamma * running * (1.0 - float(transition.done))
                discounted.append(running)
            discounted.reverse()

            idx = 0
            for transition in episode.transitions:
                combat = transition.observation.combat
                if combat is None:
                    idx += 1
                    continue
                states.append(encode_combat_state(combat))
                returns.append(discounted[idx])
                idx += 1

        if not states:
            raise ValueError("No combat transitions available for value training")

        x = torch.tensor(np.vstack(states), dtype=torch.float32)
        y = torch.tensor(np.asarray(returns), dtype=torch.float32)
        return x, y

    def train_behavior_cloning(self, episodes: list[Episode], epochs: int = 50) -> list[float]:
        x, targets = self._build_policy_batch(episodes)
        losses: list[float] = []
        self.net.train()
        criterion = nn.NLLLoss()

        for _ in range(epochs):
            self.optimizer.zero_grad()
            log_probs, _ = self.net(x)
            loss = criterion(log_probs, targets)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.item()))

        return losses

    def train_value(self, episodes: list[Episode], epochs: int = 50) -> list[float]:
        x, targets = self._build_value_batch(episodes)
        losses: list[float] = []
        self.net.train()
        criterion = nn.MSELoss()

        for _ in range(epochs):
            self.optimizer.zero_grad()
            _, values = self.net(x)
            loss = criterion(values, targets)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.item()))

        return losses

    def save_checkpoint(self, path: str | Path) -> Path:
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.net.state_dict(),
            "input_dim": self.net.trunk[0].in_features,
        }
        torch.save(payload, ckpt_path)
        return ckpt_path

    @staticmethod
    def load_checkpoint(path: str | Path) -> PolicyValueNet:
        payload = torch.load(Path(path), map_location="cpu")
        input_dim = int(payload["input_dim"])
        net = PolicyValueNet(input_dim=input_dim)
        net.load_state_dict(payload["state_dict"])
        net.eval()
        return net
