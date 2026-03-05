"""Online game loop that connects capture, policy, and input automation."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from .capture import CaptureAdapter
from .dataset import Episode, EpisodeRecorder, save_episode
from .input import InputAdapter
from .models import CombatState, GameObservation, GamePhase
from .policy import PlannedAction
from .rewards import compute_reward


class PolicyLike(Protocol):
    """Policy protocol used by the online game loop."""

    def choose_action(self, state: CombatState) -> PlannedAction | None:
        ...

    def choose_map_path(self, observation: GameObservation) -> int | None:
        ...

    def choose_card_reward(self, observation: GameObservation) -> int | None:
        ...

    def choose_rest_action(self, observation: GameObservation) -> str:
        ...


class GameAgent:
    """Run repeated capture -> decide -> act cycles."""

    def __init__(
        self,
        policy: PolicyLike,
        capture_adapter: CaptureAdapter,
        input_adapter: InputAdapter,
        log_dir: str | Path | None = None,
        loop_delay: float = 0.3,
    ) -> None:
        self._policy = policy
        self._capture = capture_adapter
        self._input = input_adapter
        self._loop_delay = loop_delay
        self._log_dir = Path(log_dir) if log_dir else None
        if self._log_dir:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_dir / "agent_trace.jsonl"
        else:
            self._log_file = None

    def _log_event(self, observation: GameObservation, action: PlannedAction | None) -> None:
        if not self._log_file:
            return
        event = {
            "timestamp": time.time(),
            "phase": observation.phase.value,
            "observation": observation.model_dump(mode="json"),
            "action": None if action is None else asdict(action),
        }
        with self._log_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event) + "\n")

    def _choose_and_execute_combat(self, observation: GameObservation) -> PlannedAction | None:
        combat = observation.combat
        if combat is None:
            return None

        action = self._policy.choose_action(combat)
        if action is None:
            self._input.click_end_turn()
            return None

        card_index = next(
            (
                index
                for index, card in enumerate(combat.hand)
                if card.name.casefold() == action.card_name.casefold()
            ),
            None,
        )
        if card_index is not None:
            self._input.click_card(card_index, len(combat.hand))

        if action.target:
            target_index = next(
                (
                    index
                    for index, enemy in enumerate(combat.enemies)
                    if enemy.name.casefold() == action.target.casefold()
                ),
                None,
            )
            if target_index is not None:
                self._input.click_enemy(target_index, len(combat.enemies))

        return action

    def _execute_non_combat(self, observation: GameObservation) -> None:
        """Execute map/reward/rest interactions when possible."""
        if observation.phase == GamePhase.MAP and observation.map_nodes:
            choose_map = getattr(self._policy, "choose_map_path", None)
            map_index = choose_map(observation) if callable(choose_map) else 0
            if map_index is not None:
                self._input.click_map_node(map_index, len(observation.map_nodes))
            return

        if observation.phase == GamePhase.REWARD and observation.rewards:
            choose_reward = getattr(self._policy, "choose_card_reward", None)
            reward_index = choose_reward(observation) if callable(choose_reward) else 0
            if reward_index is not None:
                self._input.click_reward_option(reward_index, len(observation.rewards))
            return

        if observation.phase == GamePhase.REST:
            choose_rest = getattr(self._policy, "choose_rest_action", None)
            action = choose_rest(observation) if callable(choose_rest) else "rest"
            self._input.click_rest_action(action)
            return

        if observation.phase in {GamePhase.EVENT, GamePhase.SHOP}:
            # Conservative default: click first option lane in reward panel area.
            self._input.click_reward_option(index=0, option_count=1)

    def step(self) -> GameObservation:
        """Run one loop step and return the parsed observation."""
        observation = self._capture.capture_and_parse()

        chosen_action: PlannedAction | None = None
        if observation.phase == GamePhase.COMBAT:
            chosen_action = self._choose_and_execute_combat(observation)
        else:
            self._execute_non_combat(observation)

        self._log_event(observation, chosen_action)
        time.sleep(self._loop_delay)
        return observation

    def run(self, max_steps: int = 1000) -> GameObservation | None:
        """Run the agent for *max_steps* and return the last observation."""
        last_observation: GameObservation | None = None
        for _ in range(max_steps):
            last_observation = self.step()
            if last_observation.phase == GamePhase.GAME_OVER:
                break
        return last_observation

    def run_episode(self, max_steps: int = 1000) -> Episode:
        """Run one episode and return structured transition data."""
        recorder = EpisodeRecorder.start()
        prev_observation: GameObservation | None = None
        prev_action: PlannedAction | None = None
        final_observation: GameObservation | None = None

        for _ in range(max_steps):
            current_observation = self._capture.capture_and_parse()
            done = current_observation.phase == GamePhase.GAME_OVER

            if prev_observation is not None:
                reward = compute_reward(current_observation, prev_observation, done)
                recorder.add_transition(
                    observation=prev_observation,
                    action=prev_action,
                    reward=reward,
                    next_observation=current_observation,
                    done=done,
                )

            chosen_action: PlannedAction | None = None
            if not done and current_observation.phase == GamePhase.COMBAT:
                chosen_action = self._choose_and_execute_combat(current_observation)
            elif not done:
                self._execute_non_combat(current_observation)

            self._log_event(current_observation, chosen_action)
            prev_observation = current_observation
            prev_action = chosen_action
            final_observation = current_observation

            if done:
                break
            time.sleep(self._loop_delay)

        if final_observation is None:
            final_observation = GameObservation(phase=GamePhase.UNKNOWN)

        outcome = "loss" if final_observation.phase == GamePhase.GAME_OVER else "incomplete"
        episode = recorder.finalize(outcome=outcome, final_floor=final_observation.floor)

        if self._log_dir:
            save_episode(
                episode,
                self._log_dir / f"episode_{episode.episode_id}.jsonl.gz",
            )
        return episode
