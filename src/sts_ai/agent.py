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

    def choose_neow_blessing(self, observation: GameObservation) -> int | None:
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

        if not combat.hand:
            # Parser fallback: attempt a safe card+target click when hand OCR is empty.
            self._input.click_card(index=0, hand_size=5)
            self._input.click_enemy(index=0, enemy_count=max(len(combat.enemies), 1))
            return PlannedAction(
                card_name="__fallback_card_0__",
                target=None,
                score=0.0,
                rationale="sparse combat parse fallback",
            )

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

    def _execute_non_combat(self, observation: GameObservation) -> tuple[str | None, int | None]:
        """Execute map/reward/rest interactions when possible."""
        if observation.phase == GamePhase.NEOW:
            choose_neow = getattr(self._policy, "choose_neow_blessing", None)
            option_index = choose_neow(observation) if callable(choose_neow) else 0
            if option_index is not None:
                highlight_neow = getattr(self._input, "highlight_neow_option_with_arrows", None)
                confirm_neow = getattr(self._input, "confirm_with_enter", None)
                if callable(highlight_neow) and callable(confirm_neow):
                    highlight_neow(option_index)
                    confirm_neow()
                    return ("neow", option_index)

                press_neow = getattr(self._input, "press_neow_option_hotkey", None)
                if callable(press_neow):
                    press_neow(option_index)
                    return ("neow", option_index)

                option_count = max(len(observation.rewards), 3)
                click_neow = getattr(self._input, "click_neow_option", None)
                if callable(click_neow):
                    click_neow(option_index, option_count)
                else:
                    self._input.click_reward_option(option_index, option_count)
                return ("neow", option_index)
            return (None, None)

        if observation.phase == GamePhase.MAP:
            choose_map = getattr(self._policy, "choose_map_path", None)
            map_index = choose_map(observation) if callable(choose_map) else 0
            if map_index is not None:
                node_count = len(observation.map_nodes) if observation.map_nodes else 7
                clamped_index = max(0, min(map_index, node_count - 1))
                self._input.click_map_node(clamped_index, node_count)
                return ("map", clamped_index)
            return (None, None)

        if observation.phase == GamePhase.PROCEED:
            highlight = getattr(self._input, "highlight_proceed_button", None)
            confirm = getattr(self._input, "confirm_with_enter", None)
            if callable(highlight):
                highlight()
            if callable(confirm):
                confirm()
            else:
                click_proceed = getattr(self._input, "click_proceed_button", None)
                if callable(click_proceed):
                    click_proceed()
                else:
                    self._input.click_reward_option(index=0, option_count=1)
            return ("proceed", 0)

        if observation.phase == GamePhase.REWARD and observation.rewards:
            choose_reward = getattr(self._policy, "choose_card_reward", None)
            reward_index = choose_reward(observation) if callable(choose_reward) else 0
            if reward_index is not None:
                self._input.click_reward_option(reward_index, len(observation.rewards))
                return ("reward", reward_index)
            return (None, None)

        if observation.phase == GamePhase.REST:
            choose_rest = getattr(self._policy, "choose_rest_action", None)
            action = choose_rest(observation) if callable(choose_rest) else "rest"
            self._input.click_rest_action(action)
            return ("rest", 0 if action.casefold() == "rest" else 1)

        if observation.phase in {GamePhase.EVENT, GamePhase.SHOP}:
            # Conservative default: click first option lane in reward panel area.
            self._input.click_reward_option(index=0, option_count=1)
            return (observation.phase.value, 0)

        return (None, None)

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
        prev_choice_label: str | None = None
        prev_choice_index: int | None = None
        final_observation: GameObservation | None = None

        for _ in range(max_steps):
            current_observation = self._capture.capture_and_parse()
            done = current_observation.phase == GamePhase.GAME_OVER

            if prev_observation is not None:
                reward = compute_reward(current_observation, prev_observation, done)
                recorder.add_transition(
                    observation=prev_observation,
                    action=prev_action,
                    choice_label=prev_choice_label,
                    choice_index=prev_choice_index,
                    reward=reward,
                    next_observation=current_observation,
                    done=done,
                )

            chosen_action: PlannedAction | None = None
            chosen_choice_label: str | None = None
            chosen_choice_index: int | None = None
            if not done and current_observation.phase == GamePhase.COMBAT:
                chosen_action = self._choose_and_execute_combat(current_observation)
            elif not done:
                chosen_choice_label, chosen_choice_index = self._execute_non_combat(
                    current_observation
                )

            self._log_event(current_observation, chosen_action)
            prev_observation = current_observation
            prev_action = chosen_action
            prev_choice_label = chosen_choice_label
            prev_choice_index = chosen_choice_index
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
