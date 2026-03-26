"""Dataset structures and serialization for trajectory collection."""

from __future__ import annotations

import gzip
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .models import GameObservation
from .policy import PlannedAction


@dataclass(slots=True)
class Transition:
    observation: GameObservation
    action: PlannedAction | None
    choice_label: str | None
    choice_index: int | None
    reward: float
    next_observation: GameObservation | None
    done: bool


@dataclass(slots=True)
class Episode:
    episode_id: str
    transitions: list[Transition]
    outcome: str
    final_floor: int


@dataclass(slots=True)
class EpisodeRecorder:
    """Incrementally build and persist one episode."""

    episode_id: str
    transitions: list[Transition]

    @classmethod
    def start(cls) -> "EpisodeRecorder":
        return cls(episode_id=uuid.uuid4().hex, transitions=[])

    def add_transition(
        self,
        observation: GameObservation,
        action: PlannedAction | None,
        choice_label: str | None,
        choice_index: int | None,
        reward: float,
        next_observation: GameObservation | None,
        done: bool,
    ) -> None:
        self.transitions.append(
            Transition(
                observation=observation,
                action=action,
                choice_label=choice_label,
                choice_index=choice_index,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        )

    def finalize(self, outcome: str, final_floor: int) -> Episode:
        return Episode(
            episode_id=self.episode_id,
            transitions=self.transitions,
            outcome=outcome,
            final_floor=final_floor,
        )


def _transition_to_jsonable(transition: Transition) -> dict:
    return {
        "observation": transition.observation.model_dump(mode="json"),
        "action": None if transition.action is None else asdict(transition.action),
        "choice_label": transition.choice_label,
        "choice_index": transition.choice_index,
        "reward": transition.reward,
        "next_observation": None
        if transition.next_observation is None
        else transition.next_observation.model_dump(mode="json"),
        "done": transition.done,
    }


def save_episode(episode: Episode, path: str | Path) -> Path:
    """Save one episode as gzip-compressed JSONL."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(out_path, "wt", encoding="utf-8") as file:
        meta = {
            "type": "meta",
            "episode_id": episode.episode_id,
            "outcome": episode.outcome,
            "final_floor": episode.final_floor,
        }
        file.write(json.dumps(meta) + "\n")

        for transition in episode.transitions:
            row = {"type": "transition", **_transition_to_jsonable(transition)}
            file.write(json.dumps(row) + "\n")

    return out_path


def _decode_observation(payload: dict | None) -> GameObservation | None:
    if payload is None:
        return None
    return GameObservation.model_validate(payload)


def load_episodes(directory: str | Path) -> list[Episode]:
    """Load all gzip JSONL episodes from *directory*."""
    root = Path(directory)
    episodes: list[Episode] = []

    for file_path in sorted(root.glob("*.jsonl.gz")):
        episode_id = file_path.stem
        outcome = "unknown"
        final_floor = 0
        transitions: list[Transition] = []

        with gzip.open(file_path, "rt", encoding="utf-8") as file:
            for raw_line in file:
                row = json.loads(raw_line)
                if row.get("type") == "meta":
                    episode_id = row.get("episode_id", episode_id)
                    outcome = row.get("outcome", "unknown")
                    final_floor = int(row.get("final_floor", 0))
                    continue

                if row.get("type") != "transition":
                    continue

                action_payload = row.get("action")
                action = None
                if action_payload is not None:
                    action = PlannedAction(**action_payload)

                transitions.append(
                    Transition(
                        observation=GameObservation.model_validate(row["observation"]),
                        action=action,
                        choice_label=row.get("choice_label"),
                        choice_index=row.get("choice_index"),
                        reward=float(row["reward"]),
                        next_observation=_decode_observation(row.get("next_observation")),
                        done=bool(row["done"]),
                    )
                )

        episodes.append(
            Episode(
                episode_id=episode_id,
                transitions=transitions,
                outcome=outcome,
                final_floor=final_floor,
            )
        )

    return episodes


def episodes_to_arrays(episodes: list[Episode]) -> dict[str, np.ndarray]:
    """Flatten episodes into simple arrays for training prototypes."""
    rewards: list[float] = []
    dones: list[int] = []
    floors: list[int] = []
    actions: list[str] = []
    choice_labels: list[str] = []
    choice_indices: list[int] = []

    for episode in episodes:
        for transition in episode.transitions:
            rewards.append(transition.reward)
            dones.append(1 if transition.done else 0)
            floors.append(transition.observation.floor)
            actions.append("" if transition.action is None else transition.action.card_name)
            choice_labels.append(
                "" if transition.choice_label is None else transition.choice_label
            )
            choice_indices.append(
                -1 if transition.choice_index is None else transition.choice_index
            )

    return {
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=np.int8),
        "floors": np.array(floors, dtype=np.int16),
        "actions": np.array(actions, dtype=object),
        "choice_labels": np.array(choice_labels, dtype=object),
        "choice_indices": np.array(choice_indices, dtype=np.int16),
    }
