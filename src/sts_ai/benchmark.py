"""Benchmark utilities for comparing policy performance over episodes."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .dataset import Episode


@dataclass(slots=True)
class BenchmarkResult:
    wins: int
    losses: int
    avg_floor: float
    avg_hp_remaining: float
    total_games: int

    @property
    def win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.wins / self.total_games


def _episode_hp_remaining(episode: Episode) -> float:
    if not episode.transitions:
        return 0.0

    for transition in reversed(episode.transitions):
        if transition.next_observation and transition.next_observation.combat:
            return float(transition.next_observation.combat.player.hp)
        if transition.observation.combat:
            return float(transition.observation.combat.player.hp)
    return 0.0


def summarize_episodes(episodes: list[Episode]) -> BenchmarkResult:
    total = len(episodes)
    if total == 0:
        return BenchmarkResult(
            wins=0,
            losses=0,
            avg_floor=0.0,
            avg_hp_remaining=0.0,
            total_games=0,
        )

    wins = sum(1 for ep in episodes if ep.outcome == "win")
    losses = total - wins
    avg_floor = sum(ep.final_floor for ep in episodes) / total
    avg_hp = sum(_episode_hp_remaining(ep) for ep in episodes) / total

    return BenchmarkResult(
        wins=wins,
        losses=losses,
        avg_floor=avg_floor,
        avg_hp_remaining=avg_hp,
        total_games=total,
    )


def run_benchmark(episode_runner: Callable[[], Episode], n_games: int) -> BenchmarkResult:
    """Run *episode_runner* n times and summarize outcomes."""
    episodes = [episode_runner() for _ in range(n_games)]
    return summarize_episodes(episodes)


def benchmark_result_to_dict(result: BenchmarkResult) -> dict[str, float | int]:
    """Convert benchmark results to a JSON-serializable dictionary."""
    return {
        "total_games": result.total_games,
        "wins": result.wins,
        "losses": result.losses,
        "win_rate": result.win_rate,
        "avg_floor": result.avg_floor,
        "avg_hp_remaining": result.avg_hp_remaining,
    }


def save_benchmark_result(
    result: BenchmarkResult,
    json_path: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> list[Path]:
    """Write benchmark output to optional JSON/CSV files and return saved paths."""
    saved: list[Path] = []
    row = benchmark_result_to_dict(result)

    if json_path is not None:
        out_json = Path(json_path)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(row, indent=2), encoding="utf-8")
        saved.append(out_json)

    if csv_path is not None:
        out_csv = Path(csv_path)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        saved.append(out_csv)

    return saved
