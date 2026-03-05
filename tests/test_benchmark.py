from dataclasses import dataclass
from pathlib import Path

from sts_ai.benchmark import (
    benchmark_result_to_dict,
    run_benchmark,
    save_benchmark_result,
    summarize_episodes,
)
from sts_ai.dataset import Episode


@dataclass
class EpisodeFactory:
    episodes: list[Episode]

    def __post_init__(self) -> None:
        self._idx = 0

    def __call__(self) -> Episode:
        episode = self.episodes[self._idx % len(self.episodes)]
        self._idx += 1
        return episode


def _episode(outcome: str, floor: int) -> Episode:
    return Episode(
        episode_id=f"{outcome}-{floor}",
        transitions=[],
        outcome=outcome,
        final_floor=floor,
    )


def test_summarize_episodes_basic_metrics() -> None:
    episodes = [_episode("win", 20), _episode("loss", 8), _episode("win", 15)]
    result = summarize_episodes(episodes)

    assert result.total_games == 3
    assert result.wins == 2
    assert result.losses == 1
    assert result.avg_floor == (20 + 8 + 15) / 3


def test_run_benchmark_uses_runner_n_times() -> None:
    episodes = [_episode("win", 10), _episode("loss", 5)]
    factory = EpisodeFactory(episodes)

    result = run_benchmark(factory, n_games=4)

    assert result.total_games == 4
    assert result.wins == 2
    assert result.losses == 2


def test_benchmark_result_export_json_and_csv(tmp_path: Path) -> None:
    result = summarize_episodes([_episode("win", 12), _episode("loss", 6)])

    payload = benchmark_result_to_dict(result)
    assert payload["total_games"] == 2
    assert payload["wins"] == 1

    saved = save_benchmark_result(
        result,
        json_path=tmp_path / "bench.json",
        csv_path=tmp_path / "bench.csv",
    )

    assert len(saved) == 2
    assert (tmp_path / "bench.json").exists()
    assert (tmp_path / "bench.csv").exists()
