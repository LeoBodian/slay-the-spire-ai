"""Slay the Spire AI prototype package."""

from .agent import GameAgent
from .benchmark import (
    BenchmarkResult,
    benchmark_result_to_dict,
    run_benchmark,
    save_benchmark_result,
    summarize_episodes,
)
from .capture import CaptureAdapter, CaptureFrame
from .dataset import Episode, EpisodeRecorder, Transition
from .input import InputAdapter
from .models import (
    CardState,
    CombatState,
    EnemyState,
    GameObservation,
    GamePhase,
    PlayerState,
    ScreenRegion,
    StatusEffect,
)
from .parser import parse_frame
from .policy import HeuristicPolicy, PolicyBase
from .rewards import compute_reward
from .search import SearchPolicy, beam_search
from .simulator import END_TURN_ACTION, legal_action_indices, simulate_action

__all__ = [
    "GameAgent",
    "InputAdapter",
    "BenchmarkResult",
    "benchmark_result_to_dict",
    "run_benchmark",
    "save_benchmark_result",
    "summarize_episodes",
    "CaptureAdapter",
    "CaptureFrame",
    "Episode",
    "EpisodeRecorder",
    "CardState",
    "CombatState",
    "EnemyState",
    "GameObservation",
    "GamePhase",
    "HeuristicPolicy",
    "PolicyBase",
    "PlayerState",
    "ScreenRegion",
    "StatusEffect",
    "SearchPolicy",
    "beam_search",
    "END_TURN_ACTION",
    "legal_action_indices",
    "simulate_action",
    "Transition",
    "compute_reward",
    "parse_frame",
]
