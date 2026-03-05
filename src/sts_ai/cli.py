from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from .agent import GameAgent
from .benchmark import run_benchmark, save_benchmark_result
from .capture import CaptureAdapter
from .dataset import EpisodeRecorder, episodes_to_arrays, load_episodes, save_episode
from .evaluator import evaluate_policy, sample_scenarios
from .input import InputAdapter
from .models import GameObservation, GamePhase
from .policy import HeuristicPolicy
from .regions import REGIONS, load_regions

app = typer.Typer(no_args_is_help=True)


def _build_policy(
    policy_name: str,
    checkpoint: Path | None,
    beam_width: int = 5,
    search_depth: int = 3,
):
    if policy_name == "heuristic":
        return HeuristicPolicy()

    if policy_name == "model":
        if checkpoint is None:
            raise typer.BadParameter("--checkpoint is required when --policy model")
        try:
            from .model_policy import ModelPolicy
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Model policy requires training dependencies. "
                "Install with: pip install -e .[train]"
            ) from exc
        return ModelPolicy(str(checkpoint))

    if policy_name == "search":
        if checkpoint is None:
            raise typer.BadParameter("--checkpoint is required when --policy search")
        try:
            from .model_policy import ModelPolicy
            from .search import SearchPolicy
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Search policy requires training dependencies. "
                "Install with: pip install -e .[train]"
            ) from exc
        scorer = ModelPolicy(str(checkpoint))
        return SearchPolicy(scorer=scorer, beam_width=beam_width, depth=search_depth)

    raise typer.BadParameter("Supported policies: heuristic, model, search")


@app.callback()
def main() -> None:
    """Slay the Spire AI — policy evaluation and game-state capture."""


@app.command()
def evaluate() -> None:
    """Run the baseline heuristic against bundled sample combat states."""
    results = evaluate_policy()
    payload = [
        {
            "scenario": result.scenario_name,
            "action": None
            if result.chosen_action is None
            else {
                "card": result.chosen_action.card_name,
                "target": result.chosen_action.target,
                "score": round(result.chosen_action.score, 2),
                "rationale": result.chosen_action.rationale,
            },
        }
        for result in results
    ]
    typer.echo(json.dumps(payload, indent=2))


@app.command()
def parse(
    image: Annotated[Path, typer.Argument(help="Path to a game screenshot to parse.")],
    regions: Annotated[
        Path | None,
        typer.Option(help="Optional JSON file with region overrides."),
    ] = None,
) -> None:
    """Parse a saved screenshot into structured game state."""
    adapter = CaptureAdapter()
    active_regions = load_regions(regions) if regions else None
    observation = adapter.capture_and_parse(image, regions=active_regions)
    typer.echo(observation.model_dump_json(indent=2))


@app.command()
def capture(
    output: Annotated[
        Path, typer.Option(help="Where to save the captured frame.")
    ] = Path("frame.png"),
) -> None:
    """Grab a live screenshot and save it to disk."""
    adapter = CaptureAdapter()
    frame = adapter.grab_live()
    saved = adapter.save_frame(frame, output)
    typer.echo(f"Saved {frame.width}x{frame.height} frame to {saved}")


@app.command()
def calibrate(
    image: Annotated[Path, typer.Argument(help="Input screenshot to annotate.")],
    output: Annotated[
        Path,
        typer.Option(help="Output path for region-annotated image."),
    ] = Path("calibration.png"),
    regions: Annotated[
        Path | None,
        typer.Option(help="Optional JSON file with region overrides."),
    ] = None,
) -> None:
    """Draw configured regions over a screenshot for calibration."""
    try:
        import cv2  # noqa: WPS433
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Calibration rendering requires 'opencv-python'. "
            "Install it with: pip install -e .[capture]"
        ) from exc

    frame = cv2.imread(str(image))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image}")

    active_regions = load_regions(regions) if regions else REGIONS
    for region in active_regions.values():
        top_left = (region.x, region.y)
        bottom_right = (region.x + region.w, region.y + region.h)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)
        label_y = max(10, region.y - 8)
        cv2.putText(
            frame,
            region.name,
            (region.x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), frame)
    typer.echo(f"Saved calibration overlay to {output}")


@app.command()
def play(
    policy: Annotated[
        str,
        typer.Option(help="Policy to run: 'heuristic', 'model', or 'search'."),
    ] = "heuristic",
    checkpoint: Annotated[
        Path | None,
        typer.Option(help="Required when policy='model'."),
    ] = None,
    delay: Annotated[
        float,
        typer.Option(help="Delay between actions in seconds."),
    ] = 0.3,
    max_steps: Annotated[
        int,
        typer.Option(help="Maximum control-loop steps to run."),
    ] = 1000,
    beam_width: Annotated[
        int,
        typer.Option(help="Beam width when policy='search'."),
    ] = 5,
    search_depth: Annotated[
        int,
        typer.Option(help="Search depth when policy='search'."),
    ] = 3,
    log_dir: Annotated[
        Path | None,
        typer.Option(help="Optional directory for JSONL step logs."),
    ] = Path("logs"),
    dry_run: Annotated[
        bool,
        typer.Option(help="If true, choose actions but do not click."),
    ] = True,
) -> None:
    """Run the live agent loop against the current game window."""
    active_policy = _build_policy(
        policy,
        checkpoint,
        beam_width=beam_width,
        search_depth=search_depth,
    )

    game_agent = GameAgent(
        policy=active_policy,
        capture_adapter=CaptureAdapter(),
        input_adapter=InputAdapter(click_delay=delay, dry_run=dry_run),
        log_dir=log_dir,
        loop_delay=delay,
    )
    last_observation = game_agent.run(max_steps=max_steps)

    if last_observation is None:
        typer.echo("No observations captured.")
    else:
        typer.echo(
            "Run complete: "
            f"phase={last_observation.phase.value}, floor={last_observation.floor}, "
            f"gold={last_observation.gold}"
        )


@app.command()
def collect(
    games: Annotated[
        int,
        typer.Option(help="Number of episodes to collect."),
    ] = 1,
    policy: Annotated[
        str,
        typer.Option(help="Policy to run: heuristic, model, or search."),
    ] = "heuristic",
    checkpoint: Annotated[
        Path | None,
        typer.Option(help="Checkpoint path for model/search policies."),
    ] = None,
    delay: Annotated[
        float,
        typer.Option(help="Delay between actions in seconds."),
    ] = 0.3,
    max_steps: Annotated[
        int,
        typer.Option(help="Maximum steps per episode."),
    ] = 1000,
    beam_width: Annotated[int, typer.Option(help="Beam width for search policy.")] = 5,
    search_depth: Annotated[int, typer.Option(help="Search depth for search policy.")] = 3,
    log_dir: Annotated[
        Path,
        typer.Option(help="Directory where episodes will be written."),
    ] = Path("data/episodes"),
    dry_run: Annotated[
        bool,
        typer.Option(help="If true, choose actions but do not click."),
    ] = True,
) -> None:
    """Collect one or more episodes into gzip JSONL files."""
    active_policy = _build_policy(
        policy,
        checkpoint,
        beam_width=beam_width,
        search_depth=search_depth,
    )

    for game_index in range(1, games + 1):
        game_agent = GameAgent(
            policy=active_policy,
            capture_adapter=CaptureAdapter(),
            input_adapter=InputAdapter(click_delay=delay, dry_run=dry_run),
            log_dir=log_dir,
            loop_delay=delay,
        )
        episode = game_agent.run_episode(max_steps=max_steps)
        typer.echo(
            f"Episode {game_index}/{games}: id={episode.episode_id}, "
            f"transitions={len(episode.transitions)}, outcome={episode.outcome}, "
            f"final_floor={episode.final_floor}"
        )

    episodes = load_episodes(log_dir)
    arrays = episodes_to_arrays(episodes)
    typer.echo(
        "Collection summary: "
        f"episodes={len(episodes)}, "
        f"transitions={arrays['rewards'].shape[0]}"
    )


@app.command()
def train(
    data_dir: Annotated[
        Path,
        typer.Option(help="Directory containing collected episode .jsonl.gz files."),
    ] = Path("data/episodes"),
    epochs: Annotated[
        int,
        typer.Option(help="Training epochs for each phase."),
    ] = 50,
    lr: Annotated[
        float,
        typer.Option(help="Learning rate."),
    ] = 1e-3,
    checkpoint: Annotated[
        Path,
        typer.Option(help="Output checkpoint path."),
    ] = Path("checkpoints/latest.pt"),
) -> None:
    """Train a baseline policy/value model from collected episodes."""
    try:
        from .features import encode_combat_state
        from .network import PolicyValueNet
        from .trainer import Trainer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Training requires torch dependencies. "
            "Install with: pip install -e .[train]"
        ) from exc

    episodes = load_episodes(data_dir)
    if not episodes:
        raise typer.BadParameter(f"No episodes found in {data_dir}")

    sample_state = next(
        (
            transition.observation.combat
            for episode in episodes
            for transition in episode.transitions
            if transition.observation.combat is not None
        ),
        None,
    )
    if sample_state is None:
        raise typer.BadParameter("No combat transitions found in dataset")

    input_dim = int(encode_combat_state(sample_state).shape[0])
    net = PolicyValueNet(input_dim=input_dim)
    trainer = Trainer(net=net, lr=lr)

    policy_losses = trainer.train_behavior_cloning(episodes, epochs=epochs)
    value_losses = trainer.train_value(episodes, epochs=epochs)
    saved = trainer.save_checkpoint(checkpoint)

    typer.echo(
        "Training complete: "
        f"policy_loss={policy_losses[-1]:.4f}, "
        f"value_loss={value_losses[-1]:.4f}, "
        f"checkpoint={saved}"
    )


@app.command()
def benchmark(
    games: Annotated[int, typer.Option(help="Number of benchmark games.")] = 10,
    policy: Annotated[
        str,
        typer.Option(help="Policy to benchmark: heuristic, model, search."),
    ] = "heuristic",
    checkpoint: Annotated[
        Path | None,
        typer.Option(help="Checkpoint path for model/search policies."),
    ] = None,
    beam_width: Annotated[int, typer.Option(help="Beam width for search policy.")] = 5,
    search_depth: Annotated[int, typer.Option(help="Search depth for search policy.")] = 3,
    delay: Annotated[float, typer.Option(help="Delay between actions in seconds.")] = 0.3,
    max_steps: Annotated[int, typer.Option(help="Maximum steps per game.")] = 1000,
    log_dir: Annotated[
        Path,
        typer.Option(help="Directory where benchmark episodes will be saved."),
    ] = Path("data/benchmarks"),
    output_json: Annotated[
        Path | None,
        typer.Option(help="Optional JSON file path for benchmark metrics."),
    ] = None,
    output_csv: Annotated[
        Path | None,
        typer.Option(help="Optional CSV file path for benchmark metrics."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(help="If true, choose actions but do not click."),
    ] = True,
) -> None:
    """Run repeated game episodes and report benchmark metrics."""
    active_policy = _build_policy(
        policy,
        checkpoint,
        beam_width=beam_width,
        search_depth=search_depth,
    )

    def run_one_episode():
        game_agent = GameAgent(
            policy=active_policy,
            capture_adapter=CaptureAdapter(),
            input_adapter=InputAdapter(click_delay=delay, dry_run=dry_run),
            log_dir=log_dir,
            loop_delay=delay,
        )
        return game_agent.run_episode(max_steps=max_steps)

    result = run_benchmark(episode_runner=run_one_episode, n_games=games)
    saved_paths = save_benchmark_result(result, json_path=output_json, csv_path=output_csv)
    typer.echo(
        "Benchmark: "
        f"games={result.total_games}, "
        f"wins={result.wins}, losses={result.losses}, "
        f"win_rate={result.win_rate:.2%}, "
        f"avg_floor={result.avg_floor:.2f}, "
        f"avg_hp_remaining={result.avg_hp_remaining:.2f}"
    )
    if saved_paths:
        typer.echo("Saved benchmark outputs: " + ", ".join(str(path) for path in saved_paths))


@app.command()
def smoke(
    train_epochs: Annotated[int, typer.Option(help="Epochs for smoke training run.")] = 1,
) -> None:
    """Run a lightweight train+benchmark pipeline smoke check."""
    with tempfile.TemporaryDirectory(prefix="sts_ai_smoke_") as tmp_dir:
        root = Path(tmp_dir)
        episodes_dir = root / "episodes"
        ckpt_path = root / "checkpoints" / "smoke.pt"

        # Build tiny synthetic episode data from bundled scenarios.
        recorder = EpisodeRecorder.start()
        policy = HeuristicPolicy()
        for combat_state in sample_scenarios():
            obs = GameObservation(
                phase=GamePhase.COMBAT,
                combat=combat_state,
                floor=combat_state.turn,
            )
            action = policy.choose_action(combat_state)
            recorder.add_transition(
                observation=obs,
                action=action,
                reward=1.0,
                next_observation=obs,
                done=False,
            )
        episode = recorder.finalize(outcome="win", final_floor=2)
        save_episode(episode, episodes_dir / f"episode_{episode.episode_id}.jsonl.gz")

        # Train if torch is available; otherwise skip gracefully.
        trained = False
        try:
            from .features import encode_combat_state
            from .network import PolicyValueNet
            from .trainer import Trainer

            episodes = load_episodes(episodes_dir)
            sample_state = episodes[0].transitions[0].observation.combat
            if sample_state is not None:
                net = PolicyValueNet(input_dim=int(encode_combat_state(sample_state).shape[0]))
                trainer = Trainer(net=net, lr=1e-3)
                trainer.train_behavior_cloning(episodes, epochs=train_epochs)
                trainer.train_value(episodes, epochs=train_epochs)
                trainer.save_checkpoint(ckpt_path)
                trained = True
        except ModuleNotFoundError:
            trained = False

        bench_result = run_benchmark(episode_runner=lambda: episode, n_games=2)
        typer.echo(
            "Smoke OK: "
            f"trained={trained}, "
            f"games={bench_result.total_games}, "
            f"win_rate={bench_result.win_rate:.2%}, "
            f"avg_floor={bench_result.avg_floor:.2f}"
        )


if __name__ == "__main__":
    app()
