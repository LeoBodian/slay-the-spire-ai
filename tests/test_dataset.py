from pathlib import Path

from sts_ai.dataset import EpisodeRecorder, episodes_to_arrays, load_episodes, save_episode
from sts_ai.models import GameObservation, GamePhase
from sts_ai.policy import PlannedAction


def _obs(phase: GamePhase, floor: int) -> GameObservation:
    return GameObservation(phase=phase, floor=floor)


def test_episode_roundtrip_and_arrays(tmp_path: Path) -> None:
    recorder = EpisodeRecorder.start()
    obs_1 = _obs(GamePhase.COMBAT, floor=5)
    obs_2 = _obs(GamePhase.REWARD, floor=5)

    recorder.add_transition(
        observation=obs_1,
        action=PlannedAction(card_name="Strike", target=None, score=1.0, rationale="test"),
        reward=2.5,
        next_observation=obs_2,
        done=False,
    )

    episode = recorder.finalize(outcome="incomplete", final_floor=5)
    out_file = tmp_path / f"episode_{episode.episode_id}.jsonl.gz"
    save_episode(episode, out_file)

    loaded = load_episodes(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].episode_id == episode.episode_id
    assert len(loaded[0].transitions) == 1

    arrays = episodes_to_arrays(loaded)
    assert arrays["rewards"].shape[0] == 1
    assert arrays["dones"].shape[0] == 1
    assert arrays["actions"][0] == "Strike"
