import json
from pathlib import Path

from sts_ai.models import CombatState
from sts_ai.policy import HeuristicPolicy


def test_policy_regression_fixtures() -> None:
    policy = HeuristicPolicy()
    fixture_dir = Path(__file__).parent / "fixtures"

    for fixture_path in sorted(fixture_dir.glob("combat_*.json")):
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        state = CombatState.model_validate(payload["state"])
        expected = payload["expected_card"]

        action = policy.choose_action(state)

        assert action is not None, f"No action for fixture {fixture_path.name}"
        assert action.card_name == expected, f"Fixture failed: {fixture_path.name}"
