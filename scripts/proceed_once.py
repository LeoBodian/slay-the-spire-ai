from __future__ import annotations

import time

import cv2

from sts_ai.capture import CaptureAdapter
from sts_ai.input import InputAdapter
from sts_ai.parser import parse_frame


def main() -> None:
    cap = CaptureAdapter()
    inp = InputAdapter(dry_run=False, click_delay=0.2)

    before = cap.grab_live()
    before_obs = parse_frame(before.pixels)
    print(f"before_phase {before_obs.phase.value}")

    if before_obs.phase.value in {"proceed", "reward", "event", "shop"}:
        inp.highlight_proceed_button()
        inp.confirm_with_enter()
        time.sleep(0.3)
        inp.confirm_with_enter()
        time.sleep(0.8)

    after = cap.grab_live()
    cv2.imwrite("frame_after_proceed_push.png", after.pixels)
    after_obs = parse_frame(after.pixels)
    print(f"after_phase {after_obs.phase.value}")


if __name__ == "__main__":
    main()
