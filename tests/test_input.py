from sts_ai.input import InputAdapter


def test_to_screen_coords_scales_to_window(monkeypatch) -> None:
    monkeypatch.setattr(
        "sts_ai.input.CaptureAdapter._find_game_window_region",
        staticmethod(lambda: {"left": 100, "top": 50, "width": 960, "height": 540}),
    )

    adapter = InputAdapter(dry_run=True)
    x, y = adapter._to_screen_coords(960, 540)

    assert x == 580
    assert y == 320


def test_to_screen_coords_falls_back_without_window(monkeypatch) -> None:
    monkeypatch.setattr(
        "sts_ai.input.CaptureAdapter._find_game_window_region",
        staticmethod(lambda: None),
    )

    adapter = InputAdapter(dry_run=True)
    x, y = adapter._to_screen_coords(123, 456)

    assert x == 123
    assert y == 456
