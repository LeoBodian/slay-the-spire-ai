# Contributing

Thanks for your interest in contributing.

## Development Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## Quality Gate

Run these before opening a PR:

```powershell
pytest -q
ruff check .
```

Or use the VS Code task `Run STS AI CI checks v2`.

## Pull Request Guidelines

- Keep changes focused and atomic.
- Add or update tests for behavior changes.
- Preserve existing module boundaries when possible.
- Update README usage examples if CLI behavior changes.

## Commit Message Style

Use short imperative commit messages, for example:

- `Add benchmark CSV export`
- `Fix parser phase detection fallback`
- `Refactor policy interface defaults`
