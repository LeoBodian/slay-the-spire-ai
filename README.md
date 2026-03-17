# Slay the Spire 2 AI

An end-to-end Python project for building, evaluating, and iterating on a game-playing AI for Slay the Spire 2.

This repository demonstrates practical AI engineering across the full pipeline:

- Frame capture and state parsing scaffolds
- Heuristic, model-based, and beam-search policies
- Input automation and live agent loop
- Dataset collection and reward shaping
- Training and checkpointing workflow
- Benchmarking, regression fixtures, and CI tasks

## Why This Project

This project is designed as a portfolio-quality applied AI system rather than a toy model. It focuses on reproducible workflows and production-minded tooling:

- Strong separation of concerns (`capture`, `parser`, `policy`, `trainer`, `benchmark`)
- CLI-first operations for training and evaluation
- Automated checks (`pytest`, `ruff`, smoke pipeline)
- Extensible architecture for future RL and richer game integration

## Tech Stack

- Python 3.11+
- Typer (CLI)
- Pydantic (typed models)
- NumPy (feature encoding)
- Optional: OpenCV, MSS, Tesseract (vision)
- Optional: PyTorch (policy/value training)

## Repository Structure

```text
src/sts_ai/
	agent.py         # Live control loop (capture -> decide -> act)
	benchmark.py     # Benchmark metrics and result export
	capture.py       # Live screenshot and image loading adapters
	cli.py           # Unified command-line interface
	dataset.py       # Episode/transition collection and serialization
	evaluator.py     # Offline scenario evaluation harness
	features.py      # Fixed-size model input encoding
	input.py         # Input automation (dry-run supported)
	model_policy.py  # Checkpoint-backed inference policy
	network.py       # Policy/value neural network
	parser.py        # Frame -> structured observation extraction
	policy.py        # Heuristic policy + policy base contract
	rewards.py       # Reward shaping functions
	search.py        # Beam search policy over simulator
	simulator.py     # Approximate combat state transition model
	trainer.py       # Behavior cloning/value training + checkpoints
tests/
	fixtures/        # Regression combat scenarios
	test_*.py        # Unit and integration coverage
```

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Run tests and lint:

```powershell
pytest -q
ruff check .
```

## CLI Commands

- `sts-ai evaluate` : run baseline offline combat scenarios
- `sts-ai parse <image>` : parse a screenshot into structured state
- `sts-ai capture` : capture a live frame
- `sts-ai calibrate <image>` : draw parser regions on a screenshot
- `sts-ai play` : run live policy loop (`heuristic|model|search`)
- `sts-ai collect` : collect episodes for training
- `sts-ai train` : train policy/value model from dataset
- `sts-ai benchmark` : run repeated episodes + metrics
- `sts-ai smoke` : lightweight train+benchmark pipeline check

## Example Workflow

1. Collect episodes:

```powershell
sts-ai collect --games 10 --policy heuristic --log-dir data/episodes --dry-run true
```

2. Train checkpoint:

```powershell
sts-ai train --data-dir data/episodes --epochs 50 --checkpoint checkpoints/latest.pt
```

3. Benchmark with exports:

```powershell
sts-ai benchmark --games 20 --policy search --checkpoint checkpoints/latest.pt --output-json data/benchmarks/latest.json --output-csv data/benchmarks/latest.csv --dry-run true
```

## VS Code Tasks

- `Run STS AI tests v2`
- `Run STS AI smoke v2`
- `Run STS AI CI checks v2`

## Current Status

- End-to-end CLI and testing pipeline implemented
- 40+ tests with regression fixtures and smoke checks
- CI-oriented automation scripts in `.vscode/`
- Architecture ready for richer vision parsing and advanced RL training

## License

MIT (see `LICENSE`)

