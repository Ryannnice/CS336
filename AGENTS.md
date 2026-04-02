# Repository Guidelines

## Project Structure & Module Organization
This repository is organized by assignment, and each assignment is effectively a standalone Python project. Use the assignment directory as your working root.

- `assignment1-basics/`: core language-model components in `cs336_basics/`, helper scripts in `scripts/`, tests in `tests/`
- `assignment2-systems/`: systems and Flash Attention work in `cs336_systems/`, tests in `tests/`
- `assignment3-scaling/`: scaling-law experiments in `cs336_scaling/`
- `assignment4-data/`: data-quality pipeline in `cs336_data/`, tests in `tests/`
- `assignment5-alignment/`: alignment code in `cs336_alignment/`, experiment scripts in `scripts/`, tests in `tests/`

Generated artifacts such as datasets, checkpoints, and tokenizer outputs should stay local and out of commits.

## Build, Test, and Development Commands
Most assignments use `uv` with per-directory `pyproject.toml` files:

```bash
cd assignment1-basics
uv sync
uv run pytest
uv run python scripts/train_bpe.py
```

Use targeted test runs while iterating, for example `uv run pytest tests/test_tokenizer.py`.

`assignment2-systems` is the exception: follow `assignment2-systems/SETUP.md`, install with `uv pip install -r requirements.txt`, and run scripts with `.venv/bin/python` instead of `uv run`.

Submission helpers exist in several assignments, for example `test_and_make_submission.sh` and `make_submission.sh`.

## Coding Style & Naming Conventions
Follow existing Python conventions in each assignment:

- 4-space indentation
- `snake_case` for modules, functions, variables, and test names
- type hints where practical
- keep lines within the Ruff limit of 120 characters where Ruff is configured

Prefer small, assignment-local changes over cross-assignment refactors. Keep package names consistent with the existing pattern: `cs336_basics`, `cs336_systems`, `cs336_data`, `cs336_alignment`.

## Testing Guidelines
Tests live beside each assignment under `tests/`. Add new tests as `test_<feature>.py` and name functions `test_<behavior>()`. Reuse fixtures in `tests/conftest.py`, and wire assignment implementations through `tests/adapters.py` where required.

Run the narrowest relevant suite before opening a PR, then run the full assignment test suite.

## Commit & Pull Request Guidelines
Current history uses short, imperative subjects such as `Update README.md`. Keep commits focused and scoped to one assignment when possible.

PRs should include a brief summary, affected assignment(s), exact test commands run, and screenshots or metrics when changing training, evaluation, or data-processing behavior. Do not commit secrets, large datasets, model weights, or machine-specific paths.
