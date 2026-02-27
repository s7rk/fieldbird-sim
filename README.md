# fieldbird-sim

Field bird population simulation, currently focused on wrapping and stabilizing a legacy simulation pipeline in a reusable Python package structure.

## Current status

This repository is **work in progress**:

- A package/CLI wrapper exists in `src/fieldbird_sim`.
- The simulation core still lives mostly in legacy modules.
- CI checks (`ruff` + `pytest`) run on every push and pull request.

## Installation (developer)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## Run the simulation

### Via installed CLI

```bash
fieldbird-sim run --config configs/run_default.yml
```

### Via script entry point

```bash
python scripts/run.py
```

## Configuration

Default run configuration is provided in:

- `configs/run_default.yml`

Main options:

- `run.use_synthetic`
- `run.n_fields`
- `run.n_years`
- `run.year_to_show`
- `run.day_to_show`
- `seeds.*`

## Development checks

```bash
ruff check .
python -m pytest -q
```

`ruff` is intentionally scoped to maintained package and test code. Legacy simulation modules are excluded for now and will be migrated incrementally.

## Governance and community files

- `LICENSE` (MIT)
- `CONTRIBUTING.md`
- `SECURITY.md`
- `CHANGELOG.md`
- `.github/ISSUE_TEMPLATE/*`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/dependabot.yml`

## Releases

This project uses Semantic Versioning. User-facing changes should be added to `CHANGELOG.md` under `Unreleased` and released via tagged versions.
