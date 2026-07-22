# Contributing

Thanks for contributing to `fieldbird-sim`.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Optional local hooks:

```bash
pre-commit install
pre-commit install --hook-type pre-push
```

## Required checks

Run these before opening a pull request:

```bash
git diff --check
ruff check .
python -m pytest -q
python -m build
```

For packaging changes, also install the wheel in a fresh virtual environment and
run a starter simulation outside the repository checkout.

Ruff currently excludes `src/fieldbird_sim/legacy`. New modules and extracted
legacy components should be lint-clean; exclusions should be narrowed rather than
expanded.

## Branches and commits

- Use a short-lived branch for each change.
- Keep commits focused and use imperative commit messages.
- Do not commit generated outputs, private input data, local paths, or backups.
- Prefer one topic per pull request.

## Pull request checklist

- [ ] Behavior changes are documented.
- [ ] `CHANGELOG.md` (`Unreleased`) is updated for user-facing changes.
- [ ] Tests cover the change or the missing coverage is explained.
- [ ] Ruff, Pytest, and package build pass locally.
- [ ] Backwards-compatibility and scientific-result impacts are described.
