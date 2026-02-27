# Contributing

Thanks for contributing to `fieldbird-sim`.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Optional local hooks:

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

## Local quality checks

Run these before opening a PR:

```bash
ruff check .
python -m pytest -q
```

`ruff` excludes legacy modules for now. Keep new changes in maintained package and tests lint-clean.

## Branching and commits

- Keep commits focused and small.
- Use clear commit messages in imperative style.
- Prefer one topic per PR.

## Pull request checklist

- [ ] Changes are documented in `README.md` if behavior changed.
- [ ] `CHANGELOG.md` (`Unreleased`) updated for user-facing changes.
- [ ] Lint/test status was checked locally.
- [ ] Backwards compatibility impact is described.
