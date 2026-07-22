# Release checklist

## Before making the repository public

- Confirm that no private landscape, weather, location, or credential data is tracked.
- Confirm that `main` is the GitHub default branch.
- Require the CI lint/test and package-smoke jobs before merging to `main`.
- Disable force pushes and branch deletion for `main`.
- Review repository description, topics, license detection, and security-advisory settings.

## Local release checks

```bash
git diff --check
ruff check .
python -m pytest --cov=fieldbird_sim --cov-report=term-missing -q
python -m build
```

Install the generated wheel into a fresh environment and run:

```bash
fieldbird-sim --version
fieldbird-sim init /tmp/fieldbird-release-smoke
cd /tmp/fieldbird-release-smoke
fieldbird-sim run --config run.yml
```

## Release metadata

- Move completed entries from `Unreleased` into a dated version section.
- Keep the version in `pyproject.toml` and `CITATION.cff` aligned.
- Commit the release metadata through a pull request.
- Create an annotated `vX.Y.Z` tag only after CI passes on the release commit.
- Create a GitHub release from that tag and attach the wheel and source distribution.
