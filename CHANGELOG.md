# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Runtime provenance in `run_meta.json`, including simulator, Python, platform,
  and dependency versions.
- A guide for the optional analysis scripts and a project Code of Conduct.
- Tests for provenance metadata and the source-tree version fallback.
- Focused regression tests for KPI calculations, active scenario callbacks,
  landscape geometry, plotting, species validation, and diagnostic logging.

### Changed
- GitHub Actions are pinned to immutable commit SHAs.
- Packaged species parameter files are now the single maintained source for
  repository runs and generated starter projects.
- The enforced project coverage floor increased from 60% to 70%.

### Fixed
- Source trees without installed package metadata no longer report the stale
  version `0.1.0`; they report `0+unknown` instead.
- Reproduction/debug logging is now wired explicitly, including simulations
  that run without annual population feedback.
- Species parameters are validated before every simulation run.

### Removed
- The unmaintained single-file simulation snapshot, its compatibility runner,
  an unused legacy module entry point, and duplicate root-level species
  parameter files, plus a superseded scenario-helper module. Historical
  versions remain available through Git history and release tags.

## [0.2.1] - 2026-07-22

### Added
- A complete hands-on guide covering installation, first use, parameter experiments,
  results, reproducibility, and troubleshooting.
- A local README in every generated starter project.
- A Windows installed-package simulation smoke test in GitHub Actions.
- Focused behavioral tests for habitat and management-scenario logic.
- A 60% project-wide coverage floor to prevent regression of the hardened test baseline.

### Fixed
- Scenario field selection now accepts input tables without an optional
  `flower_strip_share` column.
- Habitat helpers now import their CSV and date-window dependencies explicitly.

### Changed
- Removed unreachable legacy scenario code that could no longer be executed.
- Updated GitHub Actions to Node.js 24-based checkout and Python setup releases.

## [0.2.0] - 2026-07-22

### Added
- `fieldbird-sim init` for creating a self-contained editable starter project.
- Bundled crop-calendar and species defaults in built distributions.
- Installed-wheel smoke testing outside the repository checkout.
- Public configuration guide and package metadata.
- CI workflow for linting and tests on pull requests and pushes.
- GitHub community and governance files (`LICENSE`, issue/PR templates, `SECURITY.md`, Dependabot config).
- Additional smoke tests for config loading and CLI run dispatch.
- Deterministic end-to-end regression coverage for the bundled starter project.
- Optional weather impact module with:
  - daily weather CSV loader and date-based lookup
  - latent weather memory states (`stress_state`, `food_state`, `habitat_state`)
  - bounded nonlinear multi-state factors for reproduction and survival
  - Phase 3 interactions and season sensitivity (`drought`, `heat*drought`, `cold*wet`, season weighting)
  - Phase 4 event shocks (`heatwave`, `heavy_rain`, `storm`, `cold_snap`) with configurable trigger thresholds/streaks, durations, probabilities, and multiplicative impact factors
  - feature-flag integration (`weather_module.enabled`) and per-run CSV diagnostics (`weather_impact_log.csv`)
- Weather-module smoke/behavior tests.

### Changed
- Default runtime resources now fall back to packaged data instead of requiring the
  current working directory to be the source checkout.
- CI now tests Python 3.10 through 3.13 and builds an installable wheel and source
  distribution.
- The active legacy simulation package now participates in Ruff with a narrow,
  documented baseline for inherited structural patterns.
- Species KPI tables are written to their respective result directories instead
  of overwriting one shared working-directory file.
- Weather module defaults were calibrated for mild daily survival impact (near 1.0) to avoid artificial population collapse/explosion.

## [0.1.0] - 2026-02-27

### Added
- Initial package layout under `src/fieldbird_sim`.
- CLI entry point (`fieldbird-sim`).
- YAML-driven run configuration loader.
