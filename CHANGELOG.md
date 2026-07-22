# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `fieldbird-sim init` for creating a self-contained editable starter project.
- Bundled crop-calendar and species defaults in built distributions.
- Installed-wheel smoke testing outside the repository checkout.
- Public configuration guide and package metadata.
- CI workflow for linting and tests on pull requests and pushes.
- GitHub community and governance files (`LICENSE`, issue/PR templates, `SECURITY.md`, Dependabot config).
- Additional smoke tests for config loading and CLI run dispatch.
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
- Ruff is now scoped to maintained package and test code (legacy modules are excluded).
- Weather module defaults were calibrated for mild daily survival impact (near 1.0) to avoid artificial population collapse/explosion.

## [0.1.0] - 2026-02-27

### Added
- Initial package layout under `src/fieldbird_sim`.
- CLI entry point (`fieldbird-sim`).
- YAML-driven run configuration loader.
