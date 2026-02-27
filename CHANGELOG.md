# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CI workflow for linting and tests on pull requests and pushes.
- GitHub community and governance files (`LICENSE`, issue/PR templates, `SECURITY.md`, Dependabot config).
- Additional smoke tests for config loading and CLI run dispatch.

### Changed
- Ruff is now scoped to maintained package and test code (legacy modules are excluded).

## [0.1.0] - 2026-02-27

### Added
- Initial package layout under `src/fieldbird_sim`.
- CLI entry point (`fieldbird-sim`).
- YAML-driven run configuration loader.
