# Architecture

## Goal

This project should stay:

- easy to understand (clear entrypoint + readable flow)
- easy to modify (small, testable modules)
- stable for research/simulation work (reproducible runs and outputs)

The architecture intentionally uses **one orchestration file** and **thematic helper modules**.

## Current Structure

### Entrypoints

- `src/fieldbird_sim/cli.py`: CLI, config parsing, `fieldbird-sim run`.

### Simulation Orchestration

- `src/fieldbird_sim/legacy/kpis.py`:
  - top-level `run(cfg)` workflow
  - scenario mutation functions (`apply_scenario_A/B/C/D_*`)
  - KPI/reporting/plot triggering

### Focused Helper Modules

- `src/fieldbird_sim/legacy/plot_controls.py`:
  - normalize map/heatmap year selections
- `src/fieldbird_sim/legacy/weather_builder.py`:
  - build weather module + config from YAML dataclasses
- `src/fieldbird_sim/legacy/hedge_setup.py`:
  - build hedge geometries from config
  - attach hedge metrics to fields
- `src/fieldbird_sim/legacy/scenario_plan.py`:
  - scenario list creation
  - scenario dispatch (`apply_scenario_to_inputs`)
- `src/fieldbird_sim/provenance.py`:
  - record simulator, Python, platform, and dependency versions in run metadata

### Domain/Core Modules

- `src/fieldbird_sim/legacy/simulate.py`: core daily/yearly population dynamics.
- `src/fieldbird_sim/legacy/species.py`: species CSV loading and parameter mapping.
- `src/fieldbird_sim/legacy/layout.py`: landscape geometry + map plotting utilities.
- `src/fieldbird_sim/weather/weather_data.py`: weather CSV ingestion and date lookup.
- `src/fieldbird_sim/weather/weather_impact.py`: weather states/factors/event shock model.

## Runtime Flow

1. CLI loads config (`cli.py`).
2. `kpis.run(cfg)` prepares landscape inputs (`synthetic` or CSV).
3. Plot controls are normalized (`plot_controls.py`).
4. Scenario plan is built (`scenario_plan.py`).
5. For each scenario:
   - apply scenario mutations
   - build hedges + field hedge metrics (`hedge_setup.py`)
6. For each species:
   - load species parameters
   - build weather module (`weather_builder.py`) if enabled
   - run simulation (`simulate.py`)
7. Save outputs (daily tables, optional maps/heatmaps, KPI tables).

## Why Not Split Everything

Over-splitting makes navigation harder.  
This repo follows a practical rule:

- Keep orchestration in one place (`kpis.py`).
- Extract blocks only when they are:
  - large,
  - reused,
  - independently testable,
  - or frequently changed.

## Refactor Policy

Create a new module only if at least one applies:

- a block is roughly >150 lines and has a single responsibility
- unit tests can target it directly
- the block causes merge conflicts or slows reviews

Do **not** split if it only saves a few lines or creates indirection without clarity.

## Current Recommendation

No mandatory additional split is needed right now for readability.  
Optional next step (only if needed): move `apply_scenario_A/B/C/D_*` from `kpis.py` into a dedicated scenario-ops module.

## Testing Expectations

- Unit tests for helper modules (`plot_controls`, `weather_builder`, `hedge_setup`, `scenario_plan`)
- Integration/smoke via CLI run (`fieldbird-sim run --config ...`)
- CI gate: `ruff` + `pytest`

Historical pre-package implementations remain available through Git history and
release tags, but the current tree contains only maintained runtime code.
