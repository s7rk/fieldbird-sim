# fieldbird-sim

`fieldbird-sim` is a configurable field-bird population simulator for synthetic or
CSV-defined agricultural landscapes. The current model includes Skylark
(`Feldlerche`) and Grey Partridge (`Rebhuhn`), configurable management scenarios,
hedges, and an optional weather-impact module.

> **Project status:** research software, alpha. The simulator is suitable for
> reproducible experiments and method development. Model output should not be
> interpreted as a validated population forecast without an application-specific
> calibration and uncertainty assessment.

## Quick start

Python 3.10 or newer is required.

```bash
git clone https://github.com/s7rk/fieldbird-sim.git
cd fieldbird-sim

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install .
```

Create a self-contained, editable simulation directory:

```bash
fieldbird-sim init my-simulation
cd my-simulation
fieldbird-sim run --config run.yml
```

Results are written to `outputs/my_simulation/`. The generated directory contains:

- `run.yml` — landscape, runtime, scenario, seed, plotting, and weather settings
- `species_Feldlerche.csv` and `species_Rebhuhn.csv` — demographic parameters
- `species_*_pref.csv` — crop preferences and breeding-window weights
- `crop_calendar.csv` — crop management timing and probabilities

Edit these files with a text editor or spreadsheet application, then run the same
command again. Existing starter files are never overwritten unless `--force` is
passed to `fieldbird-sim init`.

## Run a repository preset

From a source checkout, a short baseline run is available as:

```bash
fieldbird-sim run --config configs/run_20fields_quick.yml
```

Additional presets cover longer runs, two-species runs, and CSV-defined landscapes.
See [Configuration](docs/CONFIGURATION.md) for the parameter reference and examples.

## What can be configured?

- number of fields and simulation years
- species and initial population
- deterministic random seeds
- synthetic versus CSV-defined landscapes
- crop management calendar and field operations
- hedges and permanent-cover habitat
- management scenarios A–D
- maps, heatmaps, and output locations
- optional daily weather effects and event shocks
- species demography, movement, reproduction, survival, and crop preferences

## Reproducibility

All stochastic components have explicit seeds in the YAML configuration. Keep the
configuration and the generated parameter CSV files together when sharing a run.
The output directory contains data snapshots and diagnostics intended to make a run
auditable.

## Optional weather input

Set `weather_module.enabled: true` and provide `weather_module.weather_csv` in the
YAML file. Weather input is deliberately not bundled because it is location- and
study-specific. Missing dates, interaction terms, seasonal sensitivity, and event
shocks are configurable.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

ruff check .
python -m pytest -q
python -m build
```

The maintained package, active simulation core, tests, and scripts are linted.
The active legacy modules have a narrow baseline for existing star-import and
compact-statement patterns; all other Ruff checks remain enforced. See
[Architecture](ARCHITECTURE.md) and [Contributing](CONTRIBUTING.md).

## Repository data policy

Small parameter tables and example configurations required for a default simulation
are versioned. Local CSV landscape inputs, weather datasets, generated outputs,
plots, model artifacts, and backups are not intended for version control.

## License and security

The project is released under the [MIT License](LICENSE). Please report security
issues through a private GitHub security advisory as described in
[SECURITY.md](SECURITY.md).

For research use, citation metadata is provided in [CITATION.cff](CITATION.cff).
Maintainers can follow the [release checklist](docs/RELEASING.md) before publishing
a version.
