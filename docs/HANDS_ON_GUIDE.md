# Hands-on guide

This guide takes you from a clean computer to a reproducible field-bird simulation
that you can modify and rerun. No Python programming is required. You only need a
terminal and a text editor or spreadsheet application.

## What you will do

1. install `fieldbird-sim` in an isolated Python environment
2. create a self-contained simulation project
3. run the bundled two-species baseline
4. find and interpret the main result files
5. make one controlled parameter change and compare the result

The default starter simulates Skylark (`Feldlerche`) and Grey Partridge (`Rebhuhn`)
for two years on a synthetic 20-field landscape. A first run normally takes less
than a minute on a current desktop computer.

## 1. Prerequisites

- Python 3.10, 3.11, 3.12, or 3.13
- about 1 GB of free disk space for Python dependencies and generated results
- Git, if you install from a repository checkout

Check Python from a terminal:

```bash
python --version
```

On systems where `python` is not found, try `python3`. On Windows, `py` is also a
common launcher. Commands below use `python`; substitute the command that works on
your computer.

## 2. Install in an isolated environment

### macOS or Linux

```bash
git clone https://github.com/s7rk/fieldbird-sim.git
cd fieldbird-sim
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install .
fieldbird-sim --version
```

### Windows PowerShell

```powershell
git clone https://github.com/s7rk/fieldbird-sim.git
Set-Location fieldbird-sim
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install .
fieldbird-sim --version
```

If PowerShell blocks environment activation, run this once in the current terminal
and then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

A successful installation prints `fieldbird-sim 0.2.1` or a newer version. Keep the
environment activated while using the program. Later, reactivate it with
`source .venv/bin/activate` on macOS/Linux or `.\.venv\Scripts\Activate.ps1` on
Windows.

### Install directly from a release package

If you do not want to clone the repository, download the `.whl` file for your
version from the [GitHub Releases](https://github.com/s7rk/fieldbird-sim/releases)
page. Create and activate a virtual environment as shown above, open a terminal in
the download directory, and install the downloaded file:

```bash
python -m pip install fieldbird_sim-0.2.1-py3-none-any.whl
fieldbird-sim --version
```

Replace `0.2.1` with the downloaded version. The wheel contains the starter
configuration and editable parameter files; Git is not required for this route.

## 3. Create your simulation project

Choose a location outside the repository for your work. From any directory, run:

```bash
fieldbird-sim init my-first-simulation
cd my-first-simulation
```

Windows PowerShell uses the same two commands. The new directory contains:

| File | Purpose |
| --- | --- |
| `README.md` | short instructions that stay with the project |
| `run.yml` | landscape, runtime, scenario, seed, plot, and weather settings |
| `crop_calendar.csv` | crop operations, timing, spread, and probability |
| `species_Feldlerche.csv` | Skylark demography and movement parameters |
| `species_Rebhuhn.csv` | Grey Partridge demography and movement parameters |
| `species_*_pref.csv` | crop preferences and breeding-window weights |

These files are ordinary text and CSV files. You may edit YAML in a text editor and
CSV files in a text editor or spreadsheet application. When saving from a
spreadsheet, preserve comma-separated CSV format and the existing column names.

## 4. Run the baseline

Run this command from inside `my-first-simulation`:

```bash
fieldbird-sim run --config run.yml
```

The terminal reports landscape construction, each simulated species, adult counts,
and yearly KPIs. A successful run ends with:

```text
All files written under: outputs/my_simulation
```

If you need to stop a run, press `Ctrl+C`. Existing starter inputs are not changed
by a simulation.

## 5. Find the important results

The default run creates this structure:

```text
outputs/my_simulation/
└── Baseline/
    ├── data/
    │   ├── fields.geojson
    │   ├── management.csv
    │   └── rotation.csv
    ├── Feldlerche/
    │   ├── analysis/kpi_yearly.csv
    │   ├── birdsim_daily_full.csv
    │   └── year_1/
    └── Rebhuhn/
        ├── analysis/kpi_yearly.csv
        ├── birdsim_daily_full.csv
        └── year_1/
```

Start with `analysis/kpi_yearly.csv` for each species. Important columns include:

- `year`: simulated year
- `adults_arrival`: adults present at annual arrival
- `fledglings_total`: fledged juveniles produced during the year
- `lambda_prev`: change in arriving adults relative to the previous year
- `colonization_rate`: share of previously empty fields that became occupied

Use `birdsim_daily_full.csv` for daily and field-level analysis. `run_meta.json`
records the simulator, Python, platform, and dependency versions, while the `data`
directory snapshots the realized landscape, rotation, management, and hedges.

## 6. Run a controlled experiment

Do not change several parameter groups at once. First preserve the baseline by
changing only the output name.

Open `run.yml` and change:

```yaml
run:
  run_name: "baseline"
```

Run the baseline again:

```bash
fieldbird-sim run --config run.yml
```

Now enable one flower-strip scenario and give the experiment a new name:

```yaml
run:
  run_name: "flower_strip_experiment"

scenarios:
  A:
    enabled: false
  B:
    enabled: true
    shares: [0.10]
    fractions: [0.50]
    select: "no_flower_first"
    seed: 2025
  C:
    enabled: false
  D:
    enabled: false
```

Run the same command again. The program writes the baseline and scenario outputs
below `outputs/flower_strip_experiment/`. Compare the species KPI tables while
keeping all values under `seeds` unchanged. This isolates the management change
from random variation.

## 7. Parameters you can change

Good first parameters in `run.yml` are:

- `run.n_fields`: number of agricultural fields in a synthetic landscape
- `run.n_years`: simulation duration
- `run.species`: `Feldlerche`, `Rebhuhn`, or both
- `run.n_init_first`: adults at the beginning of the first year
- `run.run_name`: separate output directory for each experiment
- `plots.do_maps` and `plots.do_heatmaps`: optional visual outputs
- `hedges.p_edge`: probability of hedges on eligible shared edges
- scenario blocks A–D: management timing, flower strips, fallow, and Skylark plots
- `weather_module.enabled`: optional weather-driven effects

Species CSVs contain more sensitive model parameters such as breeding windows,
movement radius, survival, clutch size, fledging rate, density dependence, and
territory size. Crop-preference CSVs control food, cover, disturbance, and clutch
weights by crop. Change these only with a documented biological rationale.

See the [configuration guide](CONFIGURATION.md) for the complete YAML wiring and
examples.

## 8. Reproducible comparisons

For every experiment:

1. give the run a unique `run_name`
2. keep all five seeds fixed when comparing parameter changes
3. change one parameter group at a time
4. keep the entire starter directory with the results
5. record why each parameter was changed
6. compare the same species, years, and KPI columns

To study stochastic variation, copy the configuration and change one seed at a
time. Multiple seeds are needed before interpreting a difference as robust.

## 9. Use your own landscape or weather data

Set `run.use_synthetic: false` and provide the CSV paths documented in the
[configuration guide](CONFIGURATION.md). Required columns are validated at startup.
Private location, landscape, and weather data should stay outside Git and must not
be committed to the public repository.

Weather effects are disabled by default. Enable them only after providing a daily
weather CSV and reviewing the event and factor settings in a full repository
configuration such as `configs/run_default.yml`.

## 10. Troubleshooting

### `fieldbird-sim` is not found

Reactivate the virtual environment and retry `fieldbird-sim --version`. If needed,
run the CLI as `python -m fieldbird_sim.cli --version`.

### A CSV or YAML file cannot be found

Run the command inside the starter directory or pass an absolute config path:

```bash
fieldbird-sim run --config /full/path/to/my-first-simulation/run.yml
```

Paths inside `run.yml` are resolved relative to that configuration file.

### `init` refuses to overwrite files

This is intentional protection. Choose a new directory. Use `--force` only if you
want to replace every generated starter file:

```bash
fieldbird-sim init my-first-simulation --force
```

### The run is slower than expected

Start with two years, 20 fields, disabled maps and heatmaps, and only the baseline.
Runtime grows with years, fields, species, scenarios, and plots.

### Results differ after a parameter change

Confirm that the seeds, input files, package version, and all unrelated parameters
are identical. Keep separate output directories and compare like-for-like KPIs.

## Research-use warning

`fieldbird-sim` is alpha research software. It supports reproducible experiments
and method development, but its output is not a validated population forecast.
Ecological interpretation requires application-specific calibration, sensitivity
analysis, uncertainty assessment, and appropriate domain expertise.
