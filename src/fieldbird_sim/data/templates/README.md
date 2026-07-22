# Your fieldbird-sim project

This directory is self-contained: keep `run.yml` and the CSV parameter files
together when moving or sharing a simulation.

## First run

Activate the Python environment in which `fieldbird-sim` is installed, open a
terminal in this directory, and run:

```bash
fieldbird-sim run --config run.yml
```

Results appear below `outputs/my_simulation/`. The most useful summaries are:

- `Baseline/Feldlerche/analysis/kpi_yearly.csv`
- `Baseline/Rebhuhn/analysis/kpi_yearly.csv`
- `Baseline/<species>/birdsim_daily_full.csv`

## Change a run safely

1. Keep an untouched copy of this directory as the baseline.
2. Change `run.run_name` in `run.yml` so a new run has its own output directory.
3. Change one parameter group at a time.
4. Keep the values under `seeds` fixed for a controlled comparison.
5. Run the same command again and compare the species KPI tables.

The files have these roles:

- `run.yml`: runtime, landscape, scenarios, seeds, plots, and weather
- `species_<name>.csv`: demographic and movement parameters
- `species_<name>_pref.csv`: crop preferences and breeding-window weights
- `crop_calendar.csv`: crop-management dates and probabilities

For the full walkthrough and troubleshooting guide, see:

https://github.com/s7rk/fieldbird-sim/blob/main/docs/HANDS_ON_GUIDE.md

`fieldbird-sim` is research software. Treat outputs as simulation results, not as
validated population forecasts without suitable calibration and uncertainty analysis.
