# Configuration guide

The easiest way to configure a run is to create an editable starter directory:

```bash
fieldbird-sim init my-simulation
cd my-simulation
fieldbird-sim run --config run.yml
```

The starter's `species_dir` and `calendar_path` can be relative to the directory
containing `run.yml`. Bundled defaults are used when no local species directory or
default crop calendar is available.

## Run settings

```yaml
run:
  use_synthetic: true
  n_fields: 20
  n_years: 2
  year_to_show: 1
  day_to_show: 160
  species: ["Feldlerche", "Rebhuhn"]
  species_dir: "."
  n_init_first: 100
  outdir_prefix: "outputs"
  run_name: "my_simulation"
```

- `use_synthetic`: generate a landscape when `true`; read `inputs.*` CSVs when
  `false`.
- `n_fields`: number of agricultural fields in a synthetic landscape. Permanent
  cover may add an additional polygon.
- `n_years`: number of simulated years.
- `species`: any bundled species names; currently `Feldlerche` and `Rebhuhn`.
- `species_dir`: optional directory containing editable `species_*.csv` files.
- `n_init_first`: initial adult count for the first simulated year.
- `outdir_prefix` and `run_name`: output location.

## Seeds

```yaml
seeds:
  fields: 2025
  rotation: 2025
  management: 2025
  hedges: 2025
  sim: 2025
```

Keep these values fixed to reproduce a run. Change one seed at a time when studying
stochastic sensitivity.

## Crop management

```yaml
management:
  use_calendar: true
  calendar_path: "crop_calendar.csv"
  desync_region_sd: 4.0
  desync_field_sd: 3.0
  region_bins: 6
```

`crop_calendar.csv` controls central dates, widths, and probabilities for harvest,
sowing, other disturbance, and insecticide events. A bundled calendar is used by
the repository presets; `fieldbird-sim init` creates an editable copy.

## Species parameters

`species_<name>.csv` contains key/value pairs for movement, breeding windows,
survival, reproduction, density dependence, hedge effects, and territory size.
`species_<name>_pref.csv` contains crop-specific food, cover, disturbance, and
clutch-window weights.

These are research parameters. Record and review changes carefully because they can
materially affect population trajectories.

## Scenarios

The baseline always runs. Scenario blocks A–D can be enabled independently:

- A: shift management timing
- B: add flower-strip shares
- C: convert selected area to fallow
- D: add Skylark plots

The full option set is demonstrated in `configs/run_default.yml` in the source
repository.

## CSV landscape mode

Set `run.use_synthetic: false` and provide:

```yaml
inputs:
  fields_csv: "inputs/fields.csv"
  rotation_csv: "inputs/rotation.csv"
  management_csv: "inputs/management.csv"
  hedges_csv: "inputs/hedges.csv"
```

Required columns are validated at startup. The CSV presets in `configs/` show the
expected wiring. Local landscape data is intentionally not bundled.

## Weather module

```yaml
weather_module:
  enabled: false
  weather_csv: "weather.csv"
  seed: 42
  missing_date_policy: "raise"
```

When enabled, the module models stress, food, habitat, seasonal interactions, and
temporary heatwave, rain, storm, and cold-snap shocks. See `configs/run_default.yml`
for all available coefficients.
