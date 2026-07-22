# Model assumptions and parameter provenance

## Intended use

`fieldbird-sim` is research software for reproducible experiments with field-bird
population dynamics in agricultural landscapes. It is designed to compare model
scenarios and explore parameter sensitivity. It is not, by itself, a validated
forecasting tool for a particular species, place, or management decision.

The bundled parameter values are model-development and demonstration defaults.
Unless a study-specific source is recorded alongside an experiment, they must not
be presented as generally validated biological estimates.

## Parameter sources

| Parameter group | Maintained file | Default provenance status |
| --- | --- | --- |
| Simulation size, duration, seeds, scenarios, plots, and weather switch | `run.yml` created by `fieldbird-sim init` | Demonstration defaults chosen for a short reproducible run |
| Crop operation timing and probabilities | `crop_calendar.csv` | Synthetic model defaults; replace or justify for a real application |
| Species demography, movement, reproduction, survival, and territory size | `species_Feldlerche.csv`, `species_Rebhuhn.csv` | Model-development defaults; calibration required |
| Crop preference and breeding-window weights | `species_*_pref.csv` | Model-development assumptions; sensitivity analysis recommended |
| Optional weather factors and event shocks | `run.yml` plus user-supplied weather CSV | Mechanistic scenario assumptions; disabled by default |

German species and crop identifiers are retained in configuration files because
they are stable model data keys. The command-line interface, warnings, and plot
labels are English.

## Structural assumptions

- A year contains 365 simulated days; leap days are not represented.
- Synthetic landscapes approximate fields and optional permanent cover rather than
  reproducing a specific real landscape.
- Population processes are stochastic and controlled by explicit seeds.
- Habitat scores summarize crop food, cover, and disturbance effects.
- Movement favors locally suitable fields subject to species and breeding-state
  constraints.
- Reproduction, juvenile development, adult survival, winter survival, density
  dependence, and annual redistribution are simplified model processes.
- Management scenarios modify selected operations or habitat shares; they do not
  establish a causal real-world treatment effect by themselves.
- Weather effects are optional and require study-specific daily input. Event
  thresholds and multiplicative effects require calibration.

## Calibration and uncertainty

For an applied study:

1. record a source or rationale for every changed parameter
2. calibrate against observations that are appropriate for the target population
3. validate against data not used for calibration
4. run multiple stochastic seeds and parameter-sensitivity analyses
5. report uncertainty, structural limitations, software version, and input files
6. retain each run's `run_meta.json`, configuration, parameter CSV files, and outputs

The yearly and daily outputs are model results, not observations. Differences
between scenarios should be interpreted as conditional on the chosen structure,
parameters, seeds, and inputs.

## Adding literature-backed parameters

When parameters are derived from a publication or dataset, keep a study-level
provenance table with the parameter name, value, unit, species/population, source,
extraction or transformation method, and access date. Do not replace the bundled
defaults with location-specific values unless they are suitable as general example
data and their redistribution rights are clear.
