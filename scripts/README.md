# Optional analysis scripts

The public `fieldbird-sim` interface is the `fieldbird-sim` command. Files in this
directory are optional, source-checkout utilities for plotting or analysing CSV
outputs; they are not installed as additional commands and are not required to run
the simulator.

Run any utility with `--help` before using it:

```bash
python scripts/plot_total_abundance_by_species.py --help
```

## Simulation-output utilities

- `plot_field_distribution_heatmaps.py`: field-level distribution heatmaps.
- `plot_landscape_crop_legend_en.py`: landscape map with an English crop legend.
- `plot_total_abundance_by_species.py`: total abundance trajectories by species.

## Model-comparison research utilities

The remaining scripts consume user-supplied prediction or experiment CSV files.
They support the project's associated MLP/CNN/GNN comparison workflow but are not
part of the simulator API:

- `build_comparable_unified_predictions.py`
- `plot_daily_model_comparison.py`
- `plot_prediction_csvs.py`
- `plot_run10_by_field.py`
- `plot_new_presi_run10.py`
- `plot_gnn_failure_diagnostics.py`
- `analyze_stack_runs.py`
- `plot_stack_explainer_panels.py`
- `plot_stack_singlecount_diagnostics.py`
- `audit_model_comparison_run10.py`
- `plot_audit_presentation_pack.py`

These tools do not download data. Keep private landscapes, observations,
predictions, and generated figures outside version control. Their input files and
output directories must be supplied explicitly; inspect each tool's `--help` text
for its expected columns and arguments.
