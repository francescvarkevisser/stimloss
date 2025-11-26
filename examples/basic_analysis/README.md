# Basic Analysis Example

Run a minimal Stimloss workflow against the bundled demo dataset to compare supply strategies and generate plots/tables.

## What this example does
- Generates `combined_df_generated.parquet` from `Literature_overview`.
- Loads `data/combined_df_generated.parquet`.
- Runs the `strategies` analysis (`fixed`, `stepped`, `global`) with modest sampling counts to keep runtime short.
- Writes figures and summary tables to `examples/basic_analysis/output`.

## How to run
From the project root:

```bash
stimloss run-and-generate -c examples/basic_analysis/demo_config.yaml
```

If you need a specific Python executable, prefix the command with it (e.g., `python -m stimloss …`).

## Outputs
The config requests:
- Figures: `Efficiency_compare.png`, `Ploss_compare.png`, `Efficiency_fixed_vs_global.png`.
- Tables: `final.parquet`, `mean.parquet`.

All land in `examples/basic_analysis/output` (will be created if it does not exist).

## Adjusting the run
- Speed up: lower `default_n_samples` or `n_repeats` in `demo_config.yaml`.
- Change strategies or grouping: edit the `strategies` list or `group_cols` under `analyses` in the config.

## Input data
The example expects the demo Parquet file at `data/combined_df_20250316_1835.parquet`. If you relocate it, update `data.datasets[0].path` in `demo_config.yaml`.
