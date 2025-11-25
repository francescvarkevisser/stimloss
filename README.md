# stimloss

Stimloss is a framework to analyse the power losses and power efficiency in multichannel electrical stimulation applications, and to evaluate the efficiacy of supply scaling strategies.

The framework is described in this preprint: https://arxiv.org/abs/2501.08025.

## Install (Windows)
```Command Prompt
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
# dev tools (optional)
python -m pip install -e .[dev]
```
## Install (macOS/Linux)
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
# dev tools (optional)
python -m pip install -e .[dev]
```
## Run example
`stimloss run -c .\examples\basic_analysis\demo_config.yaml`
figures and dataframes are saved in ./examples/basic_analysis/output

## Project layout
```
src/stimloss/
  generation.py     # data synthesis helpers
  strategies.py     # loss models + strategy evaluation helpers
  plotting.py       # plotting helpers
  utils.py          # dataset loading helpers
  cli.py            # Typer CLI
configs/
  default.yaml              # template for authoring your own configs
  literature.yaml           # example generation task
examples/
  basic_analysis/           # simple strategies eval example
  sweep_analysis/           # example sweeps (e.g., yield_sweep.yaml)
  recreate_paper_figures/   # recreate figures from the preprint
  compare_measured_results/ # mapping + comparison demos
scripts/
  replot.py                 # customize plots from saved CSVs
```

## Ways to run
- **Direct CLI**: subcommands such as `stimloss generate`, `stimloss run`, `stimloss new-dataset`, `stimloss load-check`.
- **YAML config**: run end-to-end via configs; e.g. `stimloss run -c examples/mapping_measurements.yaml` (mapping-only, outputs `mapped` CSV) or `stimloss generate -c configs/literature.yaml`. Start new configs from `configs/default.yaml`.
- **Python scripts**: execute demos or build custom flows; e.g. `python examples/demo_recreate_paper_figures.py`, `python examples/demo_vheadroom_compare.py`, or import APIs directly (`from stimloss.strategies import calculate_ploss`).

## Workflow
1. **Start from a config**: copy `configs/default.yaml` or an example (e.g. `examples/basic_analysis/demo_config.yaml`, `examples/sweep_analysis/yield_sweep.yaml`) into your own file.
2. **Register datasets**: add `datasets` entries (kind = `combined`, `measured`, `sim`, or `sim_pl`) that point to your Parquet/CSV files. You can edit YAML directly or use `stimloss add-dataset -c <config> --id ... --path ... --kind ...`.
3. **(Optional) Synthesize data**: either run generation tasks declared in your config (`stimloss generate -c <config>`) or build from a literature table (`stimloss generate --from-table data/LiteratureOverview.xlsx --sheet DataSets --n-samples 10000 --output-dir data/bundles`).
4. **Validate the config**: `stimloss validate -c <config>` checks paths, columns, and analysis references.
5. **Run analyses**: `stimloss run -c <config>` executes the analyses in the file (or a single one with `--analysis <id>`), writing figures/tables under `project.output_dir`. Use `strategies` for fixed/stepped/global evaluations or `strategies_sweep` to sweep a parameter.
6. **Inspect results**: figures are saved as declared in YAML; tables can be written to CSV/Parquet. Use `stimloss list-ids --path <file> --column id` to quickly inspect unique ids.

## Strategy model (Fixed, Stepped, Global)
Strategies are config-driven and evaluated by `build_long_from_strategies` (`src/stimloss/strategies.py`). Supported types:
- `fixed`: single rail (`nsteps=1`). Params: `Vheadroom`, optional `Vmax`, or per-source percentile via `channel_yield` + `Vsub=True`.
- `stepped`: multi-rail. Params: `nsteps` , `dist` (`uniform`/`exp`/`invexp`), `alpha`, `Vheadroom`, optional `Vmax`, or `channel_yield` + `Vsub=True` for per-source maxima.
- `global`: uses aggregated `Efficiency_global` / `Ploss_global` from the sampling stage (resample-by-group).

Custom strategies: add a helper/column in `src/stimloss/strategies.py` and reference it via a column or pandas expression in your YAML `strategies` block (type-less entries are treated as expr/column).

## Quick CLI examples

```powershell
# Check a dataset bundle (folder with parquet or timestamped files)
stimloss load-check data\spd_v1

# Generate from LiteratureOverview.xlsx (sheet 'DataSets')
stimloss generate --excel LiteratureOverview.xlsx --N 10000

# Plot figures from a parquet bundle
stimloss plot --bundle data\spd_v1 --out figs
# If using timestamped files:
stimloss plot --timestamp 20250316_1835 --out figs
```

### Data generation paths
- From literature table (CSV/Excel):  
  `stimloss generate -c configs/default.yaml --from-table data\raw\LiteratureOverview.xlsx --sheet DataSets --n-samples 10000 --output-dir data\bundles`
- From configured task:  
  `stimloss generate -c configs/literature.yaml`
- Append one synthetic dataset:  
  `stimloss new-dataset --method mean_sd --target V1 --I-mean 5e-5 --I-sd 1e-5 --Z-mean 5e4 --Z-sd 1e4 --input-combined data\bundles\combined_df.parquet --output-combined data\bundles\combined_df_aug.parquet`
- From meta YAMLs (measured points -> CSVs):  
  `stimloss generate -c examples/mapping_measurements.yaml --task make_subject_tables`

Supported synthesis methods:
- `mean_sd`: truncated normal using mean/sd, clipped to min/max/step.
- `median_iqr`: infer std from IQR (std ~= IQR/1.349), truncated normal with min/max/step.
- `dataset`: KDE sample from provided CSV columns `I`, `Z`.

### Replot saved results
After `stimloss run`, save your dataframes to CSV, then tweak plots without re-running analyses:
```powershell
python scripts\replot.py --csv reports\asic_compare_v1_long.csv --plot box --x Target --y Efficiency --hue Condition --title "Efficiency vs Target" --ylabel "Efficiency [%]" --output reports\efficiency_custom.png
python scripts\replot.py --csv reports\asic_compare_v1_long.csv --plot box --x Target --y Ploss --hue Condition --logy --ylabel "Ploss [uW]" --output reports\ploss_custom.png
```

## Demos
- `examples/basic_analysis`: minimal analysis on combined data.
- `examples/sweep_analysis`: strategies sweep over channel yield.
- `examples/recreate_paper_figures/demo_recreate_paper_figures.py`: recreates paper-style figures directly from combined data.
- `examples/compare_measured_results/`: mapping + comparison scripts/configs.

## Contributing (living dataset + strategies)
We welcome additions to the "living" dataset and new strategy ideas.

- **Add literature datasets**
  - Append a row to `data/LiteratureOverview.xlsx` (sheet `DataSets`): fill in `method` (`mean_sd`, `median_iqr`, or `dataset`), `target`, `study`, `dataset`, and the required parameters for the chosen method. If you use `dataset`, include a small CSV under `data/raw/` with columns `I` (A) and `Z` (Ohm).
  - Regenerate the bundle locally to ensure it builds:  
    `stimloss generate --from-table data/LiteratureOverview.xlsx --sheet DataSets --n-samples 10000 --output-dir data/bundles`
  - Open a PR with your updated Excel row, any new CSV you added, and note the citation + license of the source.

- **Propose new strategies**
  - Add a helper to `src/stimloss/strategies.py`, then reference it via an expression in a config for validation.
  - Confirm it runs end-to-end: `stimloss run -c <config>` should produce figures/tables without warnings.
  - In your PR description, briefly explain the rationale of the strategy.


General PR checklist: describe the source/assumptions, keep files small (or point to external open data if too large), and ensure commands above succeed before opening the PR.