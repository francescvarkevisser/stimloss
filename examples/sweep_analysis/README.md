# Parameter Sweep Example

Sweep parameter values via the config-driven workflow and plot sweep comparisons.

## What this example does
- Sweeps the yield and Vheadroom settings by parsing it as a list of values in the analysis definition.
- The associated analysis id's are suffixed with the sweep values (e.g., `stepped_0.5_0.5`) to have unique identifiers.
- Plots several comparison figures.
- Writes summary tables/plots to `examples/sweep_analysis/output`.

## How to run
From the project root:

```bash
stimloss run -c examples/sweep_analysis/yield_sweep.yaml
```

## Adjusting the sweep
- Edit the `channel_yield` values inside `yield_sweep.yaml` (under `analyses -> strategies` conditions).
- Add/remove strategies or tweak sampling knobs (`default_n_samples`, `n_repeats`) in the same YAML.
