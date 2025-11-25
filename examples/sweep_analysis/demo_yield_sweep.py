"""
Yield sweep demo using the config-driven mapping + strategies workflow.
- Runs strategies defined in examples/yield_sweep.yaml
- Reads the summary output and plots efficiency/loss vs. yield.
"""
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

from stimloss.config import load_config
from stimloss.analyses import run_analysis
from stimloss.plotting import plot_line


def main():
    cfg = load_config("examples/yield_sweep.yaml")
    outdir = Path(cfg.project["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = {d.id: d for d in cfg.data["datasets"]}
    analysis = next(a for a in cfg.analyses if a.id == "yield_sweep_v1")
    artifacts = run_analysis(analysis, datasets, str(outdir))

    # Build simple curves: parse yield from Condition label (e.g., "Yield 0.85")
    long_df = artifacts["dataframes"]["long"]
    def _extract_yield(label: str) -> float:
        m = re.search(r"([0-9]+\\.[0-9]+)", str(label))
        return float(m.group(1)) if m else None

    curves = []
    for _, row in long_df.iterrows():
        yv = _extract_yield(row["Condition"])
        if yv is None:
            continue
        curves.append(
            {
                "channel_yield": yv,
                "Efficiency_pct": row["Efficiency"],
                "Ploss_uW": row["Ploss"],
                "Condition": row["Condition"],
            }
        )
    curve_df = pd.DataFrame(curves)
    if not curve_df.empty:
        eff_curve = curve_df.groupby(["channel_yield", "Condition"], as_index=False)["Efficiency_pct"].mean()
        loss_curve = curve_df.groupby(["channel_yield", "Condition"], as_index=False)["Ploss_uW"].mean()
        plot_line(eff_curve, x="channel_yield", y="Efficiency_pct", hue="Condition", path=str(outdir / "efficiency_vs_yield.png"))
        plot_line(loss_curve, x="channel_yield", y="Ploss_uW", hue="Condition", path=str(outdir / "ploss_vs_yield.png"))

    print(f"Wrote yield sweep outputs to {outdir}")


if __name__ == "__main__":
    main()
