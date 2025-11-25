"""
Sweep Vheadroom values to compare design headroom choices.
Outputs figures and long CSVs to reports/demo_headroom.
"""
from __future__ import annotations

import copy
from pathlib import Path

from stimloss.analyses import run_analysis
from stimloss.config import load_config

HEADROOMS = [0.10, 0.20, 0.30, 0.40]


def main():
    cfg = load_config("examples/mapping_measurements.yaml")
    datasets = {d.id: d for d in cfg.data["datasets"]}
    base_analysis = next(a for a in cfg.analyses if a.id == "asic_compare_v1")

    outdir = Path("reports/demo_headroom")
    outdir.mkdir(parents=True, exist_ok=True)

    for hr in HEADROOMS:
        analysis = copy.deepcopy(base_analysis)
        analysis.id = f"{base_analysis.id}_hr{int(hr * 100)}"

        analysis.params = copy.deepcopy(base_analysis.params)
        mapping_cfg = copy.deepcopy(analysis.params.get("mapping", {}))
        mapping_cfg.setdefault("model", {}).setdefault("params", {})["Vheadroom"] = hr
        analysis.params["mapping"] = mapping_cfg

        for fig in analysis.outputs.figures:
            fig["path"] = str(outdir / f"{analysis.id}_{Path(fig['path']).name}")
        for tbl in analysis.outputs.tables:
            tbl["path"] = str(outdir / f"{analysis.id}_{Path(tbl['path']).name}")

        artifacts = run_analysis(analysis, datasets, str(outdir))
        artifacts["dataframes"]["long"].to_csv(outdir / f"{analysis.id}_long.csv", index=False)
        print(f"[{analysis.id}] Vheadroom={hr:.2f} -> {analysis.outputs.figures[0]['path']}")


if __name__ == "__main__":
    main()
