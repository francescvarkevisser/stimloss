from __future__ import annotations
from typing import Any, Dict
import os

from .config import Analysis, Dataset
from .registry import get_analysis_runner
from .plotting import plot_box, plot_line  # generic helpers
import matplotlib.ticker as mticker

_PLOTTERS = {
    "boxplot": plot_box,
    "line": plot_line,
}

def run_analysis(analysis: Analysis, datasets_by_id: Dict[str, Dataset], outdir: str) -> Dict[str, Any]:
    runner = get_analysis_runner(analysis.type)
    artifacts = runner(analysis, datasets_by_id, {"outdir": outdir})

    # Figures declared in config
    for fig in analysis.outputs.figures:
        kind = fig["kind"]
        df_key = fig.get("dataframe", "long")
        if kind not in _PLOTTERS:
            raise ValueError(f"Unknown figure kind: {kind}")
        path = fig.get("path")
        if path is not None and not os.path.isabs(path):
            # if relative, resolve under project output dir
            path = os.path.join(outdir, path)
        if path:
            dirpath = os.path.dirname(path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
        df_fig = artifacts["dataframes"][df_key].copy()
        include_labels = fig.get("include_strategies")
        if include_labels:
            labels = include_labels if isinstance(include_labels, (list, tuple, set)) else [include_labels]
            labels = set(str(l) for l in labels)
            if "id" in df_fig.columns:
                df_fig = df_fig[df_fig["id"].astype(str).isin(labels)]
        plot_kwargs = {k: v for k, v in fig.items() if k not in {"kind", "dataframe", "include_strategies", "path"}}
        # If hue is a list of columns, build a composite hue column
        hue_val = plot_kwargs.get("hue")
        if isinstance(hue_val, (list, tuple)):
            cols = list(hue_val)
            missing = [c for c in cols if c not in df_fig.columns]
            if missing:
                raise ValueError(f"Hue columns not found: {missing}")
            df_fig["_composite_hue"] = df_fig[cols].astype(str).agg(" | ".join, axis=1)
            plot_kwargs["hue"] = "_composite_hue"
            # set legend title to describe the composition
            plot_kwargs["legend"] = True
            plot_kwargs["title"] = " | ".join(cols)
        if path:
            plot_kwargs["path"] = path
        fig_obj, ax_obj = _PLOTTERS[kind](df_fig, **plot_kwargs)

    # Tables
    for tbl in analysis.outputs.tables:
        df_key = tbl.get("dataframe", tbl.get("kind"))
        path = tbl["path"]
        if not os.path.isabs(path):
            path = os.path.join(outdir, path)
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        df_out = artifacts["dataframes"][df_key]
        cols = tbl.get("columns")
        if cols:
            missing = [c for c in cols if c not in df_out.columns]
            if missing:
                raise ValueError(f"Requested columns not in dataframe '{df_key}': {missing}")
            df_out = df_out[cols]
        if path.lower().endswith(".parquet"):
            df_out.to_parquet(path, index=False)
        else:
            df_out.to_csv(path, index=False)

    # Prints
    for p in analysis.outputs.print:
        if p == "headline_medians" and "long" in artifacts["dataframes"]:
            s = artifacts["dataframes"]["long"].groupby("Condition")["Efficiency"].median().sort_values(ascending=False)
            print(f"[{analysis.id}] median efficiencies (%):\n{s}\n")

    return artifacts
