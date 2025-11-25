from __future__ import annotations
from typing import Any, Dict
import pandas as pd
import time
import typer
import os
import numpy as np

from ..io_utils import read_table
from ..mapping import MappingPipeline
from ..registry import register_analysis


@register_analysis("mapping")
def run(analysis, datasets_by_id, project_cfg) -> Dict[str, Any]:
    """
    Legacy mapping analysis runner (kept for backward compatibility when using analyses.type = 'mapping').
    Prefer using top-level mapping tasks + the 'map' CLI.
    """
    def _get_df(ds_id):
        if ds_id is None:
            return None
        ds = datasets_by_id[ds_id]
        return read_table(ds.path, ds.format)

    combined_id = analysis.inputs.get("combined") or analysis.inputs.get("base")
    if combined_id is None:
        raise ValueError("mapping expects a 'combined' (or 'base') dataset id in analysis.inputs.")
    combined = _get_df(combined_id)

    tgt = analysis.params.get("target")
    if tgt is not None and isinstance(combined, pd.DataFrame):
        if "Target" in combined.columns:
            combined = combined.loc[combined["Target"] == tgt].copy()
        elif "target" in combined.columns:
            combined = combined.loc[combined["target"] == tgt].copy()

    pulse_width = analysis.params.get("pulse_width", 40e-6)
    mapping_cfg = analysis.params.get("mapping", {})
    ext = {}
    for j in mapping_cfg.get("joins", []) or []:
        dsid = j["id"]
        ds = datasets_by_id[dsid]  # must exist in data.datasets
        ext[dsid] = read_table(ds.path, ds.format)

    debug = analysis.params.get("debug", False)
    pipeline = MappingPipeline(mapping_cfg, pulse_width=pulse_width)
    if debug:
        print(f"[{analysis.id}] debug mode enabled")
    mapped = pipeline.run(combined_df=combined, ext_datasets=ext, debug=debug)
    typer.echo(f"[{analysis.id}] mapping complete -> {len(mapped)} rows.")

    return {
        "dataframes": {
            "mapped": mapped.reset_index(drop=True),
        },
        "figures": [],
        "prints": [],
    }
