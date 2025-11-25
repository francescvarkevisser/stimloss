from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd

from ..io_utils import read_table
from ..strategies import build_results_from_strategies
from ..registry import register_analysis


@register_analysis("strategies_sweep")
def run(analysis, datasets_by_id, project_cfg) -> Dict[str, Any]:
    """
    Sweep a parameter over multiple values and evaluate strategies at each point.
    Example use: sweep channel_yield or Vheadroom for stepped supplies.
    """
    def _get_df(ds_id):
        ds = datasets_by_id[ds_id]
        return read_table(ds.path, ds.format)

    data_id = analysis.inputs.get("data")
    if data_id is None:
        raise ValueError("strategies_sweep expects 'data' in analysis.inputs.")
    df = _get_df(data_id)

    target = analysis.params.get("target")
    if target is not None:
        if "Target" in df.columns:
            df = df[df["Target"] == target].copy()

    sweep_param = analysis.params.get("sweep_param")
    values: List[Any] = analysis.params.get("values", [])
    if not sweep_param or not values:
        raise ValueError("strategies_sweep requires sweep_param and values in params.")

    group_cols = analysis.params.get("group_cols", ["source"])
    n_repeats = analysis.params.get("n_repeats", 100)
    default_n_samples = analysis.params.get("default_n_samples", 100)
    n_samples_dict = analysis.params.get("n_samples_dict")
    show_progress = analysis.params.get("show_progress", False)
    base_model_defaults = (analysis.params.get("model") or {}).get("params", {})
    strategies_cfg = analysis.params.get("strategies") or []

    long_parts: List[pd.DataFrame] = []
    summary_parts: List[pd.DataFrame] = []

    for val in values:
        md = dict(base_model_defaults)
        md[sweep_param] = val
        long_df, summary_df = build_results_from_strategies(
            strategies_cfg,
            pointwise_df=df,
            summary_df=None,
            group_cols=group_cols,
            default_n_samples=default_n_samples,
            n_samples_dict=n_samples_dict,
            n_repeats=n_repeats,
            show_progress=show_progress,
            model_defaults=md,
            efficiency_to_percent=True,
            ploss_to_uW=True,
        )
        long_df[sweep_param] = val
        long_parts.append(long_df)
        if summary_df is not None:
            summary_df[sweep_param] = val
            summary_parts.append(summary_df)

    long_all = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()
    summary_all = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()

    return {
        "dataframes": {
            "long": long_all.reset_index(drop=True),
            "summary": summary_all.reset_index(drop=True),
        },
        "figures": [],
        "prints": [],
    }
