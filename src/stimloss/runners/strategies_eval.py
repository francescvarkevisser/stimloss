from __future__ import annotations
from typing import Any, Dict
import pandas as pd
import itertools

from ..io_utils import read_table
from ..strategies import build_results_from_strategies
from ..registry import register_analysis


def _expand_strategies(strategies_cfg):
    """
    Expand strategies when any param is a list/tuple/set.
    Generates cartesian products over list-valued params (excluding id/label/type),
    and appends suffixes to id/label.
    """
    expanded = []
    for cfg in strategies_cfg:
        vary = {k: v for k, v in cfg.items() if isinstance(v, (list, tuple, set)) and k not in {"id", "label", "type"}}
        if not vary:
            expanded.append(cfg)
            continue
        keys = list(vary.keys())
        combos = itertools.product(*[list(vary[k]) for k in keys])
        base = {k: v for k, v in cfg.items() if k not in vary}
        base_id = cfg.get("id", "")
        base_label = cfg.get("label", base_id)
        for combo in combos:
            new_cfg = dict(base)
            suffix_parts = []
            for k, val in zip(keys, combo):
                new_cfg[k] = val
                suffix_parts.append(str(val))
            suffix = "_".join(suffix_parts)
            if base_id:
                new_cfg["id"] = f"{base_id}_{suffix}"
            if base_label:
                new_cfg["label"] = f"{base_label} {suffix}"
            expanded.append(new_cfg)
    return expanded


@register_analysis("strategies")
def run(analysis, datasets_by_id, project_cfg) -> Dict[str, Any]:
    """
    Evaluate strategies (Fixed/Stepped/Global) on any dataframe that has the needed columns
    (e.g., mapped output with I, Z, Vload, source).
    """
    def _get_df(ds_id):
        ds = datasets_by_id[ds_id]
        return read_table(ds.path, ds.format)

    # Accept either explicit "data"/"mapped" id or fall back to the first provided input id.
    data_id = analysis.inputs.get("data") # or analysis.inputs.get("mapped")
    if data_id is None:
        # if user passed a dict with arbitrary key, take the first value
        if analysis.inputs:
            data_id = next(iter(analysis.inputs.values()))
        else:
            raise ValueError("strategies runner expects a dataset id in analysis.inputs (e.g., 'data' or 'mapped').")
    df = _get_df(data_id)

    # optional target filter (config param)
    tgt = analysis.params.get("target")
    if tgt is not None and isinstance(df, pd.DataFrame):
        targets = tgt if isinstance(tgt, (list, tuple, set)) else [tgt]
        if "Target" in df.columns:
            df = df.loc[df["Target"].isin(targets)].copy()
        elif "target" in df.columns:
            df = df.loc[df["target"].isin(targets)].copy()

    # optional source filter (config param)
    src = analysis.params.get("source")
    if src is not None and isinstance(df, pd.DataFrame):
        sources = src if isinstance(src, (list, tuple, set)) else [src]
        if "source" in df.columns:
            df = df.loc[df["source"].isin(sources)].copy()
        elif "Source" in df.columns:
            df = df.loc[df["Source"].isin(sources)].copy()

    group_cols = analysis.params.get("group_cols", [])#['Target', 'source', 'channel_yield', 'Vheadroom', 'label'])
    group_cols = group_cols + ['id']
    n_repeats = analysis.params.get("n_repeats", 1000)
    default_n_samples = analysis.params.get("default_n_samples", 200)
    n_samples_dict = analysis.params.get("n_samples_dict")
    show_progress = analysis.params.get("show_progress", True)
    yield_filter = analysis.params.get("yield_filter")  # optional percentile pre-filter per source
    model_defaults = (analysis.params.get("model") or {}).get("params", {})
    strategies_cfg = analysis.params.get("strategies") or []
    strategies_cfg = _expand_strategies(strategies_cfg)
    include_labels = analysis.params.get("include_strategies")

    final_df, resampled_df, mean_df = build_results_from_strategies(
        strategies_cfg,
        pointwise_df=df,
        group_cols=group_cols,
        default_n_samples=default_n_samples,
        n_samples_dict=n_samples_dict,
        n_repeats=n_repeats,
        show_progress=show_progress,
        model_defaults=model_defaults,
        efficiency_to_percent=True,
        ploss_to_uW=True,
    )

    return {
        "dataframes": {
            "data": df.reset_index(drop=True),
            "mean": mean_df.reset_index(drop=True) if mean_df is not None else pd.DataFrame(),
            "final": final_df.reset_index(drop=True),
            "resampled": resampled_df.reset_index(drop=True)
        },
        "figures": [],
        "prints": [],
    }
