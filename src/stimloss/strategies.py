from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
try:
    import typer  # for progress bars
except Exception:  # pragma: no cover - optional
    typer = None


# ---------- Common helpers ----------

def _clean_channel_yield(value: Optional[float]) -> Optional[float]:
    """Return a valid yield fraction in [0,1], else None."""
    if value is None:
        return None
    try:
        val = float(value)
    except Exception:
        return None
    if pd.isna(val) or val < 0 or val > 1:
        return None
    return val


# ---------- Loss / modeling helpers ----------

def calculate_voltage_distribution(distribution: str = "uniform", Vmax: float = 5, nsteps: int = 1, alpha: float = 0.5) -> np.ndarray:
    """
    Return a sorted voltage distribution array based on the chosen distribution.
    distribution: 'uniform' | 'exp' | 'invexp'
    """
    if distribution == "uniform":
        return np.sort(np.linspace(0, Vmax, nsteps + 1)[1:])
    if distribution == "exp":
        exp_values = np.exp(-np.linspace(0, Vmax, nsteps + 1)[1:] * alpha)
        return np.sort(exp_values * Vmax / exp_values.max())
    if distribution == "invexp":
        inv_exp = 1 - np.exp(-np.linspace(0, Vmax, nsteps + 1)[1:] * alpha)
        return np.sort(inv_exp * Vmax / inv_exp.max())
    raise ValueError(f"Invalid distribution type: {distribution}")


def calculate_ploss(
    df: pd.DataFrame,
    Vmax: Optional[float] = None,
    nsteps: int = 1,
    Vheadroom: float = 0.0,
    dist: str = "uniform",
    alpha: float = 0.5,
    Vsubject: bool = False,
    channel_yield: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute stepped-supply losses/efficiency for each row of df.
    Expects columns: I, Vload, source.
    """
    tmp = df.copy()

    # normalize channel_yield (treat NaN or out-of-range as None)
    channel_yield = _clean_channel_yield(channel_yield)
    if channel_yield is None:
        Vsubject = False
    if Vmax is None:
        Vmax = tmp["Vload"].quantile(channel_yield) if channel_yield is not None else tmp["Vload"].max()
    if "source" not in tmp.columns:
        raise ValueError("DataFrame must contain a 'source' column")

    if Vsubject:
        source_vload = tmp.groupby("source")["Vload"].quantile(channel_yield).reset_index()
        source_vload = source_vload.rename(columns={"Vload": "Vmax"})
        tmp = tmp.merge(source_vload[["source", "Vmax"]], on="source", how="left")
    else:
        tmp["Vmax"] = Vmax

    def calculate_vdist(vmax_val: float, steps: int) -> np.ndarray:
        return calculate_voltage_distribution(dist, vmax_val, steps, alpha)

    rounding_arrays = {
        source: {j: calculate_vdist(tmp[tmp["source"] == source]["Vmax"].iloc[0], j) for j in range(nsteps + 1)}
        for source in tmp["source"].unique()
    }

    tmp["Pload"] = tmp["I"] * tmp["Vload"]
    tmp["Pscaled"] = tmp["I"] * (tmp["Vload"] + Vheadroom)
    tmp["Ploss_scaled"] = tmp["Pscaled"] - tmp["Pload"]
    
    for j in range(1, nsteps + 1):
        for source, vdist in rounding_arrays.items():
            for Vstep in vdist[j][::-1]:
                mask = (tmp["source"] == source) & (tmp["Vload"] <= Vstep - Vheadroom)
                tmp.loc[mask, f"Vstep{j}"] = Vstep

            tmp[f"P_stepped{j}"] = tmp[f"Vstep{j}"] * tmp["I"]
            tmp[f"Ploss_stepped{j}"] = tmp[f"P_stepped{j}"] - tmp["Pload"]
            tmp[f"Efficiency_stepped{j}"] = tmp["Pload"] / tmp[f"P_stepped{j}"]

    tmp["Ploss_percentage"] = tmp["Ploss_stepped1"] / tmp["P_stepped1"]
    tmp["channel_yield"] = channel_yield
    return tmp.dropna(subset=["Vstep1"]).reset_index(drop=True)

def resample_dataset(
    df: pd.DataFrame,
    n_samples: int = 4,
    n_repeats: int = 1000,
    *,
    show_progress: bool = False,
    progress_label: str = "Sampling",
    compute_global: bool = True,
    compute_global_Vheadroom: float = 0.0,
) -> pd.DataFrame:
    """
    Repeatedly sample n channels from df.
    Optionally compute global losses/efficiency for each iteration (compute_global=True).
    """
    resultlist: List[pd.DataFrame] = []
    progress_every = max(1, n_repeats // 10) if show_progress else None
    for i in range(n_repeats):
        sampled_df = df.sample(n=n_samples).reset_index().rename(columns={"index": "original_index"})
        sampled_df["iteration"] = i
        sampled_df["sample"] = sampled_df.index
        if compute_global:
            global_calc = calculate_ploss(
                sampled_df,
                Vmax=sampled_df["Vload"].max(),
                nsteps=1,
                Vheadroom=compute_global_Vheadroom,
            )
            sampled_df[["Ploss_global", "Efficiency_global"]] = global_calc[["Ploss_stepped1", "Efficiency_stepped1"]]
        resultlist.append(sampled_df)
        if show_progress and ((i + 1) % progress_every == 0 or i == n_repeats - 1):
            print(f"{progress_label}: {i+1}/{n_repeats}", end="\r", flush=True)
    if show_progress:
        print()

    results_df = pd.concat(resultlist)
    return results_df


def resample_dataset_by_group(
    df: pd.DataFrame,
    group_cols: List[str] = None,
    n_samples_dict: Optional[Dict[str, int]] = None,
    default_n_samples: int = 4,
    n_repeats: int = 1000,
    show_progress: bool = False,
    compute_global: bool = True,
    compute_global_Vheadroom: float = 0.0,
) -> pd.DataFrame:
    """
    Apply resample_dataset to subsets of df grouped by group_cols.
    n_samples_dict can override sample sizes per Target.
    """
    if group_cols is None:
        group_cols = ["Target"]

    group_list = list(df.groupby(group_cols))
    if show_progress:
        msg = f"[stimloss] Preparing sampling for {len(group_list)} group(s); {n_repeats} iterations each (eligible groups only)."
        if typer is not None:
            typer.echo(msg)
        else:
            logger.info(msg)

    resultlist: List[pd.DataFrame] = []
    for group_vals, group_df in group_list:
        target = group_df["Target"].unique()[0]
        n_samples = n_samples_dict.get(target, default_n_samples) if n_samples_dict else default_n_samples
        label = ", ".join(str(v) for v in group_vals) if isinstance(group_vals, (list, tuple)) else str(group_vals)

        if n_samples < len(group_df):
            if n_samples > len(group_df) / 5:
                logger.warning(f"Sample size close to group size for: {group_vals}, ratio={n_samples/len(group_df)}")
            group_result = resample_dataset(
                group_df,
                n_samples,
                n_repeats,
                show_progress=show_progress,
                progress_label=f"Sampling {label}",
                compute_global=compute_global,
                compute_global_Vheadroom=compute_global_Vheadroom,
            )
            resultlist.append(group_result)
        else:
            logger.warning(f"Dropping group: {group_vals}")

    if not resultlist:
        return pd.DataFrame()
    return pd.concat(resultlist, ignore_index=True)


# Backwards-compatible aliases
def calculate_global_losses_repeated(*args, **kwargs):
    return resample_dataset(*args, **kwargs)


def calculate_global_losses_by_group(*args, **kwargs):
    return resample_dataset_by_group(*args, **kwargs)


@dataclass
class StrategySpec:
    """
    A strategy declares how to pull (or compute) efficiency & loss columns,
    plus a human-friendly label for plotting/reporting.
    """
    id: str
    label: str
    efficiency: str        # column name or pandas expression
    loss: str              # column name or pandas expression


def _eval_series_or_expr(df: pd.DataFrame, expr: str) -> pd.Series:
    """If `expr` is a column name, return it; otherwise evaluate as pandas expression."""
    if expr in df.columns:
        return pd.to_numeric(df[expr], errors="coerce")
    # allow round(), np.round(), and constants; block unknown names
    local = {c: df[c] for c in df.columns}
    local.update({"round": np.round, "np": np})
    return pd.eval(expr, engine="python", parser="pandas", local_dict=local)


# def evaluate_strategy_on_df(df: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
#     """
#     Build a 3-column DataFrame: [Condition, Efficiency, Ploss] from `df`.
#     """
#     out = pd.DataFrame(index=df.index)
#     out["Efficiency"] = _eval_series_or_expr(df, spec.efficiency)
#     out["Ploss"]      = _eval_series_or_expr(df, spec.loss)
#     out["Condition"]  = spec.label
#     return out

# ---------- Typed strategy helper ----------
def build_results_from_strategies(
    strategies_cfg: Iterable[Dict[str, Any]],
    *,
    pointwise_df: pd.DataFrame,
    summary_df: Optional[pd.DataFrame] = None,
    group_cols: Optional[List[str]] = None,
    default_n_samples: int = 4,
    n_samples_dict: Optional[Dict[str, int]] = None,
    n_repeats: int = 1000,
    show_progress: bool = False,
    model_defaults: Optional[Dict[str, Any]] = None,
    efficiency_to_percent: bool = True,
    ploss_to_uW: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Evaluate strategies that declare a type:
      - fixed: stepped supply with nsteps=1
      - stepped: stepped supply with configurable nsteps/dist/Vheadroom/alpha/Vsubjectject/channel_yield/Vmax
      - global: uses Efficiency_global/Ploss_global (from calculate_global_losses_by_group)
    Falls back to column/expr strategies (legacy) if no type is given.
    Returns (long_df, summary_df). summary_df is generated via resampling when needed.
    """
    if not strategies_cfg:
        return pd.DataFrame(columns=["Condition", "Efficiency", "Ploss"]), summary_df

    model_defaults = model_defaults or {}
    model_defaults.setdefault("channel_yield", 0.75)
    parts: List[pd.DataFrame] = []

    def _convert_units(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if efficiency_to_percent and "Efficiency" in out:
            out["Efficiency"] = out["Efficiency"] * 100.0
        if ploss_to_uW and "Ploss" in out:
            out["Ploss"] = out["Ploss"] * 1e6
        return out

    resultlist: List[pd.DataFrame] = []
    max_step = 0
    calculate_global = False
    for cfg in strategies_cfg:
        if cfg.get("type") == "stepped":
            max_step = max(max_step, int(cfg.get("nsteps", model_defaults.get("nsteps", 1))))

        if calculate_global == False:
            if cfg.get("type") == "global":
                calculate_global = True

    
    for cfg in strategies_cfg:
        stype = cfg.get("type")
        label = cfg.get("label", cfg.get("id"))
        Vheadroom = cfg.get("Vheadroom", model_defaults.get("Vheadroom", 0.0))
        Vmax = cfg.get("Vmax", model_defaults.get("Vmax"))
        Vsubject = cfg.get("Vsubject", model_defaults.get("Vsubject", False))
        alpha = cfg.get("alpha", model_defaults.get("alpha", 0.5))
        dist = cfg.get("dist", model_defaults.get("dist", "uniform"))
        channel_yield = _clean_channel_yield(cfg.get("channel_yield", model_defaults.get("channel_yield")))
        if channel_yield is None:
            Vsubject = False

        calc_df = calculate_ploss(
            pointwise_df,
            Vmax=Vmax,
            nsteps=max_step,
            Vheadroom=Vheadroom,
            dist=dist,
            alpha=alpha,
            Vsubject=Vsubject,
            channel_yield=channel_yield,
        )
        calc_df['label'] = label
        calc_df["id"] = cfg.get("id")
        calc_df['dist_type'] = dist
        calc_df['alpha'] = alpha
        calc_df['Vheadroom'] = Vheadroom
        resultlist.append(calc_df)
    
    
    final_df = pd.concat(resultlist, ignore_index=True)
    final_df = final_df.astype({col: 'float32' for col in final_df.select_dtypes(include=['float64']).columns})
    logger.info('Calculated final_df')
    del resultlist

    resampled_df = resample_dataset_by_group(
        df=final_df,
        group_cols=group_cols,
        n_samples_dict=n_samples_dict,
        default_n_samples=default_n_samples,
        n_repeats=n_repeats,
        show_progress=show_progress,
        compute_global=calculate_global,
        compute_global_Vheadroom=Vheadroom
    )
    group_keys = group_cols + ["iteration"]
    meta_columns = resampled_df.select_dtypes('object').columns.difference(group_keys)
    numeric_columns = resampled_df.select_dtypes(include="number").columns.difference(group_keys)
    agg = {col: "mean" for col in numeric_columns}
    agg.update({col: "first" for col in meta_columns})
    mean_df = resampled_df.groupby(group_keys).agg(agg).reset_index()

    # Populate generic ploss/efficiency columns on mean_df based on strategy types
    mean_df["ploss"] = np.nan
    mean_df["efficiency"] = np.nan
    for cfg in strategies_cfg:
        stype = cfg.get("type")
        label = cfg.get("label", cfg.get("id"))
        id = cfg.get('id')
        mask = mean_df["id"] == id
        if not mask.any():
            continue
        if stype == "fixed":
            mean_df.loc[mask, "nsteps"] = 'Fixed'
            mean_df.loc[mask, "ploss"] = mean_df.loc[mask, "Ploss_stepped1"]
            mean_df.loc[mask, "efficiency"] = mean_df.loc[mask, "Efficiency_stepped1"]
        elif stype == "stepped":
            n = cfg.get("nsteps")
            mean_df.loc[mask, "nsteps"] = str(n)
            mean_df.loc[mask, "ploss"] = mean_df.loc[mask, f"Ploss_stepped{n}"]
            mean_df.loc[mask, "efficiency"] = mean_df.loc[mask, f"Efficiency_stepped{n}"]
        elif stype == "expression" or stype is None:
            eff_expr = cfg.get("efficiency")
            loss_expr = cfg.get("loss")
            if eff_expr is None or loss_expr is None:
                continue
            try:
                mean_df.loc[mask, "efficiency"] = _eval_series_or_expr(mean_df.loc[mask], eff_expr)
                mean_df.loc[mask, "ploss"] = _eval_series_or_expr(mean_df.loc[mask], loss_expr)
            except Exception as e:
                logger.warning(f"Skipping expression aggregation for '{id}': {e}")
                continue
        else:  # global or other
            mean_df.loc[mask, "ploss"] = mean_df.loc[mask, "Ploss_global"]
            mean_df.loc[mask, "efficiency"] = mean_df.loc[mask, "Efficiency_global"]
            mean_df.loc[mask, "nsteps"] = 'Global'

    if efficiency_to_percent:
        mean_df['efficiency'] = mean_df['efficiency']*100
    if ploss_to_uW:
        mean_df['ploss'] = mean_df['ploss']*1e6
    return final_df.reset_index(drop=True), resampled_df, mean_df #, long_df
