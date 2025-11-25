# src/stimloss/mapping.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, Iterable, Tuple

import numpy as np
import re
import pandas as pd

from .strategies import calculate_ploss


__all__ = ["MappingPipeline", "quantize_and_window", "map_measured_to_dataset"]


# ---------- helpers ----------

def _to_scalar_float_series(s: pd.Series) -> pd.Series:
    """
    Coerce a Series to scalar floats.
    - numeric -> float
    - strings -> parsed float
    - lists/arrays -> mean
    - else -> NaN
    """
    sn = pd.to_numeric(s, errors="coerce")
    if sn.dtype != "O":
        return sn.astype(float)

    def _coerce(v):
        if v is None:
            return np.nan
        if isinstance(v, (list, tuple, np.ndarray)):
            a = np.asarray(v, dtype=float).ravel()
            return float(a.mean()) if a.size else np.nan
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        if isinstance(v, str):
            t = v.strip().replace(",", "")
            try:
                return float(t)
            except Exception:
                return np.nan
        try:
            return float(v)
        except Exception:
            return np.nan

    return s.map(_coerce).astype(float)


def _eval_into(df: pd.DataFrame, target_col: str, expr: str, local_extras: Optional[Dict[str, Any]] = None):
    """
    Evaluate a pandas expression into a new column `target_col`.
    """
    local = {**{c: df[c] for c in df.columns}, **({"round": np.round} if local_extras is None or "round" not in local_extras else {})}
    if local_extras:
        local.update(local_extras)
    # Use pandas.eval with python engine to allow callables in local dict.
    df[target_col] = pd.eval(expr, engine="python", parser="pandas", local_dict=local)


def _apply_simple_transform(col: pd.Series, spec: Dict[str, Any]) -> pd.Series:
    """
    Apply tiny set of simple transforms used in joins 'pre' section.
    Supported keys: mul, add, sub, div, round (bool), astype (pandas dtype string)
    """
    out = col
    if "mul" in spec:
        out = pd.to_numeric(out, errors="coerce") * float(spec["mul"])
    if "add" in spec:
        out = pd.to_numeric(out, errors="coerce") + float(spec["add"])
    if "sub" in spec:
        out = pd.to_numeric(out, errors="coerce") - float(spec["sub"])
    if "div" in spec:
        out = pd.to_numeric(out, errors="coerce") / float(spec["div"])
    if spec.get("round"):
        out = pd.to_numeric(out, errors="coerce").round()
    if "astype" in spec:
        out = out.astype(spec["astype"])
    return out


# ---------- exported helpers ----------

def quantize_and_window(
    df: pd.DataFrame,
    *,
    I_step: float = 5e-6,
    Z_step: float = 1.0e4,
    I_range: Optional[Iterable[float]] = None,
    Z_range: Optional[Iterable[float]] = None,
    sanitize_cols: Tuple[str, str] = ("I", "Z"),
) -> pd.DataFrame:
    """
    Convenience helper: sanitize -> quantize -> window on I/Z.
    - I and Z are rounded to the provided steps.
    - Rows outside the provided ranges are dropped.
    """
    df = df.copy()
    for c in sanitize_cols:
        if c in df.columns:
            df[c] = _to_scalar_float_series(df[c])
    if "I" in df.columns:
        df["I"] = np.round(pd.to_numeric(df["I"], errors="coerce") / I_step) * I_step
    if "Z" in df.columns:
        df["Z"] = np.round(pd.to_numeric(df["Z"], errors="coerce") / Z_step) * Z_step
    if I_range is not None and "I" in df.columns:
        lo, hi = I_range
        df = df[(df["I"] >= lo) & (df["I"] <= hi)]
    if Z_range is not None and "Z" in df.columns:
        lo, hi = Z_range
        df = df[(df["Z"] >= lo) & (df["Z"] <= hi)]
    return df.dropna(subset=[c for c in ["I", "Z"] if c in df.columns]).reset_index(drop=True)


def map_measured_to_dataset(
    base_df: pd.DataFrame,
    measured_df: pd.DataFrame,
    *,
    I_step: float = 5e-6,
    Z_step: float = 1.0e4,
    I_col: str = "I",
    Z_col: str = "Z",
    meas_I_col: str = "Amp",
    meas_Z_col: str = "Rload",
    filter_query: Optional[str] = None,
    suffix: str = "_measured",
) -> pd.DataFrame:
    """
    Map measured datapoints onto a base dataframe by rounding I/Z to the same grid.
    - Both dataframes are quantized on the given steps.
    - Measured rows can be filtered with a query (e.g., 'Status == 1').
    - Result is a left-join; measured columns gain the provided suffix to avoid collisions.
    """
    bd = quantize_and_window(
        base_df,
        I_step=I_step,
        Z_step=Z_step,
        I_range=None,
        Z_range=None,
        sanitize_cols=(I_col, Z_col),
    ).copy()
    md = quantize_and_window(
        measured_df,
        I_step=I_step,
        Z_step=Z_step,
        I_range=None,
        Z_range=None,
        sanitize_cols=(meas_I_col, meas_Z_col),
    ).copy()

    if filter_query:
        try:
            md = md.query(filter_query).copy()
        except Exception as e:
            print(f"[WARN] Could not apply filter_query '{filter_query}': {e}")

    bd["I_int"] = (bd[I_col] * 1e6).round().astype("Int64")
    bd["Z_int"] = bd[Z_col].round().astype("Int64")
    md["I_int"] = (md[meas_I_col] * 1e6).round().astype("Int64")
    md["Z_int"] = md[meas_Z_col].round().astype("Int64")

    # avoid renaming keys; suffix only data columns
    data_cols = [c for c in md.columns if c not in {"I_int", "Z_int"}]
    md_renamed = md[["I_int", "Z_int"] + data_cols].copy()
    md_renamed = md_renamed.rename(columns={c: f"{c}{suffix}" for c in data_cols})

    merged = bd.merge(md_renamed, on=["I_int", "Z_int"], how="left")
    return merged.reset_index(drop=True)


# ---------- pipeline ----------

@dataclass
class MappingPipeline:
    """
    Config-driven mapping pipeline.

    cfg (dict) expected sections (all optional):
      - sanitize: {columns: ["I","Z"], strategy: "mean-of-iterables"}
      - quantize: {I_step: 5e-6, Z_step: 1.0e4}
      - window:   {I_range: [lo, hi], Z_range: [lo, hi]}
      - model:    {fn: "calculate_ploss", params: {...}}   # fn reserved for future, we call calculate_ploss now
      - join_keys: {I_int: "round(I * 1e6)", Z_int: "round(Z)"}
      - joins: list of:
          - id: <dataset id string>
            left_on: [...]
            right_on: [...]
            pre:
              <right_col>: {mul/add/sub/div/round/astype: ...}
            coerce:
              left:
                <col>: {astype: "Int64", round: true}
              right:
                <col>: {astype: "Int64", round: true}
            derive:
              <new_col>: "<pandas expression using columns on merged df>"
      - filters:
          vir_tolerance: 0.10
          vload_column: "Vload_meas"

    pulse_width is provided so derives can use it (e.g., "Eout / pulse_width").
    """
    cfg: Dict[str, Any]
    pulse_width: float = 40e-6

    # ---- phases ----

    def _sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        scfg = self.cfg.get("sanitize") or {}
        cols = scfg.get("columns", ["I", "Z"])
        if not cols:
            return df
        df = df.copy()
        for c in cols:
            if c in df.columns:
                df[c] = _to_scalar_float_series(df[c])
        return df

    def _quantize(self, df: pd.DataFrame) -> pd.DataFrame:
        qcfg = self.cfg.get("quantize") or {}
        I_step = float(qcfg.get("I_step", 5e-6))
        Z_step = float(qcfg.get("Z_step", 1.0e4))
        df = df.copy()
        if "I" in df.columns:
            df["I"] = np.round(pd.to_numeric(df["I"], errors="coerce") / I_step) * I_step
        if "Z" in df.columns:
            df["Z"] = np.round(pd.to_numeric(df["Z"], errors="coerce") / Z_step) * Z_step
        return df

    def _window(self, df: pd.DataFrame) -> pd.DataFrame:
        wcfg = self.cfg.get("window") or {}
        I_range = wcfg.get("I_range")
        Z_range = wcfg.get("Z_range")
        df = df.copy()
        if I_range is not None and "I" in df.columns:
            lo, hi = I_range
            df = df[(df["I"] >= lo) & (df["I"] <= hi)]
        if Z_range is not None and "Z" in df.columns:
            lo, hi = Z_range
            df = df[(df["Z"] >= lo) & (df["Z"] <= hi)]
        return df.dropna(subset=[c for c in ["I", "Z"] if c in df.columns])

    def _model(self, df: pd.DataFrame) -> pd.DataFrame:
        mcfg = self.cfg.get("model")
        if not mcfg:
            # If no model section, skip loss computation.
            return df
        params = mcfg.get("params", {})
        # We call your calculate_ploss directly; 'fn' in config is reserved for future
        return calculate_ploss(df, **params)

    def _compute_join_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        jcfg = self.cfg.get("join_keys") or {}
        if not jcfg:
            return df
        df = df.copy()
        for new_col, expr in jcfg.items():
            _eval_into(df, new_col, expr, local_extras={"round": np.round})
            # common case: integer-ish keys
            if df[new_col].dtype.kind in "fc":
                # nullable Int64 keeps NaN capability
                df[new_col] = pd.to_numeric(df[new_col], errors="coerce").round().astype("Int64")
        return df

    def _apply_one_join(self, left: pd.DataFrame, right: pd.DataFrame, j: Dict[str, Any]) -> pd.DataFrame:
        if right is None:
            return left
        # work on a copy of right
        r = right.copy()

        # 1) pre transforms on right
        for col, spec in (j.get("pre") or {}).items():
            if col in r.columns:
                r[col] = _apply_simple_transform(r[col], spec)

        # 2) coerce types/round both sides
        for side, df_side in (("left", left), ("right", r)):
            for col, spec in (j.get("coerce", {}).get(side) or {}).items():
                if col in df_side.columns:
                    tmp = df_side[col]
                    if spec.get("round"):
                        tmp = pd.to_numeric(tmp, errors="coerce").round()
                    if "to" in spec:
                        tmp = tmp.astype(spec["to"])
                    elif "astype" in spec:
                        tmp = tmp.astype(spec["astype"])
                    df_side[col] = tmp

        # 2b) optional filter on right (e.g., Status == 1)
        filt = (j.get("filter") or {}).get("right")
        if filt:
            if "query" in filt:
                r = r.query(filt["query"])
            if "equals" in filt:
                for col, val in filt["equals"].items():
                    if col in r.columns:
                        r = r[r[col] == val]

        # 3) merge
        left_on = j.get("left_on", [])
        right_on = j.get("right_on", [])
        # Use deterministic suffix so derives can reference right-hand columns unambiguously.
        suffix = f"_{j.get('id', 'right')}"
        right_cols = list(r.columns)
        merged = left.merge(
            r,
            left_on=left_on,
            right_on=right_on,
            how="left",
            suffixes=("", suffix),
        )
        # Force-suffix any right-hand columns that did not collide (pandas only suffixes collisions).
        for col in right_cols:
            if col in right_on:
                continue  # keep join keys as-is
            target_name = f"{col}{suffix}"
            if target_name in merged.columns:
                continue  # already suffixed by pandas collision
            if col in merged.columns:
                merged = merged.rename(columns={col: target_name})

        # 4) derive new columns on merged (expressions)
        for new_col, expr in (j.get("derive") or {}).items():
            # ensure column exists even if eval fails
            if new_col not in merged.columns:
                merged[new_col] = pd.NA
            # only warn about Ein/Eout if the expression references those exact names
            tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))
            missing = [x for x in ("Ein", "Eout") if x in tokens and x not in merged.columns]
            if missing:
                print(f"[DEBUG] Missing columns {missing} in join id={j.get('id')}! Available: {list(merged.columns)[:20]}...")
            try:
                _eval_into(merged, new_col, expr, local_extras={"round": np.round, "pulse_width": self.pulse_width})
            except Exception as e:
                # Keep column as NaN but continue pipeline
                print(f"[WARN] Derive '{new_col}' for join id={j.get('id')} failed; leaving NaN. Error: {e}")
                continue

        # for new_col, expr in (j.get("derive") or {}).items():
        #     _eval_into(merged, new_col, expr, local_extras={"round": np.round, "pulse_width": self.pulse_width})

        return merged

    def _apply_joins(self, df: pd.DataFrame, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        for j in self.cfg.get("joins", []) or []:
            ds_id = j["id"]
            right = datasets.get(ds_id)
            df = self._apply_one_join(df, right, j)
        return df

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        fcfg = self.cfg.get("filters") or {}
        tol = fcfg.get("vir_tolerance")
        vcol = fcfg.get("vload_column", "Vload_meas")
        if tol is None or vcol not in df.columns or "I" not in df.columns or "Z" not in df.columns:
            return df

        V = df[vcol].apply(lambda v: np.mean(v) if hasattr(v, "__iter__") and not np.isscalar(v) else v)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (V / df["Z"]) / df["I"]
        mask = np.isfinite(ratio) & (np.abs(ratio - 1.0) < float(tol))
        return df[mask].copy()

    # ---- public API ----

    def run(self, combined_df: pd.DataFrame, ext_datasets: Dict[str, pd.DataFrame], debug: bool = False) -> pd.DataFrame:
        """
        Execute the pipeline and return the pointwise, fully-merged DataFrame.
        """
        if "I" not in combined_df.columns or "Z" not in combined_df.columns:
            raise ValueError("combined_df must contain 'I' and 'Z' columns (in A and Ω respectively).")

        df = combined_df.copy()

        # 1) sanitize -> 2) quantize -> 3) window
        df = self._sanitize(df)
        if debug:
            print(f"[mapping] after sanitize: {len(df)} rows")
        df = self._quantize(df)
        if debug:
            print(f"[mapping] after quantize: {len(df)} rows")
        df = self._window(df)
        if debug:
            print(f"[mapping] after window: {len(df)} rows")

        # 4) model (calculate_ploss)
        df = self._model(df)
        if debug:
            cols_preview = list(df.columns)[:8]
            print(f"[mapping] after model: columns={cols_preview} ...")

        # 5) join keys (e.g., I_int, Z_int)
        df = self._compute_join_keys(df)
        if debug:
            print(f"[mapping] after join key compute: {len(df)} rows")
        
        # 6) joins (measured/sim, in order)
        df = self._apply_joins(df, ext_datasets)
        if debug:
            print(f"[mapping] after joins: {len(df)} rows")

        # 7) filters (e.g., V ≈ I·R)
        df = self._apply_filters(df)
        if debug:
            print(f"[mapping] after filters: {len(df)} rows")

        return df.reset_index(drop=True)
