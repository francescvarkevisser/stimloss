from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from .config import GenerationTask


def _sample_truncated_normal(mean: float, sd: float, low: float, high: float, step: Optional[float], size: int) -> np.ndarray:
    """Sample from a truncated normal; optionally snap to step size."""
    a, b = (low - mean) / sd, (high - mean) / sd
    samples = stats.truncnorm.rvs(a, b, loc=mean, scale=sd, size=size)
    if step:
        samples = np.round(samples / step) * step
    return np.clip(samples, low, high)


def _sample_from_kde(series: pd.Series, size: int) -> np.ndarray:
    """Sample from a KDE estimated on the provided series."""
    arr = pd.to_numeric(series.dropna(), errors="coerce").dropna().to_numpy()
    if arr.size == 0:
        return np.array([])
    kde = stats.gaussian_kde(arr)
    samples = kde.resample(size).ravel()
    return samples


def _estimate_sd_from_iqr(iqr: float) -> float:
    """For normal distributions, std ≈ IQR / 1.349."""
    return iqr / 1.349 if pd.notna(iqr) else np.nan


@dataclass
class SynthConfig:
    method: str              # "mean_sd" | "median_iqr" | "dataset"
    target: str
    study: Optional[str]
    dataset: Optional[str]
    n_samples: int = 10000

    # mean/sd fields
    I_mean: Optional[float] = None
    I_sd: Optional[float] = None
    I_min: Optional[float] = None
    I_max: Optional[float] = None
    I_step: Optional[float] = None
    Z_mean: Optional[float] = None
    Z_sd: Optional[float] = None
    Z_min: Optional[float] = None
    Z_max: Optional[float] = None
    Z_step: Optional[float] = None

    # median/IQR fields
    I_median: Optional[float] = None
    I_iqr: Optional[float] = None
    Z_median: Optional[float] = None
    Z_iqr: Optional[float] = None

    # dataset path
    data_path: Optional[str] = None


def _synthesize_one(cfg: SynthConfig) -> pd.DataFrame:
    """Synthesize a single dataset row according to method."""
    if cfg.method == "dataset":
        if not cfg.data_path or not os.path.exists(cfg.data_path):
            raise FileNotFoundError(f"data_path not found: {cfg.data_path}")
        raw = pd.read_csv(cfg.data_path)
        if not {"I", "Z"} <= set(raw.columns):
            raise ValueError(f"data_path must contain columns 'I' and 'Z': {cfg.data_path}")
        I_vals = _sample_from_kde(raw["I"], cfg.n_samples)
        Z_vals = _sample_from_kde(raw["Z"], cfg.n_samples)
    elif cfg.method == "median_iqr":
        I_mean = cfg.I_median
        I_sd = _estimate_sd_from_iqr(cfg.I_iqr)
        Z_mean = cfg.Z_median
        Z_sd = _estimate_sd_from_iqr(cfg.Z_iqr)
        I_vals = _sample_truncated_normal(I_mean, I_sd, cfg.I_min or 0, cfg.I_max or np.inf, cfg.I_step, cfg.n_samples)
        Z_vals = _sample_truncated_normal(Z_mean, Z_sd, cfg.Z_min or 0, cfg.Z_max or np.inf, cfg.Z_step, cfg.n_samples)
    elif cfg.method == "mean_sd":
        I_vals = _sample_truncated_normal(cfg.I_mean, cfg.I_sd, cfg.I_min or 0, cfg.I_max or np.inf, cfg.I_step, cfg.n_samples)
        Z_vals = _sample_truncated_normal(cfg.Z_mean, cfg.Z_sd, cfg.Z_min or 0, cfg.Z_max or np.inf, cfg.Z_step, cfg.n_samples)
    else:
        raise ValueError(f"Unknown method: {cfg.method}")

    out = pd.DataFrame({
        "I": I_vals,
        "Z": Z_vals,
        "Target": cfg.target,
        "Study": cfg.study,
        "Dataset": cfg.dataset,
    })
    # drop any rows with missing values
    out = out.dropna(subset=["I", "Z"])
    # assign a synthetic source id (per row config)
    out["source"] = np.arange(len(out))
    return out


def generate_from_table(path: str, *, sheet: Optional[str] = None, n_samples: int = 10000, output_dir: str = "data/bundles") -> str:
    """
    Generate a combined_df from a table (CSV or Excel) describing datasets.
    Expected columns include:
      - method: mean_sd | median_iqr | dataset
      - target, study, dataset (metadata)
      - For mean_sd: I_mean, I_sd, I_min, I_max, I_step, Z_mean, Z_sd, Z_min, Z_max, Z_step
      - For median_iqr: I_median, I_iqr, I_min, I_max, I_step, Z_median, Z_iqr, Z_min, Z_max, Z_step
      - For dataset: data_path (CSV with columns I,Z)
    """
    if path.lower().endswith((".xlsx", ".xls")):
        df_src = pd.read_excel(path, sheet_name=sheet or 0)
    else:
        df_src = pd.read_csv(path)

    rows = []
    for _, r in df_src.iterrows():
        cfg = SynthConfig(
            method=str(r.get("method", "")).strip(),
            target=r.get("target") or r.get("Target"),
            study=r.get("study") or r.get("Study"),
            dataset=r.get("dataset") or r.get("Dataset"),
            n_samples=int(r.get("n_samples", n_samples)),
            I_mean=r.get("I_mean"), I_sd=r.get("I_sd"), I_min=r.get("I_min"), I_max=r.get("I_max"), I_step=r.get("I_step"),
            Z_mean=r.get("Z_mean"), Z_sd=r.get("Z_sd"), Z_min=r.get("Z_min"), Z_max=r.get("Z_max"), Z_step=r.get("Z_step"),
            I_median=r.get("I_median"), I_iqr=r.get("I_iqr"),
            Z_median=r.get("Z_median"), Z_iqr=r.get("Z_iqr"),
            data_path=r.get("data_path"),
        )
        rows.append(_synthesize_one(cfg))

    combined = pd.concat(rows, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"combined_df_{ts}.parquet")
    combined.to_parquet(out_path, index=False)
    return out_path


def generate_from_meta(meta_glob: str, output_dir: str) -> List[str]:
    """
    Looks for YAML metafiles, expects each to have a 'measured_points' list,
    writes them out as CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    outputs: List[str] = []

    for meta_path in glob.glob(meta_glob):
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)

        df = pd.DataFrame(meta.get("measured_points", []))
        base = os.path.splitext(os.path.basename(meta_path))[0]
        out = os.path.join(output_dir, f"{base}_generated.csv")
        df.to_csv(out, index=False)
        outputs.append(out)

    return outputs


def run_generation_task(task: GenerationTask) -> List[str]:
    """
    Dispatch a simple generation task declared in config.
    Supports:
      - type: from_meta (uses meta_glob/output_dir)
      - type: from_literature (uses path/sheet/n_samples/output_dir)
    """
    outs: List[str] = []
    if task.type == "from_meta":
        if not task.meta_glob:
            raise ValueError(f"[{task.id}] meta_glob is required for type=from_meta")
        outs = generate_from_meta(task.meta_glob, task.output_dir)
    elif task.type == "from_literature":
        if not task.path:
            raise ValueError(f"[{task.id}] path is required for type=from_literature")
        out = generate_from_table(task.path, sheet=task.sheet, n_samples=task.n_samples, output_dir=task.output_dir)
        outs = [out]
    else:
        raise ValueError(f"[{task.id}] unknown generation task type: {task.type}")
    return outs
