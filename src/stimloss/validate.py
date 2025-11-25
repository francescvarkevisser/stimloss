from __future__ import annotations

import os
from typing import List, Tuple

from .config import StimlossConfig, Dataset
from .io_utils import peek_columns

REQUIRED_COLS = {
    "combined": ["I", "Z", "Target", "source"],   # <- capital T
    "measured": ["Amp", "Rload", "Pload_mean", "Pout_mean", "Pin_compensated"],
    "sim":      ["Amp", "Load", "Ein", "Eout", "efficiency"],
    "sim_pl":   ["Amp", "Load", "Ein", "Eout", "efficiency"],
}

def check_paths(cfg: StimlossConfig) -> List[str]:
    errs: List[str] = []
    for d in cfg.data["datasets"]:
        assert isinstance(d, Dataset)
        if not os.path.exists(d.path):
            errs.append(f"[dataset:{d.id}] missing file: {d.path}")
    return errs


def check_columns(cfg: StimlossConfig) -> List[str]:
    errs: List[str] = []
    for d in cfg.data["datasets"]:
        assert isinstance(d, Dataset)
        required = REQUIRED_COLS.get(d.kind, [])
        if not required:
            continue
        have = peek_columns(d.path, d.format)
        missing = [c for c in required if c not in have]
        if missing:
            errs.append(
                f"[dataset:{d.id}] missing columns: {missing} (have: {have})"
            )
    return errs


def check_analyses(cfg: StimlossConfig) -> List[str]:
    errs: List[str] = []
    dataset_ids = {d.id for d in cfg.data["datasets"]}
    for a in cfg.analyses:
        for key, val in a.inputs.items():
            if val is None:
                continue
            if val not in dataset_ids:
                errs.append(
                    f"[analysis:{a.id}] input '{key}' refers to unknown dataset '{val}'"
                )
    return errs


def validate_config(cfg: StimlossConfig) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    errs += check_paths(cfg)
    errs += check_columns(cfg)
    errs += check_analyses(cfg)
    return (len(errs) == 0, errs)
