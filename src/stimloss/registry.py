from __future__ import annotations
from typing import Callable, Dict

_ANALYSIS_REGISTRY: Dict[str, Callable] = {}

def register_analysis(name: str):
    def _wrap(fn: Callable):
        _ANALYSIS_REGISTRY[name] = fn
        return fn
    return _wrap

def get_analysis_runner(name: str) -> Callable:
    try:
        return _ANALYSIS_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown analysis type: {name}")

def list_analysis_types():
    return sorted(_ANALYSIS_REGISTRY.keys())
