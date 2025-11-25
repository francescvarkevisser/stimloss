# src/stimloss/runners/__init__.py
from .mapping_runner import run as _mapping_run  # triggers @register_analysis (mapping)
from .strategies_eval import run as _strategies_run  # triggers @register_analysis (strategies)
from .strategies_sweep import run as _strategies_sweep_run  # triggers @register_analysis (strategies_sweep)
