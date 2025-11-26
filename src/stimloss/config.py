from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class Dataset:
    id: str
    path: str
    kind: str               # "combined" | "measured" | "sim" | "sim_pl"
    format: str = "parquet"


@dataclass
class Subject:
    id: str
    datasets: List[str]
    meta: Optional[str] = None


@dataclass
class AnalysisOutput:
    figures: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    print: List[str] = field(default_factory=list)


@dataclass
class Analysis:
    id: str
    type: str               # "mapping_compare" etc.
    inputs: Dict[str, Optional[str]]
    params: Dict[str, Any] = field(default_factory=dict)
    outputs: AnalysisOutput = field(default_factory=AnalysisOutput)


@dataclass
class GenerationTask:
    id: str
    type: str               # e.g. "from_meta" | "from_literature"
    meta_glob: Optional[str] = None
    output_dir: str = "data/bundles"
    output_path: Optional[str] = None
    path: Optional[str] = None       # for from_literature
    sheet: Optional[str] = None
    n_samples: int = 10000


@dataclass
class Generation:
    tasks: List[GenerationTask] = field(default_factory=list)


@dataclass
class StimlossConfig:
    version: int
    project: Dict[str, Any]
    data: Dict[str, Any]           # {"datasets": List[Dataset], "subjects": List[Subject]}
    analyses: List[Analysis]
    generation: Optional[Generation] = None


def load_config(path: str) -> StimlossConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    datasets = [Dataset(**d) for d in cfg["data"]["datasets"]]
    subjects = [Subject(**s) for s in cfg["data"].get("subjects", [])]

    analyses: List[Analysis] = []
    for a in cfg.get("analyses", []):
        ao = a.get("outputs", {})
        outputs = AnalysisOutput(
            figures=ao.get("figures", []),
            tables=ao.get("tables", []),
            print=ao.get("print", []),
        )
        analyses.append(
            Analysis(
                id=a["id"],
                type=a["type"],
                inputs=a["inputs"],
                params=a.get("params", {}),
                outputs=outputs,
            )
        )

    generation_obj: Optional[Generation] = None
    if "generation" in cfg:
        tasks = [GenerationTask(**t) for t in cfg["generation"].get("tasks", [])]
        generation_obj = Generation(tasks=tasks)

    return StimlossConfig(
        version=cfg["version"],
        project=cfg["project"],
        data={"datasets": datasets, "subjects": subjects},
        analyses=analyses,
        generation=generation_obj,
    )


def save_config(obj: StimlossConfig, path: str) -> None:
    """Dump StimlossConfig back to YAML."""
    import dataclasses as dc

    def _conv(o):
        if dc.is_dataclass(o):
            return {k: _conv(v) for k, v in dc.asdict(o).items()}
        if isinstance(o, list):
            return [_conv(x) for x in o]
        if isinstance(o, dict):
            return {k: _conv(v) for k, v in o.items()}
        return o

    with open(path, "w") as f:
        yaml.safe_dump(_conv(obj), f, sort_keys=False)
