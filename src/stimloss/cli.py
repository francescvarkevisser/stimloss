from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import pandas as pd

import typer

from .config import (
    StimlossConfig,
    Dataset,
    Analysis,
    AnalysisOutput,
    load_config,
    save_config,
)
from .validate import validate_config
from .generator import generate_from_table, run_generation_task
from .analyses import run_analysis
from .generator import SynthConfig, _synthesize_one
from .io_utils import read_table

app = typer.Typer(help="Stimloss analysis tool")


def _datasets_index(cfg: StimlossConfig):
    return {d.id: d for d in cfg.data["datasets"]}


@app.command()
def validate(config: Path = typer.Option(..., "-c", "--config", help="Path to YAML config")):
    """Validate a config file: paths, columns, analysis references."""
    cfg = load_config(str(config))
    ok, errs = validate_config(cfg)
    if ok:
        typer.echo("OK: configuration is valid.")
    else:
        typer.echo("Errors:")
        for e in errs:
            typer.echo(f" - {e}")
        raise typer.Exit(code=1)


@app.command("add-dataset")
def add_dataset(
    config: Path = typer.Option(..., "-c", "--config", help="Config YAML to modify"),
    id: str = typer.Option(..., "--id", help="Dataset id"),
    path: Path = typer.Option(..., "--path", help="Dataset path"),
    kind: str = typer.Option(..., "--kind", help="combined | measured | sim | sim_pl"),
    format: str = typer.Option("parquet", "--format", help="parquet | csv"),
):
    """Append a dataset entry to a config."""
    cfg = load_config(str(config))
    cfg.data["datasets"].append(Dataset(id=id, path=str(path), kind=kind, format=format))
    save_config(cfg, str(config))
    typer.echo(f"Added dataset '{id}' -> {path}")


@app.command("add-analysis")
def add_analysis(
    config: Path = typer.Option(..., "-c", "--config", help="Config YAML to modify"),
    id: str = typer.Option(..., "--id", help="Analysis id"),
    type: str = typer.Option("mapping_compare", "--type", help="Analysis type"),
    combined: Optional[str] = typer.Option(None, "--combined", help="Dataset id for combined"),
    measured: Optional[str] = typer.Option(None, "--measured", help="Dataset id for measured"),
    sim_pl: Optional[str] = typer.Option(None, "--sim-pl", help="Dataset id for partial sim"),
    sim: Optional[str] = typer.Option(None, "--sim", help="Dataset id for full sim"),
    nsteps: int = typer.Option(4, "--nsteps", help="Default steps (mapping params)"),
    headroom: float = typer.Option(0.25, "--headroom", help="Default Vheadroom (mapping params)"),
):
    """Append a minimal analysis entry to a config."""
    cfg = load_config(str(config))
    outputs = AnalysisOutput(figures=[], tables=[], print=["headline_medians"])
    a = Analysis(
        id=id,
        type=type,
        inputs={"combined": combined, "measured": measured, "sim_pl": sim_pl, "sim": sim},
        params={"nsteps": nsteps, "Vheadroom": headroom},
        outputs=outputs,
    )
    cfg.analyses.append(a)
    save_config(cfg, str(config))
    typer.echo(f"Added analysis '{id}'")


@app.command()
def generate(
    config: Path = typer.Option(..., "-c", "--config", help="Config YAML"),
    task: Optional[str] = typer.Option(None, "--task", help="Only run a specific task id"),
    from_table: Optional[Path] = typer.Option(None, "--from-table", help="CSV/Excel describing datasets to synthesize"),
    sheet: Optional[str] = typer.Option(None, "--sheet", help="Excel sheet name when using --from-table"),
    n_samples: int = typer.Option(10000, "--n-samples", help="Default samples per dataset for --from-table"),
    output_dir: Path = typer.Option(Path("data/bundles"), "--output-dir", help="Output dir for --from-table"),
):
    """
    Run generation tasks. If --from-table is supplied, generate directly from a CSV/Excel table.
    Otherwise, execute tasks defined in the config.
    """
    cfg = load_config(str(config))
    if from_table:
        out = generate_from_table(str(from_table), sheet=sheet, n_samples=n_samples, output_dir=str(output_dir))
        typer.echo(f"Generated combined_df -> {out}")
        return
    if cfg.generation is None:
        typer.echo("No generation tasks in config.")
        raise typer.Exit(code=0)
    ran = False
    for t in cfg.generation.tasks:
        if task and t.id != task:
            continue
        outs = run_generation_task(t)
        typer.echo(f"[{t.id}] generated {len(outs)} file(s) -> {t.output_dir}")
        ran = True
    if not ran:
        typer.echo("No tasks matched; nothing was run.")


@app.command()
def run(
    config: Path = typer.Option(..., "-c", "--config", help="Config YAML"),
    analysis: Optional[str] = typer.Option(None, "--analysis", help="Only run a specific analysis id"),
):
    """Execute analyses declared in a config."""
    cfg = load_config(str(config))
    os.makedirs(cfg.project["output_dir"], exist_ok=True)
    ds_idx = _datasets_index(cfg)
    ran = False
    for a in cfg.analyses:
        if analysis and a.id != analysis:
            continue
        run_analysis(a, ds_idx, cfg.project["output_dir"])
        ran = True
    if not ran:
        typer.echo("No analyses matched; nothing was run.")


@app.command("new-dataset")
def new_dataset(
    method: str = typer.Option(..., "--method", help="mean_sd | median_iqr | dataset"),
    target: str = typer.Option(..., "--target", help="Target label"),
    study: Optional[str] = typer.Option(None, "--study", help="Study name"),
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Dataset name"),
    n_samples: int = typer.Option(10000, "--n-samples"),
    I_mean: Optional[float] = typer.Option(None, "--I-mean"),
    I_sd: Optional[float] = typer.Option(None, "--I-sd"),
    I_min: float = typer.Option(0.0, "--I-min"),
    I_max: Optional[float] = typer.Option(None, "--I-max"),
    I_step: float = typer.Option(5e-6, "--I-step"),
    Z_mean: Optional[float] = typer.Option(None, "--Z-mean"),
    Z_sd: Optional[float] = typer.Option(None, "--Z-sd"),
    Z_min: float = typer.Option(0.0, "--Z-min"),
    Z_max: Optional[float] = typer.Option(None, "--Z-max"),
    Z_step: float = typer.Option(1e4, "--Z-step"),
    I_median: Optional[float] = typer.Option(None, "--I-median"),
    I_iqr: Optional[float] = typer.Option(None, "--I-iqr"),
    Z_median: Optional[float] = typer.Option(None, "--Z-median"),
    Z_iqr: Optional[float] = typer.Option(None, "--Z-iqr"),
    data_path: Optional[Path] = typer.Option(None, "--data-path", help="CSV with columns I,Z when method=dataset"),
    input_combined: Path = typer.Option(..., "--input-combined", help="Existing combined parquet"),
    output_combined: Optional[Path] = typer.Option(None, "--output-combined", help="Output parquet (default overwrite input)"),
):
    """Append one synthesized dataset to an existing combined parquet."""
    cfg = SynthConfig(
        method=method,
        target=target,
        study=study,
        dataset=dataset,
        n_samples=n_samples,
        I_mean=I_mean,
        I_sd=I_sd,
        I_min=I_min,
        I_max=I_max,
        I_step=I_step,
        Z_mean=Z_mean,
        Z_sd=Z_sd,
        Z_min=Z_min,
        Z_max=Z_max,
        Z_step=Z_step,
        I_median=I_median,
        I_iqr=I_iqr,
        Z_median=Z_median,
        Z_iqr=Z_iqr,
        data_path=str(data_path) if data_path else None,
    )
    new_df = _synthesize_one(cfg)
    combined = pd.read_parquet(input_combined)
    combined = pd.concat([combined, new_df], ignore_index=True)
    out_path = output_combined or input_combined
    combined.to_parquet(out_path, index=False)
    typer.echo(f"Appended {len(new_df)} rows -> {out_path}")


@app.command("list-ids")
def list_ids(
    path: Path = typer.Option(..., "--path", "-p", help="Path to CSV/Parquet file"),
    fmt: Optional[str] = typer.Option(None, "--format", help="File format (parquet|csv). If omitted, inferred from extension."),
    column: str = typer.Option("id", "--column", "-c", help="Column name to list unique values from."),
):
    """
    Print unique values from a column (default: 'id') in a CSV/Parquet dataframe.
    """
    if fmt is None:
        fmt = path.suffix.lstrip(".")
    df = read_table(str(path), fmt)
    if column not in df.columns:
        raise typer.BadParameter(f"Column '{column}' not found in {path}. Available: {list(df.columns)}")
    uniques = sorted(df[column].dropna().astype(str).unique())
    for val in uniques:
        typer.echo(val)


def main():
    app()


if __name__ == "__main__":
    main()
