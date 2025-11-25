"""
Replot saved analysis data with custom styling.

Usage examples:
  python scripts/replot.py --csv reports/asic_compare_v1_long.csv --plot box --x Target --y Efficiency --hue Condition --title "Efficiency vs Target" --ylabel "Efficiency [%]" --output reports/efficiency_custom.png
  python scripts/replot.py --csv reports/asic_compare_v1_long.csv --plot box --x Target --y Ploss --hue Condition --logy --ylabel "Ploss [uW]" --output reports/ploss_custom.png

Requires: pandas, seaborn, matplotlib.
"""

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _parse_figsize(s: str) -> Tuple[float, float]:
    try:
        w, h = s.split(",")
        return float(w), float(h)
    except Exception:
        raise argparse.ArgumentTypeError("figsize must be 'width,height' (e.g., 6,4)")


def replot(args):
    df = pd.read_csv(args.csv)

    plt.figure(figsize=_parse_figsize(args.figsize))
    sns.set_theme(style="whitegrid")

    if args.plot == "box":
        ax = sns.boxplot(data=df, x=args.x, y=args.y, hue=args.hue, showfliers=False, palette=args.palette)
    elif args.plot == "violin":
        ax = sns.violinplot(data=df, x=args.x, y=args.y, hue=args.hue, inner="quartile", palette=args.palette, cut=0)
    elif args.plot == "line":
        ax = sns.lineplot(data=df, x=args.x, y=args.y, hue=args.hue, marker="o", palette=args.palette)
    else:
        raise ValueError(f"Unsupported plot type: {args.plot}")

    if args.title:
        ax.set_title(args.title)
    if args.xlabel:
        ax.set_xlabel(args.xlabel)
    if args.ylabel:
        ax.set_ylabel(args.ylabel)
    if args.logy:
        ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.legend(title=args.legend_title or args.hue, bbox_to_anchor=(1.02, 1), loc="upper left") if args.hue else None
    plt.tight_layout()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Saved plot -> {args.output}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Replot saved analysis CSVs with custom styling.")
    ap.add_argument("--csv", required=True, help="Path to CSV (e.g., long or summary dataframe).")
    ap.add_argument("--plot", choices=["box", "violin", "line"], default="box")
    ap.add_argument("--x", required=True, help="Column for x-axis.")
    ap.add_argument("--y", required=True, help="Column for y-axis.")
    ap.add_argument("--hue", help="Column for hue (grouping).")
    ap.add_argument("--palette", default="colorblind", help="Seaborn palette name.")
    ap.add_argument("--logy", action="store_true", help="Use log-scale on y-axis.")
    ap.add_argument("--title", help="Plot title.")
    ap.add_argument("--xlabel", help="X-axis label.")
    ap.add_argument("--ylabel", help="Y-axis label.")
    ap.add_argument("--legend-title", help="Legend title.")
    ap.add_argument("--figsize", default="6,4", help="Figure size 'width,height' in inches.")
    ap.add_argument("--output", help="Path to save the figure (omit to show interactively).")
    args = ap.parse_args()
    replot(args)


if __name__ == "__main__":
    main()
