import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mticker

def plot_box(
    df,
    *,
    x,
    y,
    hue=None,
    path=None,
    title=None,
    xlabel=None,
    ylabel=None,
    palette=None,
    showfliers=False,
    figsize=None,
    grid=True,
    yscale=None,
    yformat=None,
    **kwargs,
):
    fig, ax = plt.subplots(constrained_layout=True, figsize=figsize)
    # Drop rows without the target y (and warn if nothing left)
    plot_df = df.dropna(subset=[y])
    if plot_df.empty:
        import warnings
        warnings.warn(f"plot_box: no data to plot for y='{y}' after dropping NaNs.")
        return fig, ax
    palette_to_use = None
    if hue is not None and palette is not None:
        palette_to_use = palette
        # Trim list palettes to the number of hue levels to avoid seaborn warnings
        if isinstance(palette, (list, tuple)):
            levels = df[hue].dropna().unique()
            palette_to_use = palette[: len(levels)]
    sns.boxplot(data=plot_df, x=x, y=y, hue=hue, showfliers=showfliers, palette=palette_to_use, ax=ax, **kwargs)
    if yscale:
        ax.set_yscale(yscale)
    if yformat == "%":
        ymax = plot_df[y].max() if y in plot_df else None
        xmax = 1 if ymax is not None and ymax <= 1.5 else 100
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=xmax, decimals=0))
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(axis="y", linestyle="--", alpha=0.7)
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_line(
    df,
    *,
    x,
    y,
    hue=None,
    path=None,
    title=None,
    xlabel=None,
    ylabel=None,
    palette=None,
    marker="o",
    figsize=None,
    grid=True,
    yformat=None,
    **kwargs,
):
    fig, ax = plt.subplots(constrained_layout=True, figsize=figsize)
    sns.lineplot(data=df, x=x, y=y, hue=hue, marker=marker, palette=palette, ax=ax, **kwargs)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    if yformat == "%":
        ymax = df[y].max() if y in df else None
        xmax = 1 if ymax is not None and ymax <= 1.5 else 100
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=xmax))
    if grid:
        ax.grid(axis="y", linestyle="--", alpha=0.7)
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_scatter(
    df,
    *,
    x,
    y,
    hue=None,
    style=None,
    path=None,
    title=None,
    xlabel=None,
    ylabel=None,
    palette=None,
    marker="o",
    markers=None,
    figsize=None,
    grid=True,
    xscale=None,
    yscale=None,
    yformat=None,
    alpha=0.8,
    s=40,
    xerr=None,
    yerr=None,
    xerr_lower=None,
    xerr_upper=None,
    yerr_lower=None,
    yerr_upper=None,
    **kwargs,
):
    fig, ax = plt.subplots(constrained_layout=True, figsize=figsize)
    palette_to_use = None
    hue_levels = None
    if hue is not None and palette is not None:
        palette_to_use = palette
        if isinstance(palette, (list, tuple)):
            hue_levels = df[hue].dropna().unique().tolist()
            palette_to_use = palette[: len(hue_levels)]
    scatter = sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        style=style,
        marker=marker,
        palette=palette_to_use,
        ax=ax,
        alpha=alpha,
        s=s,
        **kwargs,
    )
    # add error bars if provided
    if any(v is not None for v in [xerr, yerr, xerr_lower, xerr_upper, yerr_lower, yerr_upper]):
        # build color map for hue levels
        color_map = {}
        if hue is not None:
            if isinstance(palette_to_use, dict):
                color_map = palette_to_use
            elif isinstance(palette_to_use, (list, tuple)) and hue_levels is not None:
                color_map = {lvl: palette_to_use[i] for i, lvl in enumerate(hue_levels)}
        # iterate rows to support asymmetric errors
        for _, row in df.iterrows():
            cx = row[x]
            cy = row[y]
            xe = None
            ye = None
            if xerr is not None:
                xe = row[xerr]
            elif xerr_lower is not None or xerr_upper is not None:
                xe = [
                    row.get(xerr_lower, 0) if xerr_lower else 0,
                    row.get(xerr_upper, 0) if xerr_upper else 0,
                ]
            if yerr is not None:
                ye = row[yerr]
            elif yerr_lower is not None or yerr_upper is not None:
                ye = [
                    row.get(yerr_lower, 0) if yerr_lower else 0,
                    row.get(yerr_upper, 0) if yerr_upper else 0,
                ]
            clr = None
            if hue is not None and row[hue] in color_map:
                clr = color_map[row[hue]]
            # reshape asymmetric errors to (2,1) for single points
            if isinstance(xe, (list, tuple, np.ndarray)):
                xe = np.array(xe, dtype=float)
                if xe.ndim == 1 and xe.size == 2:
                    xe = xe.reshape(2, 1)
            if isinstance(ye, (list, tuple, np.ndarray)):
                ye = np.array(ye, dtype=float)
                if ye.ndim == 1 and ye.size == 2:
                    ye = ye.reshape(2, 1)
            ax.errorbar(cx, cy, xerr=xe, yerr=ye, fmt="none", ecolor=clr, alpha=alpha, capsize=3, linewidth=0.8)
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
    if yformat == "%":
        ymax = df[y].max() if y in df else None
        xmax = 1 if ymax is not None and ymax <= 1.5 else 100
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=xmax))
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(axis="both", linestyle="--", alpha=0.7)
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, ax
