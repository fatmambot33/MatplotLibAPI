"""Waffle chart helpers."""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .StyleTemplate import PIE_STYLE_TEMPLATE, StyleTemplate, validate_dataframe
from ._visualization_utils import _get_axis, _wrap_aplot


def aplot_waffle(
    pd_df: pd.DataFrame,
    category: str,
    value: str,
    rows: int = 10,
    title: Optional[str] = None,
    style: StyleTemplate = PIE_STYLE_TEMPLATE,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot a simple waffle chart as a grid of proportional squares."""
    validate_dataframe(pd_df, cols=[category, value])
    plot_ax = _get_axis(ax)
    total = float(pd_df[value].sum())
    squares = rows * rows
    colors = sns.color_palette(style.palette, n_colors=len(pd_df))
    plot_ax.set_aspect("equal")

    start = 0
    for idx, (label, val) in enumerate(zip(pd_df[category], pd_df[value])):
        count = int(round((val / total) * squares))
        for square in range(start, min(start + count, squares)):
            row = square // rows
            col = square % rows
            plot_ax.add_patch(
                plt.Rectangle((col, rows - row), 1, 1, facecolor=colors[idx], edgecolor=style.background_color)
            )
        start += count

    plot_ax.set_xlim(0, rows)
    plot_ax.set_ylim(0, rows + 1)
    plot_ax.axis("off")
    if title:
        plot_ax.set_title(title)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    plot_ax.legend(legend_handles, pd_df[category], loc="upper center", ncol=3, frameon=False)
    return plot_ax


def fplot_waffle(
    pd_df: pd.DataFrame,
    category: str,
    value: str,
    rows: int = 10,
    title: Optional[str] = None,
    style: StyleTemplate = PIE_STYLE_TEMPLATE,
    figsize: Tuple[float, float] = (8, 8),
) -> Figure:
    """Plot waffle charts on a new figure."""
    return _wrap_aplot(
        aplot_waffle,
        pd_df=pd_df,
        figsize=figsize,
        category=category,
        value=value,
        rows=rows,
        title=title,
        style=style,
    )
