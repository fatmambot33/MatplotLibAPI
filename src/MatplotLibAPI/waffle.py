"""Waffle chart helpers."""

from typing import Any, Dict, Optional, Tuple, cast

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .base_plot import BasePlot

from .style_template import PIE_STYLE_TEMPLATE, StyleTemplate, validate_dataframe
from .utils import _get_axis, _wrap_aplot


class WaffleChart(BasePlot):
    """Class for plotting waffle charts."""

    def __init__(self, pd_df: pd.DataFrame, category: str, value: str):
        validate_dataframe(self._obj, cols=[self.category, self.value])
        super().__init__(pd_df=pd_df)
        self.category = category
        self.value = value

    def aplot(
        self,
        rows: int = 10,
        title: Optional[str] = None,
        style: StyleTemplate = PIE_STYLE_TEMPLATE,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Axes:
        value_series = cast(pd.Series, self._obj[self.value])
        category_series = cast(pd.Series, self._obj[self.category])
        total = float(value_series.sum())
        squares = rows * rows
        colors = sns.color_palette(style.palette, n_colors=len(self._obj))
        plot_ax = _get_axis(ax)
        plot_ax.set_aspect("equal")

        start = 0
        for idx, (label, val) in enumerate(zip(category_series, value_series)):
            count = int(round((val / total) * squares))
            for square in range(start, min(start + count, squares)):
                row = square // rows
                col = square % rows
                plot_ax.add_patch(
                    Rectangle(
                        (col, rows - row),
                        1,
                        1,
                        facecolor=colors[idx],
                        edgecolor=style.background_color,
                    )
                )
            start += count

        plot_ax.set_xlim(0, rows)
        plot_ax.set_ylim(0, rows + 1)
        plot_ax.axis("off")
        if title:
            plot_ax.set_title(title)
        legend_handles = [Rectangle((0, 0), 1, 1, color=color) for color in colors]
        plot_ax.legend(
            legend_handles, category_series, loc="upper center", ncol=3, frameon=False
        )
        return plot_ax

    def fplot(
        self,
        rows: int = 10,
        title: Optional[str] = None,
        style: StyleTemplate = PIE_STYLE_TEMPLATE,
        figsize: Tuple[float, float] = (8, 8),
    ) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        self.aplot(
            title=title,
            style=style,
            rows=rows,
            ax=ax,
        )
        return fig


def aplot_waffle(
    pd_df: pd.DataFrame,
    category: str,
    value: str,
    rows: int = 10,
    title: Optional[str] = None,
    style: StyleTemplate = PIE_STYLE_TEMPLATE,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Plot a waffle chart on the provided axis."""
    return WaffleChart(pd_df=pd_df, category=category, value=value).aplot(
        title=title,
        style=style,
        rows=rows,
        ax=ax,
    )


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
    return WaffleChart(pd_df=pd_df, category=category, value=value).fplot(
        title=title,
        style=style,
        rows=rows,
        figsize=figsize,
    )
