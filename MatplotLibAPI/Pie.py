"""Pie and donut chart helpers."""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .StyleTemplate import PIE_STYLE_TEMPLATE, StyleTemplate, validate_dataframe
from ._visualization_utils import _get_axis, _wrap_aplot


def aplot_pie_donut(
    pd_df: pd.DataFrame,
    category: str,
    value: str,
    donut: bool = False,
    title: Optional[str] = None,
    style: StyleTemplate = PIE_STYLE_TEMPLATE,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot pie or donut charts for categorical share visualization."""

    validate_dataframe(pd_df, cols=[category, value])
    plot_ax = _get_axis(ax)
    labels = pd_df[category]
    sizes = pd_df[value]

    wedges, *_ = plot_ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=sns.color_palette(style.palette),
    )
    if donut:
        centre_circle = plt.Circle((0, 0), 0.70, fc=style.background_color)
        plot_ax.add_artist(centre_circle)
    plot_ax.axis("equal")
    if title:
        plot_ax.set_title(title)
    return plot_ax


def fplot_pie_donut(
    pd_df: pd.DataFrame,
    category: str,
    value: str,
    donut: bool = False,
    title: Optional[str] = None,
    style: StyleTemplate = PIE_STYLE_TEMPLATE,
    figsize: Tuple[float, float] = (8, 8),
) -> Figure:
    """Plot pie or donut charts on a new figure."""

    return _wrap_aplot(
        aplot_pie_donut,
        pd_df=pd_df,
        figsize=figsize,
        category=category,
        value=value,
        donut=donut,
        title=title,
        style=style,
    )
