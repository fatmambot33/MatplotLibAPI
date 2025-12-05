"""Histogram and KDE plotting helpers."""

from typing import Any, Optional, Tuple

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .StyleTemplate import (
    DISTRIBUTION_STYLE_TEMPLATE,
    StyleTemplate,
    string_formatter,
    validate_dataframe,
)
from ._visualization_utils import _get_axis, _wrap_aplot


def aplot_histogram_kde(
    pd_df: pd.DataFrame,
    column: str,
    bins: int = 20,
    kde: bool = True,
    title: Optional[str] = None,
    style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Plot a histogram with an optional kernel density estimate."""
    validate_dataframe(pd_df, cols=[column])
    plot_ax = _get_axis(ax)

    sns.histplot(
        data=pd_df,
        x=column,
        bins=bins,
        kde=kde,
        color=style.font_color,
        edgecolor=style.background_color,
        ax=plot_ax,
    )
    plot_ax.set_facecolor(style.background_color)
    plot_ax.set_xlabel(string_formatter(column))
    plot_ax.set_ylabel("Frequency")
    if title:
        plot_ax.set_title(title)
    return plot_ax


def fplot_histogram_kde(
    pd_df: pd.DataFrame,
    column: str,
    bins: int = 20,
    kde: bool = True,
    title: Optional[str] = None,
    style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
    figsize: Tuple[float, float] = (10, 6),
) -> Figure:
    """Plot a histogram with optional KDE on a new figure."""
    return _wrap_aplot(
        aplot_histogram_kde,
        pd_df=pd_df,
        figsize=figsize,
        column=column,
        bins=bins,
        kde=kde,
        title=title,
        style=style,
    )
