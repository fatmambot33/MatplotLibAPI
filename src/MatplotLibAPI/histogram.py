"""Histogram and KDE plotting helpers."""

from typing import Any, Dict, Optional, Tuple

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .style_template import (
    DISTRIBUTION_STYLE_TEMPLATE,
    StyleTemplate,
    string_formatter,
    validate_dataframe,
)
from .utils import _get_axis, _wrap_aplot

__all__ = ["DISTRIBUTION_STYLE_TEMPLATE", "aplot_histogram_kde", "fplot_histogram_kde"]


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
    """Plot a histogram with an optional kernel density estimate.

    Parameters
    ----------
    pd_df : pd.DataFrame
        The input DataFrame containing the data to plot.
    column : str
        The name of the column to plot.
    bins : int, optional
        The number of bins for the histogram, by default 20.
    kde : bool, optional
        Whether to include a kernel density estimate, by default True.
    title : Optional[str], optional
        The title of the plot, by default None.
    style : StyleTemplate, optional
        The style template to use for the plot, by default DISTRIBUTION_STYLE_TEMPLATE.
    ax : Optional[Axes], optional
        An optional matplotlib Axes to plot on. If None, a new figure and axes will be created, by default None.
    **kwargs : Any
        Additional keyword arguments to pass to seaborn.histplot.
    """
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
    """Plot a histogram with optional KDE on a new figure.

    Parameters
    ----------
    pd_df : pd.DataFrame
        The input DataFrame containing the data to plot.
    column : str
        The name of the column to plot.
    bins : int, optional
        The number of bins for the histogram, by default 20.
    kde : bool, optional
        Whether to include a kernel density estimate, by default True.
    title : Optional[str], optional
        The title of the plot, by default None.
    style : StyleTemplate, optional
        The style template to use for the plot, by default DISTRIBUTION_STYLE_TEMPLATE.
    figsize : Tuple[float, float], optional
        The size of the figure, by default (10, 6).
    """
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
