"""Histogram and KDE plotting helpers."""

from typing import Any, Optional, Tuple

import pandas as pd
from pandas.api.extensions import register_dataframe_accessor
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


from .base_plot import BasePlot

from .style_template import (
    DISTRIBUTION_STYLE_TEMPLATE,
    NETWORK_STYLE_TEMPLATE,
    StyleTemplate,
    string_formatter,
    validate_dataframe,
)
from .utils import _get_axis

__all__ = ["DISTRIBUTION_STYLE_TEMPLATE", "aplot_histogram", "fplot_histogram"]


@register_dataframe_accessor("histogram")
class Histogram(BasePlot):
    """Class for plotting histograms with optional KDE."""

    def __init__(
        self,
        pd_df: pd.DataFrame,
        column: str,
        bins: int = 20,
        kde: bool = True,
    ):
        super().__init__(pd_df=pd_df)
        self.column = column
        self.bins = bins
        self.kde = kde

    def aplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = NETWORK_STYLE_TEMPLATE,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Axes:

        validate_dataframe(self._obj, cols=[self.column])
        plot_ax = _get_axis(ax)
        histplot_kwargs: dict[str, Any] = {
            "data": self._obj,
            "x": self.column,
            "bins": self.bins,
            "kde": self.kde,
            "color": style.font_color,
            "edgecolor": style.background_color,
            "ax": plot_ax,
        }
        histplot_kwargs.update(kwargs)
        sns.histplot(**histplot_kwargs)
        plot_ax.set_facecolor(style.background_color)
        plot_ax.set_xlabel(string_formatter(self.column))
        plot_ax.set_ylabel("Frequency")
        if title:
            plot_ax.set_title(title)
        return plot_ax

    def fplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = NETWORK_STYLE_TEMPLATE,
        figsize: Tuple[float, float] = (10, 6),
    ) -> Figure:
        fig = Figure(
            figsize=figsize,
            facecolor=style.background_color,
            edgecolor=style.background_color,
        )
        ax = fig.add_subplot(111)
        ax.set_facecolor(style.background_color)
        self.aplot(
            title=title,
            style=style,
            ax=ax,
        )
        return fig


def aplot_histogram(
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
    return Histogram(
        pd_df=pd_df,
        column=column,
        bins=bins,
        kde=kde,
    ).aplot(
        title=title,
        style=style,
        ax=ax,
        **kwargs,
    )


def fplot_histogram(
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
    return Histogram(
        pd_df=pd_df,
        column=column,
        bins=bins,
        kde=kde,
    ).fplot(
        title=title,
        style=style,
        figsize=figsize,
    )
