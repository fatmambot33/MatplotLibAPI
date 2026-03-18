"""Pie and donut chart helpers."""

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_plot import BasePlot

from .style_template import PIE_STYLE_TEMPLATE, StyleTemplate, validate_dataframe
from .utils import _get_axis

__all__ = ["PIE_STYLE_TEMPLATE", "aplot_pie", "fplot_pie"]


class PieChart(BasePlot):
    """Plot pie and donut charts from categorical aggregates.

    Methods
    -------
    aplot
        Plot a pie or donut chart on an existing Matplotlib axes.
    fplot
        Plot a pie or donut chart on a new Matplotlib figure.
    """

    def __init__(self, pd_df: pd.DataFrame, category: str, value: str):
        validate_dataframe(pd_df, cols=[category, value])
        super().__init__(pd_df=pd_df)
        self.category = category
        self.value = value

    def aplot(
        self,
        donut: bool = False,
        title: Optional[str] = None,
        style: StyleTemplate = PIE_STYLE_TEMPLATE,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot a pie or donut chart on the provided axis.

        Parameters
        ----------
        donut : bool, optional
            If True, render a donut chart. The default is False.
        title : str, optional
            Title for the plot. The default is None.
        style : StyleTemplate, optional
            Style template for the plot. The default is PIE_STYLE_TEMPLATE.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, use the current axes.
        **kwargs : Any
            Additional keyword arguments reserved for compatibility.

        Returns
        -------
        Axes
            The Matplotlib axes containing the pie or donut chart.
        """
        labels = self._obj[self.category].astype(str).tolist()
        sizes = self._obj[self.value]
        plot_ax = _get_axis(ax)
        wedgeprops: Optional[Dict[str, Any]] = None
        if donut:
            wedgeprops = {"width": 0.3}
        pie_kwargs: Dict[str, Any] = {
            "labels": labels,
            "autopct": "%1.1f%%",
            "colors": sns.color_palette(style.palette),
            "wedgeprops": wedgeprops,
            "textprops": {"color": style.font_color, "fontsize": style.font_size},
        }
        pie_kwargs.update(kwargs)

        plot_ax.pie(sizes, **pie_kwargs)
        plot_ax.axis("equal")
        if title:
            plot_ax.set_title(title)
        return plot_ax

    def fplot(
        self,
        donut: bool = False,
        title: Optional[str] = None,
        style: StyleTemplate = PIE_STYLE_TEMPLATE,
        figsize: Tuple[float, float] = (8, 8),
    ) -> Figure:
        """Plot a pie or donut chart on a new figure.

        Parameters
        ----------
        donut : bool, optional
            If True, render a donut chart. The default is False.
        title : str, optional
            Title for the plot. The default is None.
        style : StyleTemplate, optional
            Style template for the plot. The default is PIE_STYLE_TEMPLATE.
        figsize : tuple[float, float], optional
            Figure size. The default is (8, 8).

        Returns
        -------
        Figure
            The Matplotlib figure containing the pie or donut chart.
        """
        fig = Figure(
            figsize=figsize,
            facecolor=style.background_color,
            edgecolor=style.background_color,
        )
        ax = fig.add_subplot(111)
        ax.set_facecolor(style.background_color)
        fig.set_facecolor(style.background_color)
        self.aplot(donut=donut, title=title, style=style, ax=ax)
        return fig


def aplot_pie(
    pd_df: pd.DataFrame,
    category: str,
    value: str,
    donut: bool = False,
    title: Optional[str] = None,
    style: StyleTemplate = PIE_STYLE_TEMPLATE,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Plot pie or donut charts for categorical share visualization."""
    return PieChart(pd_df=pd_df, category=category, value=value).aplot(
        donut=donut,
        title=title,
        style=style,
        ax=ax,
        **kwargs,
    )


def fplot_pie(
    pd_df: pd.DataFrame,
    category: str,
    value: str,
    donut: bool = False,
    title: Optional[str] = None,
    style: StyleTemplate = PIE_STYLE_TEMPLATE,
    figsize: Tuple[float, float] = (8, 8),
) -> Figure:
    """Plot pie or donut charts on a new figure."""
    return PieChart(pd_df=pd_df, category=category, value=value).fplot(
        donut=donut,
        title=title,
        style=style,
        figsize=figsize,
    )
