"""Pie and donut chart helpers."""

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.extensions import register_dataframe_accessor
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_plot import BasePlot

from .style_template import PIE_STYLE_TEMPLATE, StyleTemplate, validate_dataframe
from .utils import _get_axis, _wrap_aplot

__all__ = ["PIE_STYLE_TEMPLATE", "aplot_pie_donut", "fplot_pie_donut"]


@register_dataframe_accessor("pie")
class PieChart(BasePlot):
    """Class for plotting pie and donut charts."""

    def __init__(self, pd_df: pd.DataFrame, category: str, value: str):
        super().__init__(pd_df=pd_df)
        self.category = category
        self.value = value
        validate_dataframe(self._obj, cols=[self.category, self.value])

    def aplot(
        self,
        donut: bool = False,
        title: Optional[str] = None,
        style: StyleTemplate = PIE_STYLE_TEMPLATE,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Axes:
        labels = self._obj[self.category].astype(str).tolist()
        sizes = self._obj[self.value]
        plot_ax = _get_axis(ax)
        wedgeprops: Optional[Dict[str, Any]] = None
        if donut:
            wedgeprops = {"width": 0.3}
        plot_ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            colors=sns.color_palette(style.palette),
            wedgeprops=wedgeprops,
            textprops={"color": style.font_color, "fontsize": style.font_size},
        )
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
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(style.background_color)
        self.aplot(donut=donut, title=title, style=style, ax=ax)
        return fig


def aplot_pie_donut(
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
    return PieChart(pd_df=pd_df, category=category, value=value).fplot(
        donut=donut,
        title=title,
        style=style,
        figsize=figsize,
    )
