"""Bar and stacked bar chart helpers."""

from typing import Any, Dict, Optional, Tuple

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_plot import BasePlot

from .style_template import (
    DISTRIBUTION_STYLE_TEMPLATE,
    StyleTemplate,
    string_formatter,
    validate_dataframe,
)
from .utils import _get_axis, _wrap_aplot

__all__ = ["DISTRIBUTION_STYLE_TEMPLATE", "aplot_bar", "fplot_bar"]


class BarChart(BasePlot):
    """Class for plotting bar charts."""

    def __init__(
        self,
        pd_df: pd.DataFrame,
        category: str,
        value: str,
        group: Optional[str] = None,
        stacked: bool = False,
    ):
        super().__init__(pd_df=pd_df)
        self.category = category
        self.value = value
        self.group = group
        self.stacked = stacked

    def post_init(self):
        cols = [self.category, self.value]
        if self.group:
            cols.append(self.group)
        validate_dataframe(self._obj, cols=cols)

    def aplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot bar or stacked bar charts for categorical comparisons.

        Parameters
        ----------
        title : str, optional
            Title for the plot, by default None.
        style : StyleTemplate, optional
            Style template for the plot, by default DISTRIBUTION_STYLE_TEMPLATE.
        ax : Axes, optional
            Matplotlib Axes to plot on, by default None which uses the current Axes.
        **kwargs : Any
            Additional keyword arguments forwarded to the plotting function.
        
        Returns
        -------
        Axes
            The Matplotlib Axes object containing the plot.
        """
        plot_ax = _get_axis(ax)

        if self.group:
            pivot_df = self._obj.pivot_table(
                index=self.category,
                columns=self.group,
                values=self.value,
                aggfunc="sum",
            )
            pivot_df.plot(kind="bar", stacked=self.stacked, ax=plot_ax, alpha=0.85)
        else:
            sns.barplot(
                data=self._obj,
                x=self.category,
                y=self.value,
                palette=style.palette,
                ax=plot_ax,
            )

        plot_ax.set_facecolor(style.background_color)
        plot_ax.set_xlabel(string_formatter(self.category))
        plot_ax.set_ylabel(string_formatter(self.value))
        if title:
            plot_ax.set_title(title)
        plot_ax.tick_params(axis="x", labelrotation=45)
        return plot_ax

    def fplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
        figsize: Tuple[float, float] = (10, 6),
    ) -> Figure:
        """Plot bar or stacked bar charts on a new figure.
        
        Parameters
        ----------
        title : str, optional
            Title for the plot, by default None.
        style : StyleTemplate, optional
            Style template for the plot, by default DISTRIBUTION_STYLE_TEMPLATE.
        figsize : tuple[float, float], optional
            The size of the figure, by default (10, 6).
       
        Returns
        -------
        Figure
            The Matplotlib Figure object containing the plot.
        """
        return _wrap_aplot(
            self.aplot,
            pd_df=self._obj,
            figsize=figsize,
            title=title,
            style=style,
        )


def aplot_bar(
    pd_df: pd.DataFrame,
    category: str,
    value: str,
    group: Optional[str] = None,
    stacked: bool = False,
    title: Optional[str] = None,
    style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Plot bar or stacked bar charts for categorical comparisons."""
    return BarChart(
        pd_df=pd_df,
        category=category,
        value=value,
        group=group,
        stacked=stacked,
    ).aplot(
        title=title,
        style=style,
        ax=ax,
        **kwargs,
    )


def fplot_bar(
    pd_df: pd.DataFrame,
    category: str,
    value: str,
    group: Optional[str] = None,
    stacked: bool = False,
    title: Optional[str] = None,
    style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
    figsize: Tuple[float, float] = (10, 6),
) -> Figure:
    """Plot bar or stacked bar charts on a new figure."""
    return BarChart(
        pd_df=pd_df,
        category=category,
        value=value,
        group=group,
        stacked=stacked,
    ).fplot(
        title=title,
        style=style,
        figsize=figsize,
    )
