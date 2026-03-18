"""Box and violin plot helpers."""

from typing import Any, Optional, Tuple

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_plot import BasePlot

from .style_template import (
    DISTRIBUTION_STYLE_TEMPLATE,
    StyleTemplate,
    string_formatter,
    validate_dataframe,
)
from .utils import _get_axis

__all__ = ["DISTRIBUTION_STYLE_TEMPLATE", "aplot_box_violin", "fplot_box_violin"]


class BoxViolinPlot(BasePlot):
    """Plot box and violin distribution charts from tabular data.

    Methods
    -------
    aplot
        Plot a box or violin chart on an existing Matplotlib axes.
    fplot
        Plot a box or violin chart on a new Matplotlib figure.
    """

    def __init__(
        self,
        pd_df: pd.DataFrame,
        column: str,
        by: Optional[str] = None,
        violin: bool = False,
    ):
        cols = [column]
        if by:
            cols.append(by)
        validate_dataframe(pd_df=pd_df, cols=cols)
        super().__init__(pd_df=pd_df)
        self.column = column
        self.by = by
        self.violin = violin

    def aplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot a box or violin chart on the provided axis.

        Parameters
        ----------
        title : str, optional
            Title for the plot. The default is None.
        style : StyleTemplate, optional
            Style template for the plot. The default is DISTRIBUTION_STYLE_TEMPLATE.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, use the current axes.
        **kwargs : Any
            Additional keyword arguments reserved for compatibility.

        Returns
        -------
        Axes
            The Matplotlib axes containing the distribution chart.
        """
        plot_ax = _get_axis(ax)

        common_kwargs = {
            "data": self._obj,
            "x": self.by,
            "y": self.column,
            "palette": style.palette,
        }

        if self.violin:
            sns.violinplot(**common_kwargs, hue=self.by, legend=False, ax=plot_ax)
        else:
            sns.boxplot(**common_kwargs, hue=self.by, legend=False, ax=plot_ax)

        plot_ax.set_facecolor(style.background_color)
        plot_ax.set_ylabel(string_formatter(self.column))
        if self.by:
            plot_ax.set_xlabel(string_formatter(self.by))
        if title:
            plot_ax.set_title(title)
        return plot_ax

    def fplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
        figsize: Tuple[float, float] = (10, 6),
    ) -> Figure:
        """Plot a box or violin chart on a new figure.

        Parameters
        ----------
        title : str, optional
            Title for the plot. The default is None.
        style : StyleTemplate, optional
            Style template for the plot. The default is DISTRIBUTION_STYLE_TEMPLATE.
        figsize : tuple[float, float], optional
            Figure size. The default is (10, 6).

        Returns
        -------
        Figure
            The Matplotlib figure containing the distribution chart.
        """
        fig = Figure(
            figsize=figsize,
            facecolor=style.background_color,
            edgecolor=style.background_color,
        )
        ax = fig.add_subplot(111)
        ax.set_facecolor(style.background_color)
        self.aplot(
            column=self.column,
            by=self.by,
            violin=self.violin,
            title=title,
            style=style,
            ax=ax,
        )
        return fig


def aplot_box_violin(
    pd_df: pd.DataFrame,
    column: str,
    by: Optional[str] = None,
    violin: bool = False,
    title: Optional[str] = None,
    style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Plot box or violin charts to summarize distributions."""
    return BoxViolinPlot(
        pd_df=pd_df,
        column=column,
        by=by,
        violin=violin,
    ).aplot(
        title=title,
        style=style,
        ax=ax,
        **kwargs,
    )


def fplot_box_violin(
    pd_df: pd.DataFrame,
    column: str,
    by: Optional[str] = None,
    violin: bool = False,
    title: Optional[str] = None,
    style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
    figsize: Tuple[float, float] = (10, 6),
) -> Figure:
    """Plot box or violin charts on a new figure."""
    return BoxViolinPlot(
        pd_df=pd_df,
        column=column,
        by=by,
        violin=violin,
    ).fplot(
        title=title,
        style=style,
        figsize=figsize,
    )
