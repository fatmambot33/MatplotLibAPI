"""Area chart helpers for Matplotlib-based area visualizations."""

from typing import Any, Optional, Tuple

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_plot import BasePlot

from .style_template import (
    AREA_STYLE_TEMPLATE,
    StyleTemplate,
    string_formatter,
    validate_dataframe,
)

__all__ = ["AREA_STYLE_TEMPLATE", "aplot_area", "fplot_area"]


class AreaChart(BasePlot):
    """Plot area charts from tabular data.

    Methods
    -------
    aplot
        Plot an area chart on an existing Matplotlib axes.
    fplot
        Plot an area chart on a new Matplotlib figure.
    """

    def __init__(
        self,
        pd_df: pd.DataFrame,
        x: str,
        y: str,
        label: Optional[str] = None,
        stacked: bool = True,
    ):
        """Initialize an area chart plotter.

        Parameters
        ----------
        pd_df : pd.DataFrame
            DataFrame containing the data to visualize.
        x : str
            Column name used for the x-axis.
        y : str
            Column name used for the y-axis values.
        label : str, optional
            Column used to split the area into groups. The default is None.
        stacked : bool, optional
            Whether grouped areas are stacked. The default is True.
        """
        super().__init__(pd_df=pd_df)
        self.x = x
        self.y = y
        self.label = label
        self.stacked = stacked

        cols = [self.x, self.y]
        if self.label:
            cols.append(self.label)
        validate_dataframe(self._obj, cols=cols)

    def _plot_grouped_area(
        self,
        plot_ax: Axes,
        **kwargs: Any,
    ) -> None:
        """Plot grouped area data using a pivoted dataframe."""
        pivot_df = self._obj.pivot_table(
            index=self.x,
            columns=self.label,
            values=self.y,
            aggfunc="sum",
        ).sort_index()

        plot_kwargs: dict[str, Any] = {
            "kind": "area",
            "stacked": self.stacked,
            "alpha": 0.7,
            "ax": plot_ax,
        }
        pivot_df.plot(**BasePlot.merge_kwargs(plot_kwargs, kwargs))

        legend = plot_ax.get_legend()
        if legend is not None:
            legend.set_title(string_formatter(self.label or ""))

    def _plot_single_area(
        self,
        plot_ax: Axes,
        style: StyleTemplate,
        **kwargs: Any,
    ) -> None:
        """Plot a single-series area chart."""
        sorted_df = self._obj.sort_values(by=self.x)
        fill_between_kwargs: dict[str, Any] = {
            "color": style.font_color,
            "alpha": 0.4,
        }
        merged_fill_between_kwargs = BasePlot.merge_kwargs(fill_between_kwargs, kwargs)

        plot_ax.fill_between(
            sorted_df[self.x],
            sorted_df[self.y],
            **merged_fill_between_kwargs,
        )
        plot_ax.plot(sorted_df[self.x], sorted_df[self.y], color=style.font_color)

    def aplot(
        self,
        title: Optional[str] = None,
        style: Optional[StyleTemplate] = None,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot an area chart on the provided axis.

        Parameters
        ----------
        title : str, optional
            Title for the plot. The default is None.
        style : StyleTemplate, optional
            Style template for the plot. The default is AREA_STYLE_TEMPLATE.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, use the current axes.
        **kwargs : Any
            Additional keyword arguments forwarded to the area plotting call.

        Returns
        -------
        Axes
            The Matplotlib axes containing the area chart.
        """
        if not style:
            style = AREA_STYLE_TEMPLATE
        plot_ax = BasePlot.get_axis(ax)
        plot_ax.set_facecolor(style.background_color)

        if self.label:
            self._plot_grouped_area(plot_ax=plot_ax, **kwargs)
        else:
            self._plot_single_area(plot_ax=plot_ax, style=style, **kwargs)

        plot_ax.set_xlabel(string_formatter(self.x))
        plot_ax.set_ylabel(string_formatter(self.y))
        plot_ax.tick_params(axis="x", labelrotation=45)
        if title:
            plot_ax.set_title(title)
        return plot_ax


def aplot_area(
    pd_df: pd.DataFrame,
    x: str,
    y: str,
    label: Optional[str] = None,
    stacked: bool = True,
    title: Optional[str] = None,
    style: Optional[StyleTemplate] = None,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Plot area charts, optionally stacked for part-to-whole trends."""
    return AreaChart(
        pd_df=pd_df,
        x=x,
        y=y,
        label=label,
        stacked=stacked,
    ).aplot(
        title=title,
        style=style,
        ax=ax,
        **kwargs,
    )


def fplot_area(
    pd_df: pd.DataFrame,
    x: str,
    y: str,
    label: Optional[str] = None,
    stacked: bool = True,
    title: Optional[str] = None,
    style: StyleTemplate = AREA_STYLE_TEMPLATE,
    figsize: Tuple[float, float] = (10, 6),
    **kwargs: Any,
) -> Figure:
    """Plot area charts on a new figure."""
    return AreaChart(
        pd_df=pd_df,
        x=x,
        y=y,
        label=label,
        stacked=stacked,
    ).fplot(
        title=title,
        style=style,
        figsize=figsize,
        **kwargs,
    )
