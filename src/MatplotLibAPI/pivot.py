"""Pivot chart helpers for bar and line plots."""

from typing import Any, Optional, Tuple, cast

import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_plot import BasePlot


from .style_template import (
    FIG_SIZE,
    PIVOTBARS_STYLE_TEMPLATE,
    StyleTemplate,
    string_formatter,
    validate_dataframe,
)

__all__ = ["PIVOTBARS_STYLE_TEMPLATE", "aplot_pivoted_bars"]


def _pivot_and_sort_data(
    data: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    aggfunc: str = "sum",
    sort_by: Optional[str] = None,
    ascending: bool = False,
) -> pd.DataFrame:
    """Pivot and sort a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    index : str
        The column to use as the pivot table index.
    columns : str
        The column to use for pivot table columns.
    values : str
        The column to aggregate.
    aggfunc : str, optional
        The aggregation function, by default "sum".
    sort_by : str, optional
        The column to sort by.
    ascending : bool, optional
        The sort order, by default `False`.

    Returns
    -------
    pd.DataFrame
        A pivoted and sorted DataFrame.
    """
    pivot_df = pd.pivot_table(
        data, values=values, index=[index], columns=[columns], aggfunc=aggfunc  # type: ignore
    )  # type: ignore
    if sort_by:
        pivot_df = pivot_df.sort_values(by=sort_by, ascending=ascending)
    return pivot_df.reset_index()


class PivotBarChart(BasePlot):
    """Class for plotting bar charts from pivoted data."""

    def __init__(
        self,
        pd_df: pd.DataFrame,
        label: str,
        x: str,
        y: str,
        agg: str = "sum",
        stacked: bool = False,
    ):
        cols = [label, x, y]
        validate_dataframe(pd_df, cols=cols)
        super().__init__(pd_df=pd_df)
        self.label = label
        self.x = x
        self.y = y
        self.agg = agg
        self.stacked = stacked

    def aplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = PIVOTBARS_STYLE_TEMPLATE,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Axes:
        pivot_df = _pivot_and_sort_data(
            self._obj,
            index=self.x,
            columns=self.label,
            values=self.y,
            aggfunc=self.agg,
            sort_by=sort_by,
            ascending=ascending,
        )

        if ax is None:
            ax = cast(Axes, plt.gca())

        if pd.api.types.is_datetime64_any_dtype(pivot_df[self.x]):
            pivot_df[self.x] = pivot_df[self.x].dt.strftime("%Y-%m-%d")

        pivot_df.plot(kind="bar", x=self.x, stacked=self.stacked, ax=ax, alpha=0.7)

        ax.set_ylabel(string_formatter(self.y))
        ax.set_xlabel(string_formatter(self.x))
        if title:
            ax.set_title(title)

        ax.legend(
            fontsize=style.font_size - 2,
            title_fontsize=style.font_size + 2,
            labelcolor="linecolor",
            facecolor=style.background_color,
        )
        ax.tick_params(axis="x", rotation=90)
        return ax

    def fplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = PIVOTBARS_STYLE_TEMPLATE,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        ax: Optional[Axes] = None,
        figsize: Tuple[float, float] = FIG_SIZE,
        **kwargs: Any,
    ) -> Figure:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        fig.set_facecolor(style.background_color)
        self.aplot(
            title=title, style=style, sort_by=sort_by, ascending=ascending, ax=ax
        )
        return fig


def aplot_pivoted_bars(
    data: pd.DataFrame,
    label: str,
    x: str,
    y: str,
    agg: str = "sum",
    style: StyleTemplate = PIVOTBARS_STYLE_TEMPLATE,
    title: Optional[str] = None,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    ax: Optional[Axes] = None,
    stacked: bool = False,
    **kwargs,
) -> Axes:
    """Plot a bar chart from a pivot table.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data to plot.
    label : str
        The column to pivot into series.
    x : str
        The column for the x-axis.
    y : str
        The column for the y-values.
    agg : str, optional
        The aggregation function for the pivot. The default is "sum".
    style : StyleTemplate, optional
        The style configuration. The default is `PIVOTBARS_STYLE_TEMPLATE`.
    title : str, optional
        The plot title.
    sort_by : str, optional
        The column to sort by.
    ascending : bool, optional
        The sort order. The default is `False`.
    ax : Axes, optional
        The axes to draw on.
    stacked : bool, optional
        Whether to stack the bars. The default is `False`.

    Returns
    -------
    Axes
        The matplotlib axes with the bar chart.
    """
    return PivotBarChart(
        pd_df=data,
        label=label,
        x=x,
        y=y,
        agg=agg,
        stacked=stacked,
    ).aplot(
        title=title,
        style=style,
        sort_by=sort_by,
        ascending=ascending,
        ax=ax,
        **kwargs,
    )


def fplot_pivoted_bars(
    pd_df: pd.DataFrame,
    label: str,
    x: str,
    y: str,
    agg: str = "sum",
    style: StyleTemplate = PIVOTBARS_STYLE_TEMPLATE,
    title: Optional[str] = None,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    ax: Optional[Axes] = None,
    stacked: bool = False,
    figsize: Tuple[float, float] = FIG_SIZE,
) -> Figure:
    """Plot a bar chart from a pivot table.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data to plot.
    label : str
        The column to pivot into series.
    x : str
        The column for the x-axis.
    y : str
        The column for the y-values.
    agg : str, optional
        The aggregation function for the pivot. The default is "sum".
    style : StyleTemplate, optional
        The style configuration. The default is `PIVOTBARS_STYLE_TEMPLATE`.
    title : str, optional
        The plot title.
    sort_by : str, optional
        The column to sort by.
    ascending : bool, optional
        The sort order. The default is `False`.
    ax : Axes, optional
        The axes to draw on.
    stacked : bool, optional
        Whether to stack the bars. The default is `False`.

    Returns
    -------
    Figure
        The matplotlib figure with the bar chart.
    """
    return PivotBarChart(
        pd_df=pd_df,
        label=label,
        x=x,
        y=y,
        agg=agg,
        stacked=stacked,
    ).fplot(
        title=title,
        style=style,
        sort_by=sort_by,
        ascending=ascending,
        ax=ax,
        figsize=figsize,
    )
