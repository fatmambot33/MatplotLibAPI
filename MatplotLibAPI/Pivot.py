"""Pivot chart helpers for bar and line plots."""

from typing import Optional, cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from .StyleTemplate import (
    PIVOTBARS_STYLE_TEMPLATE,
    PIVOTLINES_STYLE_TEMPLATE,
    StyleTemplate,
    format_func,
    string_formatter,
    validate_dataframe,
)


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
        data, values=values, index=[index], columns=[columns], aggfunc=aggfunc
    )
    if sort_by:
        pivot_df = pivot_df.sort_values(by=sort_by, ascending=ascending)
    return pivot_df.reset_index()


def plot_pivoted_bars(
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
    validate_dataframe(data, cols=[label, x, y], sort_by=sort_by)

    pivot_df = _pivot_and_sort_data(
        data,
        index=x,
        columns=label,
        values=y,
        aggfunc=agg,
        sort_by=sort_by,
        ascending=ascending,
    )

    if ax is None:
        ax = cast(Axes, plt.gca())

    pivot_df.plot(kind="bar", x=x, stacked=stacked, ax=ax, alpha=0.7)

    ax.set_ylabel(string_formatter(y))
    ax.set_xlabel(string_formatter(x))
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
