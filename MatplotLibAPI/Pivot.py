"""Pivot chart helpers for bar and line plots."""

from typing import List, Optional, Union

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.axes import Axes

from MatplotLibAPI.StyleTemplate import (
    StyleTemplate,
    DynamicFuncFormatter,
    validate_dataframe,
    generate_ticks,
    string_formatter,
    percent_formatter,
    format_func,
)

PIVOTBARS_STYLE_TEMPLATE = StyleTemplate(
    background_color="black",
    fig_border="darkgrey",
    font_color="white",
    palette="magma",
    format_funcs={"y": percent_formatter, "label": string_formatter},
)
PIVOTLINES_STYLE_TEMPLATE = StyleTemplate(
    background_color="white",
    fig_border="lightgrey",
    palette="viridis",
    format_funcs={"y": percent_formatter, "label": string_formatter},
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
    """
    Pivots and sorts a DataFrame.

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
    sort_by : Optional[str], optional
        The column to sort by, by default None.
    ascending : bool, optional
        The sort order, by default False.

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
    """
    Plot a bar chart from a pivot table.

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
        The aggregation function for the pivot, by default "sum".
    style : StyleTemplate, optional
        The style configuration, by default PIVOTBARS_STYLE_TEMPLATE.
    title : Optional[str], optional
        The plot title, by default None.
    sort_by : Optional[str], optional
        The column to sort by, by default None.
    ascending : bool, optional
        The sort order, by default False.
    ax : Optional[Axes], optional
        The axes to draw on, by default None.
    stacked : bool, optional
        Whether to stack the bars, by default False.

    Returns
    -------
    Axes
        The matplotlib axes with the bar chart.
    """
    validate_dataframe(data, cols=[label, x, y], sort_by=sort_by)
    format_funcs = format_func(style.format_funcs, label=label, x=x, y=y)

    pivot_df = _pivot_and_sort_data(
        data, index=x, columns=label, values=y, aggfunc=agg, sort_by=sort_by, ascending=ascending
    )

    if not ax:
        ax = plt.gca()

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


# def plot_pivoted_lines(
#     data: pd.DataFrame,
#     label: str,
#     x: str,
#     y: str,
#     style: StyleTemplate = PIVOTLINES_STYLE_TEMPLATE,
#     title: Optional[str] = None,
#     max_series: int = 4,
#     sort_by: Optional[str] = None,
#     ascending: bool = False,
#     ax: Optional[Axes] = None,
# ) -> Axes:
#     """
#     Plot line charts for the top elements in a series.
#
#     Parameters
#     ----------
#     data : pd.DataFrame
#         The source data.
#     label : str
#         The column to group lines by.
#     x : str
#         The column for the x-axis values.
#     y : str
#         The column for the y-axis values.
#     style : StyleTemplate, optional
#         The style configuration, by default PIVOTLINES_STYLE_TEMPLATE.
#     title : Optional[str], optional
#         The plot title, by default None.
#     max_series : int, optional
#         The number of top elements to plot, by default 4.
#     sort_by : Optional[str], optional
#         The column to sort by, by default None.
#     ascending : bool, optional
#         The sort order, by default False.
#     ax : Optional[Axes], optional
#         The axes to draw on, by default None.
#
#     Returns
#     -------
#     Axes
#         The matplotlib axes with the line chart.
#     """
#     validate_dataframe(data, cols=[label, x, y], sort_by=sort_by)  # type: ignore
#     if not ax:
#         ax = plt.gca()
#
#     if title:
#         ax.set_title(title)
#
#     ax.figure.set_facecolor(style.background_color)
#     ax.figure.set_edgecolor(style.fig_border)
#
#     top_elements = data.groupby(label)[y].sum().nlargest(max_series).index
#     top_elements_df = data[data[label].isin(top_elements)]  # type: ignore
#
#     for element in top_elements:
#         subset = top_elements_df[top_elements_df[label] == element]
#         ax.plot(subset[x], subset[y], label=element)
#
#     if pd.api.types.is_datetime64_any_dtype(data[x]):
#         ax.xaxis.set_major_locator(mdates.MonthLocator())
#         ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
#     else:
#         if style.format_funcs:
#             x_formatter = style.format_funcs.get("x")
#             if x_formatter:
#                 ax.xaxis.set_major_formatter(DynamicFuncFormatter(x_formatter))
#
#     plt.setp(ax.get_xticklabels(), rotation=45)
#     ax.set_xlabel(string_formatter(x))
#     ax.set_ylabel(string_formatter(y))
#
#     if style.format_funcs:
#         y_formatter = style.format_funcs.get("y")
#         if y_formatter:
#             ax.yaxis.set_major_formatter(DynamicFuncFormatter(y_formatter))
#
#     ax.legend()
#     ax.grid(True)
#
#     return ax
