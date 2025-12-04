"""Bubble chart plotting helpers.

Provides functions to create and render bubble charts using seaborn and matplotlib,
with customizable styling via `StyleTemplate`.
"""

from typing import Dict, Optional, Tuple, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator

from .StyleTemplate import (
    BUBBLE_STYLE_TEMPLATE,
    MAX_RESULTS,
    StyleTemplate,
    bmk_formatter,
    format_func,
    generate_ticks,
    percent_formatter,
    string_formatter,
    validate_dataframe,
    DynamicFuncFormatter,
    FormatterFunc,
)


def _prepare_bubble_data(
    pd_df: pd.DataFrame,
    label: str,
    x: str,
    y: str,
    z: str,
    sort_by: Optional[str],
    ascending: bool,
    max_values: int,
    center_to_mean: bool,
    style: StyleTemplate,
) -> pd.DataFrame:
    """Prepare data for bubble chart.

    Parameters
    ----------
    pd_df : pd.DataFrame
        Input DataFrame.
    label : str
        Column name for bubble labels.
    x : str
        Column name for x-axis values.
    y : str
        Column name for y-axis values.
    z : str
        Column name for bubble sizes.
    sort_by : Optional[str]
        Column to sort by.
    ascending : bool
        Sort order.
    max_values : int
        Maximum number of bubbles to display.
    center_to_mean : bool
        Whether to center x-axis values around the mean.
    style : StyleTemplate
        Styling for the plot.

    Returns
    -------
    pd.DataFrame
        Prepared DataFrame for plotting.

    Raises
    ------
    AttributeError
        If required columns are missing from the DataFrame.
    """
    validate_dataframe(pd_df, cols=[label, x, y, z], sort_by=sort_by)
    sort_col = sort_by or z

    plot_df = (
        pd_df[[label, x, y, z]]
        .sort_values(by=[sort_col], ascending=ascending)  # type: ignore
        .head(max_values)
        .copy()
    )

    if center_to_mean:
        plot_df[x] -= plot_df[x].mean()

    plot_df["quintile"] = pd.qcut(plot_df[z], 5, labels=False, duplicates="drop")
    plot_df["fontsize"] = plot_df["quintile"].map(style.font_mapping)  # type: ignore
    return plot_df


def _setup_bubble_axes(
    ax: Axes,
    style: StyleTemplate,
    pd_df: pd.DataFrame,
    x: str,
    y: str,
    format_funcs: Optional[Dict[str, Optional[FormatterFunc]]],
) -> None:
    """Configure axes for the bubble chart.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    style : StyleTemplate
        Styling for the plot.
    pd_df : pd.DataFrame
        DataFrame used for plotting.
    x : str
        Column name for x-axis values.
    y : str
        Column name for y-axis values.
    format_funcs : Optional[Dict[str, Optional[FormatterFunc]]]
        Functions to format axis tick labels.
    """
    ax.set_facecolor(style.background_color)

    if style.xscale:
        ax.set(xscale=style.xscale)
    if style.yscale:
        ax.set(yscale=style.yscale)

    # X-axis ticks and formatting
    x_min, x_max = cast(float, pd_df[x].min()), cast(float, pd_df[x].max())
    ax.xaxis.set_ticks(generate_ticks(x_min, x_max, num_ticks=style.x_ticks))
    ax.xaxis.grid(True, "major", linewidth=0.5, color=style.font_color)
    if format_funcs and (fmt_x := format_funcs.get(x)):
        ax.xaxis.set_major_formatter(DynamicFuncFormatter(fmt_x))

    # Y-axis ticks and formatting
    y_min, y_max = cast(float, pd_df[y].min()), cast(float, pd_df[y].max())
    ax.yaxis.set_ticks(generate_ticks(y_min, y_max, num_ticks=style.y_ticks))
    if style.yscale == "log":
        ax.yaxis.set_minor_locator(NullLocator())
    else:
        ax.minorticks_off()
    ax.yaxis.grid(True, "major", linewidth=0.5, color=style.font_color)
    if format_funcs and (fmt_y := format_funcs.get(y)):
        ax.yaxis.set_major_formatter(DynamicFuncFormatter(fmt_y))

    ax.tick_params(
        axis="both", which="major", colors=style.font_color, labelsize=style.font_size
    )


def _draw_bubbles(
    ax: Axes, plot_df: pd.DataFrame, x: str, y: str, z: str, style: StyleTemplate
) -> None:
    """Draw bubbles on the axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    plot_df : pd.DataFrame
        DataFrame with data for plotting.
    x : str
        Column name for x-axis values.
    y : str
        Column name for y-axis values.
    z : str
        Column name for bubble sizes.
    style : StyleTemplate
        Styling for the plot.
    """
    sns.scatterplot(
        data=plot_df,
        x=x,
        y=y,
        size=z,
        hue="quintile",
        sizes=(100, 2000),
        legend=False,
        palette=sns.color_palette(style.palette, as_cmap=True),
        edgecolor=style.background_color,
        ax=ax,
    )


def _draw_bubble_labels(
    ax: Axes,
    plot_df: pd.DataFrame,
    label: str,
    x: str,
    y: str,
    style: StyleTemplate,
    format_funcs: Optional[Dict[str, Optional[FormatterFunc]]],
) -> None:
    """Draw labels for each bubble.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    plot_df : pd.DataFrame
        DataFrame with data for plotting.
    label : str
        Column name for bubble labels.
    x : str
        Column name for x-axis values.
    y : str
        Column name for y-axis values.
    style : StyleTemplate
        Styling for the plot.
    format_funcs : Optional[Dict[str, Optional[FormatterFunc]]]
        Functions to format bubble labels.
    """
    for _, row in plot_df.iterrows():
        x_val, y_val, label_val = row[x], row[y], str(row[label])
        if format_funcs and (fmt_label := format_funcs.get(label)):
            label_val = fmt_label(label_val, None)
        ax.text(
            cast(float, x_val),
            cast(float, y_val),
            label_val,
            ha="center",
            fontsize=row["fontsize"],
            color=style.font_color,
        )


def _draw_mean_lines(
    ax: Axes,
    plot_df: pd.DataFrame,
    x: str,
    y: str,
    hline: bool,
    vline: bool,
    style: StyleTemplate,
) -> None:
    """Draw horizontal and vertical mean lines.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    plot_df : pd.DataFrame
        DataFrame with data for plotting.
    x : str
        Column name for x-axis values.
    y : str
        Column name for y-axis values.
    hline : bool
        Whether to draw a horizontal line at the mean of y.
    vline : bool
        Whether to draw a vertical line at the mean of x.
    style : StyleTemplate
        Styling for the plot.
    """
    if vline:
        ax.axvline(
            int(cast(float, plot_df[x].mean())), linestyle="--", color=style.font_color
        )
    if hline:
        ax.axhline(
            int(cast(float, plot_df[y].mean())), linestyle="--", color=style.font_color
        )


def aplot_bubble(
    pd_df: pd.DataFrame,
    label: str,
    x: str,
    y: str,
    z: str,
    title: Optional[str] = None,
    style: StyleTemplate = BUBBLE_STYLE_TEMPLATE,
    max_values: int = MAX_RESULTS,
    center_to_mean: bool = False,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    hline: bool = False,
    vline: bool = False,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot a bubble chart onto the given axes.

    Parameters
    ----------
    pd_df : pd.DataFrame
        DataFrame containing the data to plot.
    label : str
        Column name used for labeling bubbles.
    x : str
        Column name for x-axis values.
    y : str
        Column name for y-axis values.
    z : str
        Column name for bubble sizes.
    title : str, optional
        Plot title.
    style : StyleTemplate, optional
        Plot styling options. The default is `BUBBLE_STYLE_TEMPLATE`.
    max_values : int, optional
        Max number of rows to display. The default is `MAX_RESULTS`.
    center_to_mean : bool, optional
        Whether to center x values around their mean. The default is `False`.
    sort_by : str, optional
        Column to sort by before slicing.
    ascending : bool, optional
        Sort order. The default is `False`.
    hline : bool, optional
        Whether to draw a horizontal line at the mean of y. The default is `False`.
    vline : bool, optional
        Whether to draw a vertical line at the mean of x. The default is `False`.
    ax : Axes, optional
        Existing matplotlib axes to use. If None, uses current axes.

    Returns
    -------
    Axes
        The matplotlib Axes object containing the bubble chart.

    Raises
    ------
    AttributeError
        If required columns are not in the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from MatplotLibAPI.Bubble import aplot_bubble
    >>> data = {
    ...     'country': ['A', 'B', 'C', 'D'],
    ...     'gdp_per_capita': [45000, 42000, 52000, 48000],
    ...     'life_expectancy': [81, 78, 83, 82],
    ...     'population': [10, 20, 5, 30]
    ... }
    >>> df = pd.DataFrame(data)
    >>> fig, ax = plt.subplots()
    >>> aplot_bubble(df, label='country', x='gdp_per_capita', y='life_expectancy', z='population', ax=ax)
    """
    if ax is None:
        ax = cast(Axes, plt.gca())

    plot_df = _prepare_bubble_data(
        pd_df, label, x, y, z, sort_by, ascending, max_values, center_to_mean, style
    )

    format_funcs = format_func(style.format_funcs, label=label, x=x, y=y, z=z)

    _setup_bubble_axes(ax, style, plot_df, x, y, format_funcs)

    _draw_bubbles(ax, plot_df, x, y, z, style)

    _draw_mean_lines(ax, plot_df, x, y, hline, vline, style)

    _draw_bubble_labels(ax, plot_df, label, x, y, style, format_funcs)

    if title:
        ax.set_title(title, color=style.font_color, fontsize=style.font_size * 2)

    return ax


def fplot_bubble(
    pd_df: pd.DataFrame,
    label: str,
    x: str,
    y: str,
    z: str,
    title: Optional[str] = "Bubble Chart",
    style: StyleTemplate = BUBBLE_STYLE_TEMPLATE,
    max_values: int = MAX_RESULTS,
    center_to_mean: bool = False,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    hline: bool = False,
    vline: bool = False,
    figsize: Tuple[float, float] = (19.2, 10.8),
) -> Figure:
    """Create a new matplotlib Figure with a bubble chart.

    Parameters
    ----------
    pd_df : pd.DataFrame
        DataFrame containing the data to plot.
    label : str
        Column name for bubble labels.
    x : str
        Column name for x-axis values.
    y : str
        Column name for y-axis values.
    z : str
        Column name for bubble sizes.
    title : str, optional
        Title for the chart. The default is "Bubble Chart".
    style : StyleTemplate, optional
        Plot styling. The default is `BUBBLE_STYLE_TEMPLATE`.
    max_values : int, optional
        Max number of rows to display. The default is `MAX_RESULTS`.
    center_to_mean : bool, optional
        Whether to center x around its mean. The default is `False`.
    sort_by : str, optional
        Column to sort by.
    ascending : bool, optional
        Sort order. The default is `False`.
    hline : bool, optional
        Draw horizontal line at mean y. The default is `False`.
    vline : bool, optional
        Draw vertical line at mean x. The default is `False`.
    figsize : tuple[float, float], optional
        Size of the figure. The default is (19.2, 10.8).

    Returns
    -------
    Figure
        A matplotlib Figure object containing the bubble chart.

    Raises
    ------
    AttributeError
        If required columns are not in the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> from MatplotLibAPI.Bubble import fplot_bubble
    >>> data = {
    ...     'country': ['A', 'B', 'C', 'D'],
    ...     'gdp_per_capita': [45000, 42000, 52000, 48000],
    ...     'life_expectancy': [81, 78, 83, 82],
    ...     'population': [10, 20, 5, 30]
    ... }
    >>> df = pd.DataFrame(data)
    >>> fig = fplot_bubble(df, label='country', x='gdp_per_capita', y='life_expectancy', z='population')
    """
    fig = cast(Figure, plt.figure(figsize=figsize))
    fig.patch.set_facecolor(style.background_color)
    ax = fig.add_subplot()
    aplot_bubble(
        pd_df=pd_df,
        label=label,
        x=x,
        y=y,
        z=z,
        title=title,
        style=style,
        max_values=max_values,
        center_to_mean=center_to_mean,
        sort_by=sort_by,
        ascending=ascending,
        hline=hline,
        vline=vline,
        ax=ax,
    )
    return fig
