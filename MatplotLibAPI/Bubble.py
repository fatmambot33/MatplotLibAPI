"""Bubble chart plotting helpers.

Provides functions to create and render bubble charts using seaborn and matplotlib,
with customizable styling via `StyleTemplate`.
"""

from typing import Optional, Tuple, cast
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns

from MatplotLibAPI.StyleTemplate import (
    DynamicFuncFormatter,
    StyleTemplate,
    FormatterFunc,
    generate_ticks,
    string_formatter,
    bmk_formatter,
    percent_formatter,
    format_func,
    validate_dataframe
)

MAX_RESULTS = 50

BUBBLE_STYLE_TEMPLATE = StyleTemplate(
    format_funcs=cast(dict[str, Optional[FormatterFunc]], {
        "label": string_formatter,
        "x": bmk_formatter,
        "y": percent_formatter,
        "z": bmk_formatter
    }),
    yscale="log"
)


def _prepare_bubble_data(pd_df: pd.DataFrame, label: str, x: str, y: str, z: str, sort_by: Optional[str], ascending: bool, max_values: int, center_to_mean: bool, style: StyleTemplate) -> pd.DataFrame:
    """Prepare data for bubble chart."""
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

    plot_df["quintile"] = pd.qcut(plot_df[z], 5, labels=False, duplicates='drop')
    plot_df["fontsize"] = plot_df["quintile"].map(style.font_mapping)  # type: ignore
    return plot_df


def _setup_bubble_axes(ax: Axes, style: StyleTemplate, pd_df: pd.DataFrame, x: str, y: str, format_funcs: Optional[dict[str, Optional[FormatterFunc]]]) -> None:
    """Configure axes for the bubble chart."""
    ax.set_facecolor(style.background_color)

    if style.xscale:
        ax.set(xscale=style.xscale)
    if style.yscale:
        ax.set(yscale=style.yscale)

    # X-axis ticks and formatting
    x_min, x_max = cast(float, pd_df[x].min()), cast(float, pd_df[x].max())
    ax.set_xticks(generate_ticks(x_min, x_max, num_ticks=style.x_ticks))
    ax.xaxis.grid(True, "major", linewidth=0.5, color=style.font_color)
    if format_funcs and (fmt_x := format_funcs.get(x)):
        ax.xaxis.set_major_formatter(DynamicFuncFormatter(fmt_x))

    # Y-axis ticks and formatting
    y_min, y_max = cast(float, pd_df[y].min()), cast(float, pd_df[y].max())
    ax.set_yticks(generate_ticks(y_min, y_max, num_ticks=style.y_ticks))
    if style.yscale == "log":
        ax.yaxis.set_minor_locator(NullLocator())
    else:
        ax.minorticks_off()
    ax.yaxis.grid(True, "major", linewidth=0.5, color=style.font_color)
    if format_funcs and (fmt_y := format_funcs.get(y)):
        ax.yaxis.set_major_formatter(DynamicFuncFormatter(fmt_y))

    ax.tick_params(axis="both", which="major",
                   colors=style.font_color, labelsize=style.font_size)


def _draw_bubbles(ax: Axes, plot_df: pd.DataFrame, x: str, y: str, z: str, style: StyleTemplate) -> None:
    """Draw bubbles on the axes."""
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


def _draw_bubble_labels(ax: Axes, plot_df: pd.DataFrame, label: str, x: str, y: str, style: StyleTemplate, format_funcs: Optional[dict[str, Optional[FormatterFunc]]]) -> None:
    """Draw labels for each bubble."""
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


def _draw_mean_lines(ax: Axes, plot_df: pd.DataFrame, x: str, y: str, hline: bool, vline: bool, style: StyleTemplate) -> None:
    """Draw horizontal and vertical mean lines."""
    if vline:
        ax.axvline(cast(float, plot_df[x].mean()), linestyle="--", color=style.font_color)
    if hline:
        ax.axhline(cast(float, plot_df[y].mean()), linestyle="--", color=style.font_color)


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
    ax: Optional[Axes] = None
) -> Axes:
    """Plot a bubble chart onto the given axes.

    Args:
        pd_df (pd.DataFrame): DataFrame containing the data to plot.
        label (str): Column name used for labeling bubbles.
        x (str): Column name for x-axis values.
        y (str): Column name for y-axis values.
        z (str): Column name for bubble sizes.
        title (Optional[str], optional): Plot title. Defaults to None.
        style (StyleTemplate, optional): Plot styling options. Defaults to BUBBLE_STYLE_TEMPLATE.
        max_values (int, optional): Max number of rows to display. Defaults to MAX_RESULTS.
        center_to_mean (bool, optional): Whether to center x values around their mean. Defaults to False.
        sort_by (Optional[str], optional): Column to sort by before slicing. Defaults to None.
        ascending (bool, optional): Sort order. Defaults to False.
        hline (bool, optional): Whether to draw a horizontal line at the mean of y. Defaults to False.
        vline (bool, optional): Whether to draw a vertical line at the mean of x. Defaults to False.
        ax (Optional[Axes], optional): Existing matplotlib axes to use. Defaults to current axes.

    Returns:
        Axes: The matplotlib Axes object containing the bubble chart.
    """
    if ax is None:
        ax = plt.gca()

    plot_df = _prepare_bubble_data(
        pd_df, label, x, y, z, sort_by, ascending, max_values, center_to_mean, style)

    format_funcs = format_func(
        style.format_funcs, label=label, x=x, y=y, z=z)

    _setup_bubble_axes(ax, style, pd_df, x, y, format_funcs)

    _draw_bubbles(ax, plot_df, x, y, z, style)

    _draw_mean_lines(ax, plot_df, x, y, hline, vline, style)

    _draw_bubble_labels(ax, plot_df, label, x, y, style, format_funcs)

    if title:
        ax.set_title(title, color=style.font_color,
                     fontsize=style.font_size * 2)

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
    figsize: Tuple[float, float] = (19.2, 10.8)
) -> Figure:
    """Create a new matplotlib Figure with a bubble chart.

    Args:
        pd_df (pd.DataFrame): DataFrame containing the data to plot.
        label (str): Column name for bubble labels.
        x (str): Column name for x-axis values.
        y (str): Column name for y-axis values.
        z (str): Column name for bubble sizes.
        title (Optional[str], optional): Title for the chart. Defaults to "Bubble Chart".
        style (StyleTemplate, optional): Plot styling. Defaults to BUBBLE_STYLE_TEMPLATE.
        max_values (int, optional): Max number of rows to display. Defaults to MAX_RESULTS.
        center_to_mean (bool, optional): Whether to center x around its mean. Defaults to False.
        sort_by (Optional[str], optional): Column to sort by. Defaults to None.
        ascending (bool, optional): Sort order. Defaults to False.
        hline (bool, optional): Draw horizontal line at mean y. Defaults to False.
        vline (bool, optional): Draw vertical line at mean x. Defaults to False.
        figsize (Tuple[float, float], optional): Size of the figure. Defaults to (19.2, 10.8).

    Returns:
        Figure: A matplotlib Figure object containing the bubble chart.
    """
    fig = plt.figure(figsize=figsize)
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
