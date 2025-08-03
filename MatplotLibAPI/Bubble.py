"""Bubble chart plotting helpers."""

# Hint for Visual Code Python Interactive window
# %%
from typing import Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns
    

from MatplotLibAPI.StyleTemplate import DynamicFuncFormatter, StyleTemplate, generate_ticks, string_formatter, bmk_formatter, percent_formatter, format_func, validate_dataframe

MAX_RESULTS = 50

BUBBLE_STYLE_TEMPLATE = StyleTemplate(
    format_funcs={"label": string_formatter,
                  "x": bmk_formatter,
                  "y": percent_formatter,
                  "z": bmk_formatter},
    yscale="log"
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
        hline=False,
        vline=False,
        ax: Optional[Axes] = None):
    """Plot a bubble chart on the provided axes.

    Parameters
    ----------
    pd_df : pandas.DataFrame
        DataFrame containing the data to plot.
    label : str
        Column name for bubble labels.
    x : str
        Column name for the x-axis values.
    y : str
        Column name for the y-axis values.
    z : str
        Column name for bubble sizes.
    title : Optional[str], default None
        Title of the plot.
    style : StyleTemplate, default BUBBLE_STYLE_TEMPLATE
        Style configuration for the plot.
    max_values : int, default MAX_RESULTS
        Maximum number of rows to plot.
    center_to_mean : bool, default False
        Whether to center the x values around their mean.
    sort_by : Optional[str], default None
        Column used to sort the data before plotting.
    ascending : bool, default False
        Sort order for the data.
    hline : bool, default False
        Draw a horizontal line at ``y=0`` if ``True``.
    vline : bool, default False
        Draw a vertical line at ``x=0`` if ``True``.
    ax : Optional[Axes]
        Existing matplotlib axes to plot on.

    Returns
    -------
    Axes
        The axes containing the bubble chart.
    """

    validate_dataframe(pd_df, cols=[label, x, y, z], sort_by=sort_by)
    style.format_funcs = format_func(
        style.format_funcs, label=label, x=x, y=y, z=z)
    if not sort_by:
        sort_by = z

    plot_df = pd_df[[label, x, y, z]].sort_values(
        by=sort_by, ascending=ascending).head(max_values)
    if center_to_mean:
        x_col_mean = plot_df[x].mean()
        plot_df[x] = plot_df[x] - x_col_mean
    plot_df['quintile'] = pd.qcut(
        plot_df[z], 5, labels=False)

    # styling

    plot_df["fontsize"] = plot_df['quintile'].map(style.font_mapping)

    if not ax:
        ax = plt.gca()

    ax = sns.scatterplot(
        data=plot_df,
        x=x,
        y=y,
        size=z,
        hue='quintile',
        sizes=(100, 2000),
        legend=False,
        palette=sns.color_palette(style.palette, as_cmap=True),
        edgecolor=style.background_color,
        ax=ax)
    ax.set_facecolor(style.background_color)
    if style.xscale:
        ax.set(xscale=style.xscale)
    if style.yscale:
        ax.set(yscale=style.yscale)

    x_min = pd_df[x].min()
    x_max = pd_df[x].max()
    x_mean = pd_df[x].mean()
    ax.set_xticks(generate_ticks(x_min, x_max, num_ticks=style.x_ticks))
    ax.xaxis.grid(True, "major", linewidth=.5, color=style.font_color)
    if style.format_funcs.get("x"):
        ax.xaxis.set_major_formatter(
            DynamicFuncFormatter(style.format_funcs.get("x")))

    y_min = pd_df[y].min()
    y_max = pd_df[y].max()
    y_mean = pd_df[y].mean()
    ax.set_yticks(generate_ticks(y_min, y_max, num_ticks=style.y_ticks))


    if style.yscale == 'log':
        ax.yaxis.set_minor_locator(NullLocator())  # Disable minor ticks for log scale
    else:
        ax.minorticks_off()  # Disable minor ticks for linear scale

    ax.yaxis.grid(True, "major", linewidth=.5, color=style.font_color)
    if style.format_funcs.get("y"):
        ax.yaxis.set_major_formatter(
            DynamicFuncFormatter(style.format_funcs.get("y")))

    ax.tick_params(axis='both',
                   which='major',
                   colors=style.font_color,
                   labelsize=style.font_size)
    if vline:
        ax.vlines(x=x_mean,
                ymin=y_min,
                ymax=y_max,
                linestyle='--',
                colors=style.font_color)
    if hline:
        ax.hlines(y=y_mean,
                xmin=x_min,
                xmax=x_max,
                linestyle='--',
                colors=style.font_color)

    for index, row in plot_df.iterrows():
        x_value = row[x]
        y_value = row[y]
        s_value = str(row[label])
        if style.format_funcs.get("label"):
            s_value = style.format_funcs.get("label")(s_value)
        fs = row["fontsize"]
        ax.text(x_value,
                y_value,
                s_value,
                horizontalalignment='center',
                fontdict={'color': style.font_color, 'fontsize': fs})
    if title:
        ax.set_title(title, color=style.font_color, fontsize=style.font_size*2)
    return ax


def fplot_bubble(
        pd_df: pd.DataFrame,
        label: str,
        x: str,
        y: str,
        z: str,
        title: Optional[str] = "Test",
        style: StyleTemplate = BUBBLE_STYLE_TEMPLATE,
        max_values: int = MAX_RESULTS,
        center_to_mean: bool = False,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        hline=False,
        vline=False,
        figsize: Tuple[float, float] = (19.2, 10.8)) -> Figure:
    """Return a new figure with a bubble chart."""

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(style.background_color)
    ax = fig.add_subplot()
    ax = aplot_bubble(pd_df=pd_df,
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
                        ax=ax)
    return fig



# endregion
