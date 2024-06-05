# Hint for Visual Code Python Interactive window
# %%
from typing import Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns

from .StyleTemplate import DynamicFuncFormatter, StyleTemplate, generate_ticks, string_formatter, bmk_formatter, percent_formatter, format_func, validate_dataframe

BUBBLE_STYLE_TEMPLATE = StyleTemplate(
    format_funcs={"label": string_formatter,
                  "x": bmk_formatter,
                  "y": percent_formatter,
                  "label": string_formatter,
                  "z": bmk_formatter},
    yscale="log"
)


def aplot_bubble(
        pd_df: pd.DataFrame,
        label: str,
        x: str,
        y: str,
        z: str,
        title: Optional[str] = "Test",
        style: StyleTemplate = BUBBLE_STYLE_TEMPLATE,
        max_values: int = BUBBLE_STYLE_TEMPLATE,
        center_to_mean: bool = False,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        hline=False,
        vline=False,
        ax: Optional[Axes] = None):

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
        max_values: int = BUBBLE_STYLE_TEMPLATE,
        center_to_mean: bool = False,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        hline=False,
        vline=False,
        figsize: Tuple[float, float] = (19.2, 10.8)) -> Figure:

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
