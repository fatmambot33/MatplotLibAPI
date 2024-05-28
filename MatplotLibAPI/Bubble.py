

import pandas as pd
import seaborn as sns
from typing import Optional
from .Utils import (BUBBLE_STYLE_TEMPLATE, DynamicFuncFormatter,
                    StyleTemplate, generate_ticks)


def plot_bubble(
        pd_df: pd.DataFrame,
        label: str,
        x: str,
        y: str,
        z: str,
        title: Optional[str] = "Test",
        style: StyleTemplate = BUBBLE_STYLE_TEMPLATE,
        max_values: int = BUBBLE_STYLE_TEMPLATE,
        center_to_mean: bool = False):

    plot_df = pd_df[[label, x, y, z]].sort_values(
        by=z, ascending=False).head(max_values)
    if center_to_mean:
        x_col_mean = plot_df[x].mean()
        plot_df[x] = plot_df[x] - x_col_mean
    plot_df['quintile'] = pd.qcut(
        plot_df[z], 5, labels=False)

    # styling

    plot_df["fontsize"] = plot_df['quintile'].map(style.font_mapping)

    ax = sns.scatterplot(
        data=plot_df,
        x=x,
        y=y,
        size=z,
        hue='quintile',
        sizes=(100, 2000),
        legend=False,
        palette=sns.color_palette(style.palette, as_cmap=True),
        edgecolor=style.background_color)
    ax.set_facecolor(style.background_color)
    if style.xscale:
        ax.set(xscale=style.xscale)
    if style.yscale:
        ax.set(yscale=style.yscale)

    x_min = pd_df[x].min()
    x_max = pd_df[x].max()
    ax.set_xticks(generate_ticks(x_min, x_max, num_ticks=style.x_ticks))
    ax.xaxis.grid(True, "major", linewidth=.5, color=style.font_color)
    if style.format_funcs.get("x"):
        ax.xaxis.set_major_formatter(
            DynamicFuncFormatter(style.format_funcs.get("x")))

    y_mean = pd_df[y].mean()
    ax.yaxis.grid(True, "major", linewidth=.5, color=style.font_color)
    if style.format_funcs.get("y"):
        ax.yaxis.set_major_formatter(
            DynamicFuncFormatter(style.format_funcs.get("y")))

    ax.tick_params(axis='both',
                   which='major',
                   colors=style.font_color,
                   labelsize=style.font_size)
    ax.hlines(y=y_mean, xmin=x_min, xmax=x_max,
              linestyle='--', colors=style.font_color)

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


# endregion
