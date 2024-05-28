# Hint for Visual Code Python Interactive window
# %%
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from .Utils import (TIMESERIE_STYLE_TEMPLATE,
                    DynamicFuncFormatter, StyleTemplate, string_formatter, _validate_panda)
from typing import Optional

# region Line


def plot_line(pd_df: pd.DataFrame,
              label: str,
              x: str,
              y: str,
              title: Optional[str] = None,
              style: StyleTemplate = TIMESERIE_STYLE_TEMPLATE,
              sort_by: Optional[str] = None,
              ascending: bool = False,
              ax: Optional[Axes] = None) -> Axes:
    columns = [label, x, y]
    if sort_by:
        columns.append(sort_by)
    columns = list(set(columns))
    _validate_panda(pd_df, columns)

    df = pd_df[[label, x, y]].sort_values(by=[label, x])
    df[x] = pd.to_datetime(df[x])
    df.set_index(x, inplace=True)

    sns.set_palette(style.palette)
    if ax is None:
        ax = plt.gca()

    # Get unique dimension_types
    label_types = df[label].unique()

    for label_type in label_types:
        temp_df = df[df[label] == label_type]
        temp_df = temp_df.sort_values(by=x)
        if style.format_funcs.get("label"):
            label = style.format_funcs.get("label")(label_type)
        ax.plot(temp_df.index,
                temp_df[y],
                linestyle='-',
                label=label_type)

    ax.legend(
        fontsize=style.font_size-2,
        title_fontsize=style.font_size+2,
        labelcolor='linecolor',
        facecolor=style.background_color)

    ax.set_xlabel(string_formatter(x), color=style.font_color)
    if style.format_funcs.get("x"):
        ax.xaxis.set_major_formatter(
            DynamicFuncFormatter(style.format_funcs.get("x")))
    ax.tick_params(axis='x', colors=style.font_color,
                   labelrotation=45, labelsize=style.font_size-4)

    ax.set_ylabel(string_formatter(y), color=style.font_color)
    if style.format_funcs.get("y"):
        ax.yaxis.set_major_formatter(
            DynamicFuncFormatter(style.format_funcs.get("y")))
    ax.tick_params(axis='y', colors=style.font_color,
                   labelsize=style.font_size-4)
    ax.set_facecolor(style.background_color)
    ax.grid(True)
    if title:
        ax.set_title(title, color=style.font_color, fontsize=style.font_size*2)
    return ax


# endregion
