

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from matplotlib.axes import Axes

from .Utils import (BUBBLE_STYLE_TEMPLATE, DynamicFuncFormatter,
                    StyleTemplate, generate_ticks)

# region Bubble


def plot_bubble(ax: Axes,
                data: pd.DataFrame,
                x_col: str,
                y_col: Union[str, List[str]],
                fig_title: Optional[str] = None,
                style: Optional[StyleTemplate] = None,
                legend: bool = False,
                z_col: str = "uniques",
                hue_col: str = "uniques_quintile",
                l_col: str = "dimension",
                normalize_x: bool = True,
                sizes: tuple = (20, 2000), **kwargs) -> Axes:

    # Clear the axis before plotting
    ax.clear()

    # Start formatting
    if fig_title is not None:
        ax.set_title(fig_title)
    if style is None:
        style = BUBBLE_STYLE_TEMPLATE
    ax.figure.set_facecolor(style.fig_background_color)
    ax.figure.set_edgecolor(style.fig_border)
    if normalize_x:
        # Step 1: Standardize the data to have mean=0 and variance=1
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[[x_col]])

        # Step 2: Find a scaling factor to confine data within [-100, 100]
        scale_factor = 100 / np.max(np.abs(scaled_data))

        # Apply scaling factor
        scaled_data *= scale_factor

        # Round to the nearest integer
        data[f"{x_col}"] = np.round(scaled_data).astype(int)

    if type(y_col) == list:
        y_col = y_col[0]

    g = sns.scatterplot(data=data,
                        x=x_col,
                        y=y_col,
                        size=z_col,
                        hue=hue_col,
                        palette=style.palette,
                        legend=legend,
                        sizes=sizes,
                        ax=ax)

    g.set(yscale="log")

    g.axes.xaxis.grid(True, "minor", linewidth=.25)
    g.axes.yaxis.grid(True, "minor", linewidth=.25)

    g.axes.axvline(x=0, linestyle='--')

    y_min = data[y_col].min()
    y_max = data[y_col].max()
    if style.y_formatter is not None:
        g.axes.yaxis.set_major_formatter(
            DynamicFuncFormatter(style.y_formatter))
        g.set_yticks(generate_ticks(y_min, y_max, num_ticks=style.y_ticks))
    else:
        ylabels = ['{:,.0f}%'.format(y) for y in g.get_yticks()*100]
        g.set_yticklabels(ylabels)

    y_mean = data[y_col].mean()

    if style.x_formatter is not None:
        x_min = data[x_col].min()
        x_max = data[x_col].max()
        g.xaxis.set_major_formatter(
            DynamicFuncFormatter(style.x_formatter))
        g.set_xticks(generate_ticks(x_min, x_max, num_ticks=style.x_ticks))

    g.axes.xaxis.grid(True, "minor", linewidth=.25)
    g.axes.yaxis.grid(True, "minor", linewidth=.25)
    g.hlines(y=y_mean, xmin=x_min, xmax=x_max,
             linestyle='--', colors=style.font_color)

    for index, row in data.iterrows():
        x = row[x_col]
        y = row[y_col[0] if type(y_col) == List else y_col]
        s = row[l_col]
        g.text(x, y, s, horizontalalignment='center',
               fontsize=style.font_size*row[hue_col], color=style.font_color)

    return ax

# endregion
