# Hint for Visual Code Python Interactive window
# %%
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from .Bubble import plot_bubble
from .Table import plot_table
from typing import Optional, Tuple
from .Utils import (BUBBLE_STYLE_TEMPLATE, StyleTemplate, _validate_panda)


def plot_bubble_composite(
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
        table_rows: int = 10,
        figsize: Tuple[float, float] = (19.2, 10.8)) -> Figure:

    _validate_panda(pd_df, cols=[label, x, y, z], sort_by=sort_by)

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("black")
    grid = plt.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])
    ax = fig.add_subplot(grid[0, 0:])
    ax = plot_bubble(pd_df=pd_df,
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
                     ax=ax)

    if "label" in style.format_funcs:
        style.format_funcs[x] = style.format_funcs["label"]
    if "x" in style.format_funcs:
        style.format_funcs[x] = style.format_funcs["x"]
    if "y" in style.format_funcs:
        style.format_funcs[y] = style.format_funcs["y"]
    if "z" in style.format_funcs:
        style.format_funcs[z] = style.format_funcs["z"]

    ax2 = fig.add_subplot(grid[1, 0])
    ax2 = plot_table(
        pd_df=pd_df,
        cols=[label, z, x, y],
        title=f"Top {table_rows}",
        ax=ax2,
        sort_by=sort_by,
        ascending=False,
        max_values=table_rows,
        style=style
    )
    ax3 = fig.add_subplot(grid[1, 1])
    ax3 = plot_table(
        pd_df=pd_df,
        cols=[label, z, x, y],
        title=f"Worst {table_rows}",
        ax=ax3,
        sort_by=sort_by,
        ascending=True,
        max_values=table_rows,
        style=style
    )
    fig.tight_layout()
    return fig
