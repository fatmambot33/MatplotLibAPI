# Hint for Visual Code Python Interactive window
# %%
from typing import Optional, Tuple, List, Union, Dict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import plotly.graph_objects as go
from .Bubble import aplot_bubble, BUBBLE_STYLE_TEMPLATE
from .Table import aplot_table
from .Treemap import fplot_treemap, TREEMAP_STYLE_TEMPLATE
from .StyleTemplate import StyleTemplate, format_func, validate_dataframe


def plot_composite_bubble(
        pd_df: pd.DataFrame,
        label: str,
        x: str,
        y: str,
        z: str,
        title: Optional[str] = "Test",
        style: StyleTemplate = BUBBLE_STYLE_TEMPLATE,
        max_values: int = 50,
        center_to_mean: bool = False,
        filter_by: Optional[str] = None,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        table_rows: int = 10,
        figsize: Tuple[float, float] = (19.2, 10.8)) -> Figure:

    validate_dataframe(pd_df, cols=[label, x, y, z], sort_by=sort_by)

    if not sort_by:
        sort_by = z
    if not filter_by:
        filter_by = z
    plot_df = pd_df.sort_values(by=filter_by,
                                ascending=ascending)[[label, x, y, z]].head(max_values)
    style.format_funcs = format_func(
        style.format_funcs, label=label, x=x, y=y, z=z)
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(style.background_color)
    grid = plt.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])
    ax = fig.add_subplot(grid[0, 0:])
    ax = aplot_bubble(pd_df=plot_df,
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
        style.format_funcs[label] = style.format_funcs["label"]
    if "x" in style.format_funcs:
        style.format_funcs[x] = style.format_funcs["x"]
    if "y" in style.format_funcs:
        style.format_funcs[y] = style.format_funcs["y"]
    if "z" in style.format_funcs:
        style.format_funcs[z] = style.format_funcs["z"]

    ax2 = fig.add_subplot(grid[1, 0])
    ax2 = aplot_table(
        pd_df=plot_df,
        cols=[label, z, y, x],
        title=f"Top {table_rows}",
        ax=ax2,
        sort_by=sort_by,
        ascending=False,
        max_values=table_rows,
        style=style
    )
    ax3 = fig.add_subplot(grid[1, 1])
    ax3 = aplot_table(
        pd_df=plot_df,
        cols=[label, z, y, x],
        title=f"Last {table_rows}",
        ax=ax3,
        sort_by=sort_by,
        ascending=True,
        max_values=table_rows,
        style=style
    )
    fig.tight_layout()
    return fig


def fplot_treemaps(pd_dfs: List[pd.DataFrame],
                   path: str,
                   values: str,
                   style: StyleTemplate = TREEMAP_STYLE_TEMPLATE,
                   title: Optional[str] = None,
                   color: Optional[str] = None,
                   sort_by: Optional[str] = None,
                   ascending: bool = False,
                   max_values: int = 100) -> go.Figure:

    trms = []
    num_dimensions = len(pd_dfs)
    if num_dimensions > 0:
        fig = make_subplots(
            rows=num_dimensions,
            cols=1,
            specs=[[{'type': 'domain'}] for _ in range(num_dimensions)],
            vertical_spacing=0.02
        )
        current_row = 0
        for pd_df in pd_dfs:
            trm = fplot_treemap(pd_df=pd_df,
                                path=path,
                                values=values,
                                title=title,
                                style=style,
                                color=color,
                                sort_by=sort_by,
                                ascending=ascending,
                                max_values=max_values,
                                fig=fig)
            trms.append(trm)
            current_row += 1
