"""Composite plotting routines combining multiple charts."""

from typing import Optional, Tuple, List, Dict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from MatplotLibAPI.Bubble import aplot_bubble, BUBBLE_STYLE_TEMPLATE
from MatplotLibAPI.Table import aplot_table
from MatplotLibAPI.Treemap import aplot_treemap, TREEMAP_STYLE_TEMPLATE
from MatplotLibAPI.StyleTemplate import StyleTemplate, format_func, validate_dataframe


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
    """Plot a composite bubble chart with summary tables.

    Args:
        pd_df (pd.DataFrame): Data to be plotted.
        label (str): Column name for bubble labels.
        x (str): Column name for the x-axis values.
        y (str): Column name for the y-axis values.
        z (str): Column name for bubble sizes.
        title (Optional[str], optional): Title of the plot. Defaults to ``"Test"``.
        style (StyleTemplate, optional): Style configuration. Defaults to ``BUBBLE_STYLE_TEMPLATE``.
        max_values (int, optional): Maximum number of rows to display in the chart. Defaults to ``50``.
        center_to_mean (bool, optional): Whether to center the bubbles on the mean. Defaults to ``False``.
        filter_by (Optional[str], optional): Column used to filter the data. Defaults to ``None``.
        sort_by (Optional[str], optional): Column used to sort the data. Defaults to ``None``.
        ascending (bool, optional): Sort order for the data. Defaults to ``False``.
        table_rows (int, optional): Number of rows to display in the tables. Defaults to ``10``.
        figsize (Tuple[float, float], optional): Size of the created figure. Defaults to ``(19.2, 10.8)``.

    Returns:
        Figure: Matplotlib figure containing the composite bubble chart and tables.
    """
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


def plot_composite_treemap(pd_dfs: Dict[str, pd.DataFrame],
                           values: str,
                           style: StyleTemplate = TREEMAP_STYLE_TEMPLATE,
                           title: Optional[str] = None,
                           color: Optional[str] = None,
                           sort_by: Optional[str] = None,
                           ascending: bool = False,
                           max_values: int = 100) -> Optional[go.Figure]:
    """Plot a composite treemap from multiple DataFrames.

    Args:
        pd_dfs (Dict[str, pd.DataFrame]): Mapping of dimension names to DataFrames to plot.
        values (str): Column containing the values to visualize in each treemap.
        style (StyleTemplate, optional): Style configuration. Defaults to ``TREEMAP_STYLE_TEMPLATE``.
        title (Optional[str], optional): Title of the composite plot. Defaults to ``None``.
        color (Optional[str], optional): Column name used for coloring. Defaults to ``None``.
        sort_by (Optional[str], optional): Column name used to sort values. Defaults to ``None``.
        ascending (bool, optional): Sort order for values. Defaults to ``False``.
        max_values (int, optional): Maximum number of values per treemap. Defaults to ``100``.

    Returns:
        Optional[go.Figure]: Composite treemap figure, or ``None`` if no data frames are provided.
    """
    num_dimensions = len(pd_dfs)
    if num_dimensions > 0:
        subplot_titles = [f"{title}::{dim.title()}" if title is not None else dim.title(
        ) for dim in pd_dfs.keys()]
        fig = make_subplots(
            rows=num_dimensions,
            cols=1,
            specs=[[{"type": "treemap"}
                    for _ in range(0, 1)] for _ in range(0, num_dimensions)],
            subplot_titles=subplot_titles,
            vertical_spacing=0.2
        )

        current_row = 1
        for path, df in pd_dfs.items():
            trm = aplot_treemap(pd_df=df,
                                path=path,
                                values=values,
                                style=style,
                                color=color,
                                sort_by=sort_by,
                                ascending=ascending,
                                max_values=max_values)
            fig.add_trace(trm, row=current_row, col=1)
            current_row += 1
        return fig
