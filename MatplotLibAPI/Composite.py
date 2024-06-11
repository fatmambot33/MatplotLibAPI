# Hint for Visual Code Python Interactive window
# %%
from typing import Optional, Tuple, List, Dict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .Bubble import aplot_bubble, BUBBLE_STYLE_TEMPLATE
from .Table import aplot_table
from .Treemap import aplot_treemap, TREEMAP_STYLE_TEMPLATE
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
    """
    Plot a composite bubble chart with additional tables for top and bottom values.

    Parameters:
    pd_df (pd.DataFrame): 
        The pandas DataFrame containing the data to be plotted.
    label (str): 
        The column name for the bubble labels.
    x (str): 
        The column name for the x-axis values.
    y (str): 
        The column name for the y-axis values.
    z (str): 
        The column name for the bubble sizes.
    title (Optional[str]): 
        The title of the plot. Default is "Test".
    style (StyleTemplate): 
        A StyleTemplate object to define the visual style of the plot. Default is BUBBLE_STYLE_TEMPLATE.
    max_values (int): 
        The maximum number of values to display in the bubble chart. Default is 50.
    center_to_mean (bool): 
        A flag indicating whether to center the bubbles to the mean. Default is False.
    filter_by (Optional[str]): 
        The column name to filter the data by. Default is None.
    sort_by (Optional[str]): 
        The column name to sort the data by. Default is None.
    ascending (bool): 
        A flag indicating whether to sort in ascending order. Default is False.
    table_rows (int): 
        The number of rows to display in the tables. Default is 10.
    figsize (Tuple[float, float]): 
        The size of the figure. Default is (19.2, 10.8).

    Returns:
    Figure: 
        The matplotlib figure object containing the composite bubble chart and tables.

    Example:
    >>> df = pd.DataFrame({
    ...     "Label": ["A", "B", "C"],
    ...     "X": [1, 2, 3],
    ...     "Y": [4, 5, 6],
    ...     "Z": [7, 8, 9]
    ... })
    >>> fig = plot_composite_bubble(df, label="Label", x="X", y="Y", z="Z")
    >>> fig.show()
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
    """
    Plot a composite treemap from multiple DataFrames.

    Parameters:
    pd_dfs (Dict[str, pd.DataFrame]): 
        A dictionary where keys are dimension names and values are pandas DataFrames to be plotted.
    values (str): 
        The column name in each DataFrame that contains the values to be visualized in the treemap.
    style (StyleTemplate): 
        A StyleTemplate object to define the visual style of the treemap. Default is TREEMAP_STYLE_TEMPLATE.
    title (Optional[str]): 
        An optional title for the composite treemap plot. Default is None.
    color (Optional[str]): 
        An optional column name for coloring the treemap. Default is None.
    sort_by (Optional[str]): 
        An optional column name for sorting the treemap values. Default is None.
    ascending (bool): 
        A flag indicating whether to sort in ascending order. Default is False.
    max_values (int): 
        The maximum number of values to display in each treemap. Default is 100.

    Returns:
    Optional[go.Figure]: 
        The composite treemap figure object or None if no DataFrames are provided.
    """
    num_dimensions = len(pd_dfs)
    if num_dimensions > 0:
        subplot_titles = [f"{title}::{dim.title()}" if title is not None else dim.title(
        ) for dim in pd_dfs.keys()]
        fig = make_subplots(
            rows=num_dimensions,
            cols=1,
            specs = [[{"type": "treemap"} for _ in range(0, 1)] for _ in range(0, num_dimensions)],
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
