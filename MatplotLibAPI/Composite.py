"""Composite plotting routines combining multiple charts."""

from typing import Dict, Iterable, Optional, Tuple, cast

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from plotly.subplots import make_subplots

from .Bubble import BUBBLE_STYLE_TEMPLATE, FIG_SIZE, aplot_bubble
from .Network import aplot_network
from .StyleTemplate import (
    MAX_RESULTS,
    TITLE_SCALE_FACTOR,
    StyleTemplate,
    validate_dataframe,
)
from .Table import aplot_table
from .Treemap import TREEMAP_STYLE_TEMPLATE, aplot_treemap
from .Wordcloud import WORDCLOUD_STYLE_TEMPLATE, aplot_wordcloud


def plot_composite_bubble(
    pd_df: pd.DataFrame,
    label: str,
    x: str,
    y: str,
    z: str,
    title: Optional[str] = None,
    style: StyleTemplate = BUBBLE_STYLE_TEMPLATE,
    max_values: int = 50,
    center_to_mean: bool = False,
    filter_by: Optional[str] = None,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    table_rows: int = 10,
    figsize: Tuple[float, float] = FIG_SIZE,
) -> Figure:
    """Plot a composite bubble chart with summary tables.

    Parameters
    ----------
    pd_df : pd.DataFrame
        Data to be plotted.
    label : str
        Column name for bubble labels.
    x : str
        Column name for the x-axis values.
    y : str
        Column name for the y-axis values.
    z : str
        Column name for bubble sizes.
    title : str, optional
        Title of the plot. The default is ``None``.
    style : StyleTemplate, optional
        Style configuration. The default is `BUBBLE_STYLE_TEMPLATE`.
    max_values : int, optional
        Maximum number of rows to display in the chart. The default is 50.
    center_to_mean : bool, optional
        Whether to center the bubbles on the mean. The default is `False`.
    filter_by : str, optional
        Column used to filter the data.
    sort_by : str, optional
        Column used to sort the data.
    ascending : bool, optional
        Sort order for the data. The default is `False`.
    table_rows : int, optional
        Number of rows to display in the tables. The default is 10.
    figsize : tuple[float, float], optional
        Size of the created figure. The default is FIG_SIZE.

    Returns
    -------
    Figure
        Matplotlib figure containing the composite bubble chart and tables.
    """
    validate_dataframe(pd_df, cols=[label, x, y, z], sort_by=sort_by)

    if not sort_by:
        sort_by = z
    if not filter_by:
        filter_by = z
    plot_df = pd_df.sort_values(by=filter_by, ascending=ascending)[
        [label, x, y, z]
    ].head(max_values)

    fig = cast(Figure, plt.figure(figsize=figsize))
    fig.patch.set_facecolor(style.background_color)
    grid = GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])
    ax = fig.add_subplot(grid[0, 0:])
    ax = aplot_bubble(
        pd_df=cast(pd.DataFrame, plot_df),
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
        ax=ax,
    )

    ax2 = fig.add_subplot(grid[1, 0])
    ax2 = aplot_table(
        pd_df=cast(pd.DataFrame, plot_df),
        cols=[label, z, y, x],
        title=f"Top {table_rows}",
        ax=ax2,
        sort_by=sort_by,
        ascending=False,
        max_values=table_rows,
        style=style,
    )
    ax3 = fig.add_subplot(grid[1, 1])
    ax3 = aplot_table(
        pd_df=cast(pd.DataFrame, plot_df),
        cols=[label, z, y, x],
        title=f"Last {table_rows}",
        ax=ax3,
        sort_by=sort_by,
        ascending=True,
        max_values=table_rows,
        style=style,
    )
    if title:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    return fig


def plot_composite_treemap(
    pd_dfs: Dict[str, pd.DataFrame],
    values: str,
    style: StyleTemplate = TREEMAP_STYLE_TEMPLATE,
    title: Optional[str] = None,
    color: Optional[str] = None,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    max_values: int = 100,
) -> Optional[go.Figure]:
    """Plot a composite treemap from multiple DataFrames.

    Parameters
    ----------
    pd_dfs : dict[str, pd.DataFrame]
        Mapping of dimension names to DataFrames to plot.
    values : str
        Column containing the values to visualize in each treemap.
    style : StyleTemplate, optional
        Style configuration. The default is `TREEMAP_STYLE_TEMPLATE`.
    title : str, optional
        Title of the composite plot.
    color : str, optional
        Column name used for coloring.
    sort_by : str, optional
        Column name used to sort values.
    ascending : bool, optional
        Sort order for values. The default is `False`.
    max_values : int, optional
        Maximum number of values per treemap. The default is 100.

    Returns
    -------
    go.Figure, optional
        Composite treemap figure, or None if no data frames are provided.
    """
    num_dimensions = len(pd_dfs)
    if num_dimensions > 0:
        subplot_titles = [
            f"{title}::{dim.title()}" if title is not None else dim.title()
            for dim in pd_dfs.keys()
        ]
        fig = make_subplots(
            rows=num_dimensions,
            cols=1,
            specs=[
                [{"type": "treemap"} for _ in range(0, 1)]
                for _ in range(0, num_dimensions)
            ],
            subplot_titles=subplot_titles,
            vertical_spacing=0.2,
        )

        current_row = 1
        for path, df in pd_dfs.items():
            trm = aplot_treemap(
                pd_df=df,
                path=path,
                values=values,
                style=style,
                color=color,
                sort_by=sort_by,
                ascending=ascending,
                max_values=max_values,
            )
            fig.add_trace(trm, row=current_row, col=1)
            current_row += 1
        return fig
    return None


def plot_wordcloud_network(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    text_column: str = "node",
    node_weight: Optional[str] = "weight",
    source: str = "source",
    target: str = "target",
    edge_weight: str = "weight",
    max_words: int = MAX_RESULTS,
    stopwords: Optional[Iterable[str]] = None,
    title: Optional[str] = None,
    style: StyleTemplate = WORDCLOUD_STYLE_TEMPLATE,
    wordcloud_style: Optional[StyleTemplate] = None,
    network_style: Optional[StyleTemplate] = None,
    figsize: Tuple[float, float] = FIG_SIZE,
) -> Figure:
    """Plot a word cloud above a network graph.

    Parameters
    ----------
    nodes_df : pd.DataFrame
        DataFrame containing node labels and optional weights for the word cloud.
    edges_df : pd.DataFrame
        DataFrame containing edge connections for the network plot.
    text_column : str, optional
        Column in ``nodes_df`` containing the node labels. The default is ``"node"``.
    node_weight : str, optional
        Column in ``nodes_df`` containing weights for sizing words. The default is
        ``"weight"``.
    source : str, optional
        Column in ``edges_df`` containing source nodes. The default is ``"source"``.
    target : str, optional
        Column in ``edges_df`` containing target nodes. The default is ``"target"``.
    edge_weight : str, optional
        Column in ``edges_df`` containing edge weights. The default is ``"weight"``.
    max_words : int, optional
        Maximum number of words to include in the word cloud. The default is ``50``.
    stopwords : Iterable[str], optional
        Stopwords to exclude from the word cloud. The default is ``None``.
    title : str, optional
        Title for the composite figure. The default is ``None``.
    style : StyleTemplate, optional
        Shared style configuration applied to the composite figure and used for
        subplots when specialized styles are not provided. The default is
        ``WORDCLOUD_STYLE_TEMPLATE``.
    wordcloud_style : StyleTemplate, optional
        Optional style configuration for the word cloud subplot. When ``None``
        the shared ``style`` is used. The default is ``None``.
    network_style : StyleTemplate, optional
        Optional style configuration for the network subplot. When ``None`` the
        shared ``style`` is used. The default is ``None``.
    figsize : tuple[float, float], optional
        Size of the composite figure. The default is ``FIG_SIZE``.

    Returns
    -------
    Figure
        Matplotlib figure containing the word cloud on top and network below.
    """
    validate_dataframe(nodes_df, cols=[text_column], sort_by=node_weight)
    validate_dataframe(edges_df, cols=[source, target, edge_weight], sort_by=None)

    fig_raw, axes_raw = plt.subplots(
        2,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    fig = cast(Figure, fig_raw)
    wordcloud_ax, network_ax = cast(Tuple[Axes, Axes], axes_raw)

    wordcloud_style = wordcloud_style or style
    network_style = network_style or style

    fig.patch.set_facecolor(style.background_color)
    if title:
        fig.suptitle(
            title,
            color=style.font_color,
            fontsize=style.font_size * TITLE_SCALE_FACTOR,
            fontname=style.font_name,
        )

    aplot_wordcloud(
        pd_df=nodes_df,
        text_column=text_column,
        weight_column=node_weight,
        title=None,
        style=wordcloud_style,
        max_words=max_words,
        stopwords=stopwords,
        ax=wordcloud_ax,
    )

    aplot_network(
        pd_df=edges_df,
        source=source,
        target=target,
        weight=edge_weight,
        title=None,
        style=network_style,
        ax=network_ax,
    )

    if title:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    return fig
