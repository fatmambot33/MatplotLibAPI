"""Network chart plotting helpers."""

from typing import Any, Optional, Tuple, cast
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .core import NetworkGraph
from .constants import _DEFAULT
from ..style_template import (
    NETWORK_STYLE_TEMPLATE,
    FIG_SIZE,
    TITLE_SCALE_FACTOR,
    StyleTemplate,
    string_formatter,
    validate_dataframe,
)

__all__ = [
    "trim_low_degree_nodes",
    "aplot_network",
    "aplot_network_node",
    "aplot_network_components",
    "fplot_network",
    "fplot_network_node",
    "fplot_network_components",
]


def trim_low_degree_nodes(
    pandas_df: pd.DataFrame,
    source: str = "source",
    target: str = "target",
    min_degree: int = 2,
    recursive: bool = True,
) -> pd.DataFrame:
    """Return an edge list filtered by minimum undirected node degree.

    The function preserves all original columns and only filters rows (edges).
    Degree is computed from the edge list as an undirected multigraph:
    duplicate edges contribute repeatedly, and self-loops contribute ``2`` to
    the node degree (once from ``source`` and once from ``target``).

    Parameters
    ----------
    pandas_df : pd.DataFrame
        Edge list DataFrame.
    source : str, optional
        Source node column name. The default is ``"source"``.
    target : str, optional
        Target node column name. The default is ``"target"``.
    min_degree : int, optional
        Minimum undirected degree required for each endpoint. The default is ``2``.
        Use ``0`` to disable filtering.
    recursive : bool, optional
        Whether to recursively prune until stable. The default is ``True``.
        If ``False``, the filter is applied in a single pass using degrees
        from the original edge list.

    Returns
    -------
    pd.DataFrame
        Filtered copy of ``pandas_df``.

    Raises
    ------
    AttributeError
        If required columns are missing.
    ValueError
        If ``min_degree`` is negative.

    Examples
    --------
    >>> trim_low_degree_nodes(df, source="src", target="dst", min_degree=2)
    """
    validate_dataframe(pandas_df, cols=[source, target])
    if min_degree < 0:
        raise ValueError("min_degree must be greater than or equal to 0.")

    result = pandas_df.copy()
    if result.empty or min_degree == 0:
        return result

    def _filter_once(edges_df: pd.DataFrame) -> pd.DataFrame:
        valid_endpoints = edges_df[source].notna() & edges_df[target].notna()
        valid_edges = edges_df.loc[valid_endpoints]
        if valid_edges.empty:
            return valid_edges.copy()

        degree = cast(
            pd.Series,
            pd.concat(
                [valid_edges[source], valid_edges[target]], ignore_index=True
            ).value_counts(),
        )
        keep_nodes = {
            node for node, node_degree in degree.items() if node_degree >= min_degree
        }
        keep_mask = valid_edges[source].isin(keep_nodes) & valid_edges[target].isin(
            keep_nodes
        )
        return valid_edges.loc[keep_mask].copy()

    if not recursive:
        return _filter_once(result)

    while True:
        filtered = _filter_once(result)
        if len(filtered) == len(result):
            return filtered
        result = filtered


def _sanitize_node_dataframe(
    node_df: Optional[pd.DataFrame],
    edge_df: pd.DataFrame,
    node_col: str = "node",
    node_weight_col: str = "weight",
    edge_source_col: str = "source",
    edge_target_col: str = "target",
    edge_weight_col: str = "weight",
) -> Optional[pd.DataFrame]:
    """Private helper returning ``node_df`` rows present in the edge list.

    Intended for internal use when preparing plotting data.

    Parameters
    ----------
    node_df : pd.DataFrame, optional
        DataFrame containing ``node`` and ``weight`` columns.
    edge_df : pd.DataFrame
        Edge DataFrame containing source and target columns.
    node_col : str, optional
        Column name for node identifiers. The default is "node".
    node_weight_col : str, optional
        Column name for node weights. The default is "weight".
    edge_source_col : str, optional
        Column name for source edges. The default is "source".
    edge_target_col : str, optional
        Column name for target edges. The default is "target".
    edge_weight_col : str, optional
        Column name for edge weights. The default is "weight". Included to
        keep signature parity with other sanitization helpers.

    Returns
    -------
    pd.DataFrame
        Filtered ``node_df`` with only nodes that appear as sources or targets.
    """
    if node_df is None:
        return None

    validate_dataframe(node_df, cols=[node_col, node_weight_col])
    validate_dataframe(
        edge_df, cols=[edge_source_col, edge_target_col, edge_weight_col]
    )
    filtered_node_df = node_df.copy()
    nodes_in_edges = list(set(edge_df[edge_source_col]).union(edge_df[edge_target_col]))
    return filtered_node_df.loc[filtered_node_df[node_col].isin(nodes_in_edges)]


def _prepare_network_graph(
    pd_df: pd.DataFrame,
    node_col: str = "node",
    node_weight_col: str = "weight",
    edge_source_col: str = "source",
    edge_target_col: str = "target",
    edge_weight_col: str = "weight",
    sort_by: Optional[str] = None,
    node_df: Optional[pd.DataFrame] = None,
) -> NetworkGraph:
    """Prepare a NetworkGraph for plotting from a pandas DataFrame.

    This function takes a DataFrame and prepares it for network visualization by:
    1. Filtering the DataFrame to include only the nodes in ``node_df`` (if provided).
    2. Validating the DataFrame to ensure it has the required columns.
    3. Creating a `NetworkGraph` from the edge list.
    4. Extracting the k-core of the graph (k=2) to focus on the main structure.
    5. Applying node weights provided in ``node_df`` or calculating them from
       the top-k edge weights.
    6. Trimming the graph to keep only the top k edges per node.

    Parameters
    ----------
    pd_df : pd.DataFrame
        DataFrame containing the edge list.
    node_col : str, optional
        Column name for node identifiers. The default is "node".
    node_weight_col : str, optional
        Column name for node weights. The default is "weight".
    edge_source_col : str, optional
        Column name for source nodes. The default is "source".
    edge_target_col : str, optional
        Column name for target nodes. The default is "target".
    edge_weight_col : str, optional
        Column name for edge weights. The default is "weight".
    sort_by : str, optional
        Column to sort the DataFrame by before processing.
    node_df : pd.DataFrame, optional
        DataFrame containing ``node`` and ``weight`` columns. If provided, the
        DataFrame will be filtered to include only edges connected to these
        nodes, and their provided weights will be used instead of calculated
        values.

    Returns
    -------
    NetworkGraph
        The prepared `NetworkGraph` object.

    Raises
    ------
    ValueError
        If ``node_df`` is provided but none of its nodes appear as sources or
        targets in ``pd_df``.
    """
    filtered_node_df = _sanitize_node_dataframe(
        node_df,
        pd_df,
        node_col=node_col,
        node_weight_col=node_weight_col,
        edge_source_col=edge_source_col,
        edge_target_col=edge_target_col,
        edge_weight_col=edge_weight_col,
    )
    if node_df is not None:
        if filtered_node_df is None or filtered_node_df.empty:
            raise ValueError(
                "node_df must include at least one node present as a source or target."
            )
        allowed_nodes = filtered_node_df[node_col].tolist()
        df = pd_df.loc[
            pd_df[edge_source_col].isin(allowed_nodes)
            & pd_df[edge_target_col].isin(allowed_nodes)
        ]
    else:
        df = pd_df
    validate_dataframe(
        df, cols=[edge_source_col, edge_target_col, edge_weight_col], sort_by=sort_by
    )

    graph = NetworkGraph.from_pandas_edgelist(
        df,
        source=edge_source_col,
        target=edge_target_col,
        edge_weight_col=edge_weight_col,
    )
    if filtered_node_df is not None and not filtered_node_df.empty:
        node_weights = {
            node: weight_value
            for node, weight_value in filtered_node_df.set_index(node_col)[
                node_weight_col
            ].items()
            if node in graph._nx_graph.nodes
        }
        nx.set_node_attributes(graph._nx_graph, node_weights, name=edge_weight_col)
    else:
        graph.calculate_nodes(edge_weight_col=edge_weight_col, k=10)
    return graph


def aplot_network(
    pd_df: pd.DataFrame,
    edge_source_col: str = "source",
    edge_target_col: str = "target",
    edge_weight_col: str = "weight",
    title: Optional[str] = None,
    style: Optional[StyleTemplate] = None,
    layout_seed: Optional[int] = _DEFAULT["SPRING_LAYOUT_SEED"],
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot a network graph on the provided axes.

    Parameters
    ----------
    pd_df : pd.DataFrame
        DataFrame containing edge data.
    edge_source_col : str, optional
        Column name for source nodes. The default is "source".
    edge_target_col : str, optional
        Column name for target nodes. The default is "target".
    edge_weight_col : str, optional
        Column name for edge weights. The default is "weight".
    title : str, optional
        Plot title.
    style : StyleTemplate, optional
        Style configuration. The default is `NETWORK_STYLE_TEMPLATE`.
    layout_seed : int, optional
        Seed for the spring layout used to place nodes. The default is ``_DEFAULT["SPRING_LAYOUT_SEED"]``.
    ax : Axes, optional
        Axes to draw on.

    Returns
    -------
    Axes
        Matplotlib axes with the plotted network.
    """
    return NetworkGraph(
        pd_df=pd_df,
        source=edge_source_col,
        target=edge_target_col,
        weight=edge_weight_col,
    ).aplot(
        title=title,
        style=style,
        edge_weight_col=edge_weight_col,
        layout_seed=layout_seed,
        ax=ax,
    )


def aplot_network_node(
    pd_df: pd.DataFrame,
    node: Any,
    edge_source_col: str = "source",
    edge_target_col: str = "target",
    edge_weight_col: str = "weight",
    title: Optional[str] = None,
    style: Optional[StyleTemplate] = None,
    ax: Optional[Axes] = None,
    layout_seed: Optional[int] = _DEFAULT["SPRING_LAYOUT_SEED"],
) -> Axes:
    """Plot the connected component containing ``node`` on the provided axes.

    Parameters
    ----------
    pd_df : pd.DataFrame
        DataFrame containing edge data.
    node : Any
        Node identifier whose component should be visualized.
    node_col : str, optional
        Column name for node identifiers. The default is "node".
    node_weight_col : str, optional
        Column name for node weights. The default is "weight".
    edge_source_col : str, optional
        Column name for source nodes. The default is "source".
    edge_target_col : str, optional
        Column name for target nodes. The default is "target".
    edge_weight_col : str, optional
        Column name for edge weights. The default is "weight".
    sort_by : str, optional
        Column used to sort the data.
    ascending : bool, optional
        Sort order for the data. The default is `False`.
    node_df : pd.DataFrame, optional
        DataFrame containing ``node`` and ``weight`` columns to include.
    title : str, optional
        Plot title. If ``None``, defaults to the node identifier.
    style : StyleTemplate, optional
        Style configuration. The default is `NETWORK_STYLE_TEMPLATE`.
    ax : Axes, optional
        Axes to draw on.
    layout_seed : int, optional
        Seed for the spring layout used to place nodes. The default is ``_DEFAULT["SPRING_LAYOUT_SEED"]``.

    Returns
    -------
    Axes
        Matplotlib axes with the plotted component.

    Raises
    ------
    ValueError
        If ``node`` is not present in the prepared graph.
    """
    graph = NetworkGraph(
        pd_df=pd_df,
        source=edge_source_col,
        target=edge_target_col,
        weight=edge_weight_col,
    )
    component_graph = graph.subgraph_component(node)
    resolved_title = title if title is not None else string_formatter(node)
    return component_graph.aplot(
        title=resolved_title,
        style=style,
        edge_weight_col=edge_weight_col,
        ax=ax,
        layout_seed=layout_seed,
    )


def aplot_network_components(
    pd_df: pd.DataFrame,
    node_col: str = "node",
    node_weight_col: str = "weight",
    edge_source_col: str = "source",
    edge_target_col: str = "target",
    edge_weight_col: str = "weight",
    sort_by: Optional[str] = None,
    ascending: bool = False,
    node_df: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
    style: Optional[StyleTemplate] = None,
    layout_seed: Optional[int] = _DEFAULT["SPRING_LAYOUT_SEED"],
    axes: Optional[np.ndarray] = None,
) -> None:
    """Plot network components separately on multiple axes.

    Parameters
    ----------
    pd_df : pd.DataFrame
        DataFrame containing edge data.
    node_col : str, optional
        Column name for node identifiers. The default is "node".
    node_weight_col : str, optional
        Column name for node weights. The default is "weight".
    edge_source_col : str, optional
        Column name for source nodes. The default is "source".
    edge_target_col : str, optional
        Column name for target nodes. The default is "target".
    edge_weight_col : str, optional
        Column name for edge weights. The default is "weight".
    sort_by : str, optional
        Column used to sort the data.
    ascending : bool, optional
        Sort order for the data. The default is `False`.
    node_df : pd.DataFrame, optional
        DataFrame containing ``node`` and ``weight`` columns to include.
    title : str, optional
        Base title for subplots.
    style : StyleTemplate, optional
        Style configuration. The default is `NETWORK_STYLE_TEMPLATE`.
    layout_seed : int, optional
        Seed for the spring layout used to place nodes. The default is ``_DEFAULT["SPRING_LAYOUT_SEED"]``.
    axes : np.ndarray
        Existing axes to draw on.
    """
    graph = _prepare_network_graph(
        pd_df,
        node_col=node_col,
        node_weight_col=node_weight_col,
        edge_source_col=edge_source_col,
        edge_target_col=edge_target_col,
        edge_weight_col=edge_weight_col,
        sort_by=sort_by,
        node_df=node_df,
    )
    graph.aplot_connected_components(
        title=title,
        style=style,
        edge_weight_col=edge_weight_col,
        layout_seed=layout_seed,
        axes=axes,
    )


def fplot_network(
    pd_df: pd.DataFrame,
    edge_source_col: str = "source",
    edge_target_col: str = "target",
    edge_weight_col: str = "weight",
    title: Optional[str] = None,
    style: Optional[StyleTemplate] = None,
    layout_seed: Optional[int] = _DEFAULT["SPRING_LAYOUT_SEED"],
    figsize: Tuple[float, float] = FIG_SIZE,
) -> Figure:
    """Return a figure with a network graph.

    Parameters
    ----------
    pd_df : pd.DataFrame
        DataFrame containing edge data.
    node_col : str, optional
        Column name for node identifiers. The default is "node".
    node_weight_col : str, optional
        Column name for node weights. The default is "weight".
    edge_source_col : str, optional
        Column name for source nodes. The default is "source".
    edge_target_col : str, optional
        Column name for target nodes. The default is "target".
    edge_weight_col : str, optional
        Column name for edge weights. The default is "weight".
    sort_by : str, optional
        Column used to sort the data.
    ascending : bool, optional
        Sort order for the data. The default is `False`.
    node_df : pd.DataFrame, optional
        DataFrame containing ``node`` and ``weight`` columns to include.
    title : str, optional
        Plot title.
    style : StyleTemplate, optional
        Style configuration. The default is `NETWORK_STYLE_TEMPLATE`.
    layout_seed : int, optional
        Seed for the spring layout used to place nodes. The default is ``_DEFAULT["SPRING_LAYOUT_SEED"]``.
    figsize : tuple[float, float], optional
        Size of the created figure. The default is FIG_SIZE.

    Returns
    -------
    Figure
        Matplotlib figure with the network graph.
    """
    return NetworkGraph(
        pd_df=pd_df,
        source=edge_source_col,
        target=edge_target_col,
        weight=edge_weight_col,
    ).fplot(
        title=title,
        style=style,
        figsize=figsize,
    )


def fplot_network_node(
    pd_df: pd.DataFrame,
    node: Any,
    edge_source_col: str = "source",
    edge_target_col: str = "target",
    edge_weight_col: str = "weight",
    title: Optional[str] = None,
    style: Optional[StyleTemplate] = None,
    figsize: Tuple[float, float] = FIG_SIZE,
    layout_seed: Optional[int] = _DEFAULT["SPRING_LAYOUT_SEED"],
) -> Figure:
    """Return a figure with the component containing ``node``.

    Parameters
    ----------
    pd_df : pd.DataFrame
        DataFrame containing edge data.
    node : Any
        Node identifier whose component should be visualized.
    node_col : str, optional
        Column name for node identifiers. The default is "node".
    node_weight_col : str, optional
        Column name for node weights. The default is "weight".
    edge_source_col : str, optional
        Column name for source nodes. The default is "source".
    edge_target_col : str, optional
        Column name for target nodes. The default is "target".
    edge_weight_col : str, optional
        Column name for edge weights. The default is "weight".
    sort_by : str, optional
        Column used to sort the data.
    ascending : bool, optional
        Sort order for the data. The default is `False`.
    node_df : pd.DataFrame, optional
        DataFrame containing ``node`` and ``weight`` columns to include.
    figsize : tuple[float, float], optional
        Size of the created figure. The default is FIG_SIZE.
    title : str, optional
        Plot title. If ``None``, defaults to the node identifier.
    style : StyleTemplate, optional
        Style configuration. The default is `NETWORK_STYLE_TEMPLATE`.
    save_path : str, optional
        File path to save the figure. The default is ``None``.
    savefig_kwargs : dict[str, Any], optional
        Extra keyword arguments forwarded to ``Figure.savefig``. The default is ``None``.
    layout_seed : int, optional
        Seed for the spring layout used to place nodes. The default is ``_DEFAULT["SPRING_LAYOUT_SEED"]``.

    Returns
    -------
    Figure
        Matplotlib figure with the component plot.

    Raises
    ------
    ValueError
        If ``node`` is not present in the prepared graph.
    """
    if not style:
        style = NETWORK_STYLE_TEMPLATE
    fig = cast(Figure, plt.figure(figsize=figsize))
    fig.set_facecolor(style.background_color)
    ax = fig.add_subplot(111)
    ax = aplot_network_node(
        pd_df,
        node=node,
        edge_source_col=edge_source_col,
        edge_target_col=edge_target_col,
        edge_weight_col=edge_weight_col,
        title=title,
        style=style,
        ax=ax,
        layout_seed=layout_seed,
    )
    return fig


def fplot_network_components(
    pd_df: pd.DataFrame,
    edge_source_col: str = "source",
    edge_target_col: str = "target",
    edge_weight_col: str = "weight",
    title: Optional[str] = None,
    style: Optional[StyleTemplate] = None,
    layout_seed: Optional[int] = _DEFAULT["SPRING_LAYOUT_SEED"],
    figsize: Tuple[float, float] = FIG_SIZE,
    n_cols: Optional[int] = None,
) -> Figure:
    """Return a figure showing individual network components.

    Parameters
    ----------
    pd_df : pd.DataFrame
        DataFrame containing edge data.
    node_col : str, optional
        Column name for node identifiers. The default is "node".
    node_weight_col : str, optional
        Column name for node weights. The default is "weight".
    edge_source_col : str, optional
        Column name for source nodes. The default is "source".
    edge_target_col : str, optional
        Column name for target nodes. The default is "target".
    edge_weight_col : str, optional
        Column name for edge weights. The default is "weight".
    sort_by : str, optional
        Column used to sort the data.
    ascending : bool, optional
        Sort order for the data. The default is `False`.
    node_df : pd.DataFrame, optional
        DataFrame containing ``node`` and ``weight`` columns to include.
    title : str, optional
        Plot title.
    style : StyleTemplate, optional
        Style configuration. The default is `NETWORK_STYLE_TEMPLATE`.
    layout_seed : int, optional
        Seed for the spring layout used to place nodes. The default is ``_DEFAULT["SPRING_LAYOUT_SEED"]``.
    figsize : tuple[float, float], optional
        Size of the created figure. The default is FIG_SIZE.
    n_cols : int, optional
        Number of columns for subplots. If None, it's inferred.
    save_path : str, optional
        File path to save the figure. The default is ``None``.
    savefig_kwargs : dict[str, Any], optional
        Extra keyword arguments forwarded to ``Figure.savefig``. The default is ``None``.

    Returns
    -------
    Figure
        Matplotlib figure displaying component plots.

    Raises
    ------
    ValueError
        If ``node_df`` is provided but none of its nodes appear as sources or
        targets in ``pd_df``.
    """
    if not style:
        style = NETWORK_STYLE_TEMPLATE
    graph = NetworkGraph(
        pd_df=pd_df,
        source=edge_source_col,
        target=edge_target_col,
        weight=edge_weight_col,
    )
    isolated_nodes = list(nx.isolates(graph._nx_graph))
    if isolated_nodes:
        graph._nx_graph.remove_nodes_from(isolated_nodes)
    connected_components = graph.connected_components

    n_components = max(1, len(connected_components))
    n_cols_local = int(np.ceil(np.sqrt(n_components))) if n_cols is None else n_cols
    n_rows = int(np.ceil(n_components / n_cols_local))

    fig, axes_grid = plt.subplots(n_rows, n_cols_local, figsize=figsize)
    fig = cast(Figure, fig)
    fig.set_facecolor(style.background_color)
    if not isinstance(axes_grid, np.ndarray):
        axes = np.array([axes_grid])
    else:
        axes = axes_grid.flatten()

    graph.aplot_connected_components(
        title=title,
        style=style,
        edge_weight_col=edge_weight_col,
        layout_seed=layout_seed,
        axes=axes,
    )

    if title:
        fig.suptitle(
            title,
            color=style.font_color,
            fontsize=style.font_size * TITLE_SCALE_FACTOR * 1.25,
        )

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    return fig
