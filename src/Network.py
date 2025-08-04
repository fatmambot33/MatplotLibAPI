"""Network chart plotting helpers."""

import logging
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd

from src.StyleTemplate import (
    StyleTemplate,
    string_formatter,
    format_func,
    validate_dataframe,
)

NETWORK_STYLE_TEMPLATE = StyleTemplate(
)

DEFAULT = {"MAX_EDGES": 100,
           "MAX_NODES": 30,
           "MIN_NODE_SIZE": 100,
           "MAX_NODE_SIZE": 2000,
           "MAX_EDGE_WIDTH": 10,
           "GRAPH_SCALE": 2,
           "MAX_FONT_SIZE": 20,
           "MIN_FONT_SIZE": 8
           }


def softmax(x):
    """Compute softmax values for array ``x``.

    Args:
        x (Iterable[float]): Input values.

    Returns:
        np.ndarray: Softmax-transformed values.
    """
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()


def scale_weights(weights, scale_min=0, scale_max=1):
    """Scale weights into deciles within the given range.

    Args:
        weights (Iterable[float]): Sequence of weights to scale.
        scale_min (float, optional): Minimum of the output range. Defaults to ``0``.
        scale_max (float, optional): Maximum of the output range. Defaults to ``1``.

    Returns:
        List[float]: Scaled weights.
    """
    deciles = np.percentile(weights, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    outs = np.searchsorted(deciles, weights)
    return [out * (scale_max - scale_min) / len(deciles) + scale_min for out in outs]


class NodeView(nx.classes.reportviews.NodeView):
    """Extended node view with convenience helpers."""

    def sort(self,
             attribute: str = "weight",
             reverse: bool = True) -> List[Any]:
        """Return nodes sorted by the specified attribute.

        Args:
            attribute (str, optional): Node attribute used for sorting. Defaults to ``"weight"``.
            reverse (bool, optional): Sort order. Defaults to ``True``.

        Returns:
            List[Any]: Sorted nodes.
        """
        sorted_nodes = sorted(
            self, key=lambda node: self[node].get(attribute, 1), reverse=reverse
        )
        return sorted_nodes

    def filter(self, attribute: str, value: str):
        """Return nodes where ``attribute`` equals ``value``.

        Args:
            attribute (str): Node attribute to compare.
            value (str): Desired attribute value.

        Returns:
            List: Nodes matching the condition.
        """
        filtered_nodes = [
            node
            for node in self
            if self[node].get(attribute) == value
        ]
        return filtered_nodes


class AdjacencyView(nx.classes.coreviews.AdjacencyView):
    """Adjacency view with sorting and filtering helpers."""

    def sort(self,
             attribute: str = "weight",
             reverse: bool = True) -> List[Any]:
        """Return adjacent nodes sorted by the given attribute.

        Args:
            attribute (str, optional): Attribute used for sorting. Defaults to ``"weight"``.
            reverse (bool, optional): Sort order. Defaults to ``True``.

        Returns:
            List[Any]: Sorted adjacent nodes.
        """
        sorted_nodes = sorted(
            self, key=lambda node: self[node].get(attribute, 1), reverse=reverse
        )
        return sorted_nodes

    def filter(self, attribute: str, value: str):
        """Return adjacent nodes where ``attribute`` equals ``value``.

        Args:
            attribute (str): Node attribute to compare.
            value (str): Desired attribute value.

        Returns:
            List: Adjacent nodes matching the value.
        """
        filtered_nodes = [
            node
            for node in self
            if self[node].get(attribute) == value
        ]
        return filtered_nodes


class EdgeView(nx.classes.reportviews.EdgeView):
    """Edge view with sorting and filtering helpers."""

    def sort(self,
             attribute: str = "weight",
             reverse: bool = True) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
        """Return edges sorted by the given attribute.

        Args:
            attribute (str, optional): Edge attribute used for sorting. Defaults to ``"weight"``.
            reverse (bool, optional): Sort order. Defaults to ``True``.

        Returns:
            Dict[Tuple[Any, Any], Dict[str, Any]]: Mapping of edge tuples to their attributes.
        """
        sorted_edges = sorted(
            self(data=True), key=lambda t: t[2].get(attribute, 1), reverse=reverse
        )
        return {(u, v): data for u, v, data in sorted_edges}

    def filter(self, attribute: str, value: str):
        """Return edges where ``attribute`` equals ``value``.

        Args:
            attribute (str): Edge attribute to compare.
            value (str): Desired attribute value.

        Returns:
            List[Tuple[Any, Any]]: Edges matching the condition.
        """
        filtered_edges = [
            edge
            for edge in self
            if self[edge].get(attribute) == value
        ]
        return [(edge[0], edge[1]) for edge in filtered_edges]


class NetworkGraph:
    """Custom graph class based on NetworkX's ``Graph``."""

    _nx_graph: nx.Graph

    def __init__(self, nx_graph: nx.Graph):
        """Initialize with an existing NetworkX graph.

        Args:
            nx_graph (nx.Graph): Graph to wrap.
        """
        self._nx_graph = nx_graph
        self._scale = 1.0

    @property
    def scale(self) -> float:
        """Return scaling factor for plotting sizes."""
        return self._scale

    @scale.setter
    def scale(self, value: float):
        """Set scaling factor for plotting sizes.

        Args:
            value (float): Scaling factor.
        """
        self._scale = value

    @property
    def nodes(self):
        """Return a ``NodeView`` over the graph."""
        return NodeView(self._nx_graph)

    @property
    def edges(self):
        """Return an ``EdgeView`` over the graph."""
        return EdgeView(self._nx_graph)

    @property
    def adjacency(self):
        """Return an ``AdjacencyView`` of the graph."""
        return AdjacencyView(self._nx_graph.adj)

    def edge_subgraph(self, edges: Iterable) -> 'NetworkGraph':
        """Return a subgraph containing only the specified edges.

        Args:
            edges (Iterable): Edges to include.

        Returns:
            NetworkGraph: Subgraph with only ``edges``.
        """
        return NetworkGraph(nx.edge_subgraph(self._nx_graph, edges))

    def layout(
        self,
        max_node_size: int = DEFAULT["MAX_NODE_SIZE"],
        min_node_size: int = DEFAULT["MIN_NODE_SIZE"],
        max_edge_width: int = DEFAULT["MAX_EDGE_WIDTH"],
        max_font_size: int = DEFAULT["MAX_FONT_SIZE"],
        min_font_size: int = DEFAULT["MIN_FONT_SIZE"],
        weight: str = "weight",
    ) -> Tuple[List[float], List[float], Dict[int, List[str]]]:
        """Calculate node, edge and font sizes based on weights.

        Args:
            max_node_size (int, optional): Upper bound for node size. Defaults to ``DEFAULT["MAX_NODE_SIZE"]``.
            min_node_size (int, optional): Lower bound for node size. Defaults to ``DEFAULT["MIN_NODE_SIZE"]``.
            max_edge_width (int, optional): Upper bound for edge width. Defaults to ``DEFAULT["MAX_EDGE_WIDTH"]``.
            max_font_size (int, optional): Upper bound for font size. Defaults to ``DEFAULT["MAX_FONT_SIZE"]``.
            min_font_size (int, optional): Lower bound for font size. Defaults to ``DEFAULT["MIN_FONT_SIZE"]``.
            weight (str, optional): Node attribute used for weighting. Defaults to ``"weight"``.

        Returns:
            Tuple[List[float], List[float], Dict[int, List[str]]]: Node sizes, edge widths and nodes grouped by font size.
        """
        # Normalize and scale nodes' weights within the desired range of edge widths
        node_weights = [data.get(weight, 1)
                        for node, data in self.nodes(data=True)]
        node_size = scale_weights(
            weights=node_weights, scale_max=max_node_size, scale_min=min_node_size)

        # Normalize and scale edges' weights within the desired range of edge widths
        edge_weights = [data.get(weight, 1)
                        for _, _, data in self.edges(data=True)]
        edges_width = scale_weights(
            weights=edge_weights, scale_max=max_edge_width)

        # Scale the normalized node weights within the desired range of font sizes
        node_size_dict = dict(zip(self.nodes, scale_weights(
            weights=node_weights, scale_max=max_font_size, scale_min=min_font_size)))
        fonts_size = defaultdict(list)
        for node, width in node_size_dict.items():
            fonts_size[int(width)].append(node)
        fonts_size = dict(fonts_size)

        return node_size, edges_width, fonts_size

    def subgraph(
        self,
        node_list: Optional[List[str]] = None,
        max_edges: int = DEFAULT["MAX_EDGES"],
        min_degree: int = 2,
        top_k_edges_per_node: int = 5,
    ) -> 'NetworkGraph':
        """Return a trimmed subgraph limited by nodes and edges.

        Args:
            node_list (Optional[List[str]], optional): Nodes to include. Defaults to ``None``.
            max_edges (int, optional): Maximum edges to retain. Defaults to ``DEFAULT["MAX_EDGES"]``.
            min_degree (int, optional): Minimum degree for nodes in the core subgraph. Defaults to 2.
            top_k_edges_per_node (int, optional): Number of top edges to keep per node. Defaults to 5.

        Returns:
            NetworkGraph: Trimmed subgraph.
        """
        if node_list is None:
            node_list = self.nodes.sort("weight")[: DEFAULT["MAX_NODES"]]
        core_subgraph_nodes = list(self.get_core_subgraph(k=min_degree).nodes)
        node_list = [node for node in node_list if node in core_subgraph_nodes]

        subgraph = NetworkGraph(nx.subgraph(self._nx_graph, nbunch=node_list))
        edges = subgraph.top_k_edges(attribute="weight", k=top_k_edges_per_node).keys()
        subgraph = subgraph.edge_subgraph(list(edges)[:max_edges])
        return subgraph

    def plot_network(self,
                     title: Optional[str] = None,
                     style: StyleTemplate = NETWORK_STYLE_TEMPLATE,
                     weight: str = "weight",
                     ax: Optional[Axes] = None) -> Axes:
        """Plot the graph using node and edge weights.

        Args:
            title (Optional[str], optional): Plot title. Defaults to ``None``.
            style (StyleTemplate, optional): Style configuration. Defaults to ``NETWORK_STYLE_TEMPLATE``.
            weight (str, optional): Edge attribute used for weighting. Defaults to ``"weight"``.
            ax (Optional[Axes], optional): Axes to draw on. Defaults to ``None``.

        Returns:
            Axes: Matplotlib axes with the plotted network.
        """
        sns.set_palette(style.palette)
        if ax is None:
            ax = plt.gca()

        node_sizes, edge_widths, font_sizes = self.layout(
            min_node_size=DEFAULT["MIN_NODE_SIZE"] // 5,
            max_node_size=DEFAULT["MAX_NODE_SIZE"],
            max_edge_width=DEFAULT["MAX_EDGE_WIDTH"],
            min_font_size=style.font_mapping.get(0, DEFAULT["MIN_FONT_SIZE"]),
            max_font_size=style.font_mapping.get(4, DEFAULT["MAX_FONT_SIZE"]),
            weight=weight)
        pos = nx.spring_layout(self._nx_graph, k=1)
        # nodes
        node_sizes_int = [int(size) for size in node_sizes]
        nx.draw_networkx_nodes(
            self._nx_graph,
            pos,
            ax=ax,
            node_size=cast(Any, node_sizes_int),
            node_color=cast(Any, node_sizes),
            cmap=plt.get_cmap(style.palette),
        )
        # edges
        nx.draw_networkx_edges(
            self._nx_graph,
            pos,
            ax=ax,
            edge_color=style.font_color,
            edge_cmap=plt.get_cmap(style.palette),
            width=cast(Any, edge_widths),
        )
        # labels
        for font_size, nodes in font_sizes.items():
            nx.draw_networkx_labels(
                self._nx_graph,
                pos,
                ax=ax,
                font_size=font_size,
                font_color=style.font_color,
                labels={n: string_formatter(n) for n in nodes})
        ax.set_facecolor(style.background_color)
        if title:
            ax.set_title(title, color=style.font_color,
                         fontsize=style.font_size*2)
        ax.set_axis_off()

        return ax

    def plot_network_components(self,
                                *args, **kwargs):
        """Plot network components (DEPRECATED).

        .. deprecated:: 0.1.0
            This method will be removed in a future version.
            Please use `fplot_network_components` which provides a figure-level interface
            for plotting components.
        """
        import warnings
        warnings.warn(
            "`plot_network_components` is deprecated and will be removed in a future version. "
            "Please use `fplot_network_components`.",
            DeprecationWarning,
            stacklevel=2,
        )
        # For backward compatibility, we can try to call the new function,
        # but it's better to encourage users to switch.
        # This is a bit tricky because the function signatures are different.
        # For now, we'll just warn.
        return []

    def get_core_subgraph(self, k: int = 2) -> 'NetworkGraph':
        """Return the k-core of the graph.

        The k-core is a subgraph containing only nodes with degree >= k.

        Args:
            k (int, optional): The minimum degree for nodes in the core. Defaults to 2.

        Returns:
            NetworkGraph: The k-core subgraph.
        """
        core_graph = nx.k_core(self._nx_graph, k=k)
        return NetworkGraph(core_graph)

    def top_k_edges(self, attribute: str, reverse: bool = True, k: int = 5) -> Dict[Any, List[Tuple[Any, Dict]]]:
        """Return the top ``k`` edges based on a given attribute.

        Args:
            attribute (str): Attribute name used for sorting.
            reverse (bool, optional): Whether to sort in descending order. Defaults to ``True``.
            k (int, optional): Number of top edges to return. Defaults to ``5``.

        Returns:
            Dict[Any, List[Tuple[Any, Dict]]]: Mapping of edge tuples to attribute values.
        """
        top_list = {}
        for node in self.nodes:
            edges = self.edges(node, data=True)
            edges_sorted = sorted(edges, key=lambda x: x[2].get(
                attribute, 0), reverse=reverse)
            top_k_edges = edges_sorted[:k]
            for u, v, data in top_k_edges:
                edge_key = (u, v)
                top_list[edge_key] = data[attribute]
        return top_list

    def calculate_node_weights_from_edges(self, weight: str = "weight", k: int = 10):
        """Calculate node weights by summing weights of top k edges.

        Args:
            weight (str, optional): Edge attribute to use for weighting. Defaults to "weight".
            k (int, optional): Number of top edges to consider for each node. Defaults to 10.
        """
        edge_aggregates = self.top_k_edges(attribute=weight, k=k)
        node_aggregates = {}
        for (u, v), weight_value in edge_aggregates.items():
            if u not in node_aggregates:
                node_aggregates[u] = 0
            if v not in node_aggregates:
                node_aggregates[v] = 0
            node_aggregates[u] += weight_value
            node_aggregates[v] += weight_value

        nx.set_node_attributes(self._nx_graph, node_aggregates, name=weight)

    def trim_edges(self, weight: str = "weight", top_k_per_node: int = 5) -> 'NetworkGraph':
        """Trim the graph to keep only the top k edges per node.

        Args:
            weight (str, optional): Edge attribute to use for sorting. Defaults to "weight".
            top_k_per_node (int, optional): Number of top edges to keep per node. Defaults to 5.

        Returns:
            NetworkGraph: A new graph containing only the top edges.
        """
        edges_to_keep = self.top_k_edges(attribute=weight, k=top_k_per_node)
        return self.edge_subgraph(edges=edges_to_keep)

    @staticmethod
    def from_pandas_edgelist(
        df: pd.DataFrame,
        source: str = "source",
        target: str = "target",
        weight: str = "weight",
    ) -> 'NetworkGraph':
        """Initialize a NetworkGraph from a simple DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing edge data.
            source (str, optional): Column name for source nodes. Defaults to ``"source"``.
            target (str, optional): Column name for target nodes. Defaults to ``"target"``.
            weight (str, optional): Column name for edge weights. Defaults to ``"weight"``.

        Returns:
            NetworkGraph: Initialized network graph.
        """
        network_G = nx.from_pandas_edgelist(
            df, source=source, target=target, edge_attr=weight
        )
        return NetworkGraph(network_G)


def _prepare_network_graph(pd_df: pd.DataFrame, source: str, target: str, weight: str, sort_by: Optional[str], node_list: Optional[List]) -> NetworkGraph:
    """Prepare NetworkGraph for plotting."""
    if node_list:
        df = pd_df.loc[(pd_df["source"].isin(node_list)) |
                       (pd_df["target"].isin(node_list))]
    else:
        df = pd_df
    validate_dataframe(df, cols=[source, target, weight], sort_by=sort_by)

    graph = NetworkGraph.from_pandas_edgelist(
        df, source=source, target=target, weight=weight
    )
    graph = graph.get_core_subgraph(k=2)
    graph.calculate_node_weights_from_edges(weight=weight, k=10)
    graph = graph.trim_edges(weight=weight, top_k_per_node=5)
    return graph


def aplot_network(pd_df: pd.DataFrame,
                  source: str = "source",
                  target: str = "target",
                  weight: str = "weight",
                  title: Optional[str] = None,
                  style: StyleTemplate = NETWORK_STYLE_TEMPLATE,
                  sort_by: Optional[str] = None,
                  ascending: bool = False,
                  node_list: Optional[List] = None,
                  ax: Optional[Axes] = None) -> Axes:
    """Plot a network graph on the provided axes.

    Args:
        pd_df (pd.DataFrame): DataFrame containing edge data.
        source (str, optional): Column name for source nodes. Defaults to ``"source"``.
        target (str, optional): Column name for target nodes. Defaults to ``"target"``.
        weight (str, optional): Column name for edge weights. Defaults to ``"weight"``.
        title (Optional[str], optional): Plot title. Defaults to ``None``.
        style (StyleTemplate, optional): Style configuration. Defaults to ``NETWORK_STYLE_TEMPLATE``.
        sort_by (Optional[str], optional): Column used to sort the data. Defaults to ``None``.
        ascending (bool, optional): Sort order for the data. Defaults to ``False``.
        node_list (Optional[List], optional): Nodes to include. Defaults to ``None``.
        ax (Optional[Axes], optional): Axes to draw on. Defaults to ``None``.

    Returns:
        Axes: Matplotlib axes with the plotted network.
    """
    graph = _prepare_network_graph(
        pd_df, source, target, weight, sort_by, node_list)
    return graph.plot_network(title=title,
                              style=style,
                              weight=weight,
                              ax=ax)


def aplot_network_components(pd_df: pd.DataFrame,
                             source: str = "source",
                             target: str = "target",
                             weight: str = "weight",
                             title: Optional[str] = None,
                             style: StyleTemplate = NETWORK_STYLE_TEMPLATE,
                             sort_by: Optional[str] = None,
                             node_list: Optional[List] = None,
                             ascending: bool = False,
                             axes: Optional[np.ndarray] = None) -> None:
    """Plot network components separately on multiple axes.

    Args:
        pd_df (pd.DataFrame): DataFrame containing edge data.
        source (str, optional): Column name for source nodes. Defaults to ``"source"``.
        target (str, optional): Column name for target nodes. Defaults to ``"target"``.
        weight (str, optional): Column name for edge weights. Defaults to ``"weight"``.
        title (Optional[str], optional): Base title for subplots. Defaults to ``None``.
        style (StyleTemplate, optional): Style configuration. Defaults to ``NETWORK_STYLE_TEMPLATE``.
        sort_by (Optional[str], optional): Column used to sort the data. Defaults to ``None``.
        node_list (Optional[List], optional): Nodes to include. Defaults to ``None``.
        ascending (bool, optional): Sort order for the data. Defaults to ``False``.
        axes (Optional[np.ndarray], optional): Existing axes to draw on. If None, new axes are created. Defaults to ``None``.
    """
    graph = _prepare_network_graph(
        pd_df, source, target, weight, sort_by, node_list)

    connected_components = list(nx.connected_components(graph._nx_graph))

    if not connected_components:
        if axes is not None:
            for ax in axes.flatten():
                ax.set_axis_off()
        return

    if axes is None:
        n_components = len(connected_components)
        n_cols = int(np.ceil(np.sqrt(n_components)))
        n_rows = int(np.ceil(n_components / n_cols))
        fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(19.2, 10.8))
        fig.patch.set_facecolor(style.background_color)
        if not isinstance(axes_grid, np.ndarray):
            axes = np.array([axes_grid])
        else:
            axes = axes_grid.flatten()

    i = -1
    for i, component in enumerate(connected_components):
        if i < len(axes):
            if len(component) > 5:
                component_graph = graph.subgraph(node_list=list(component))
                component_graph.plot_network(
                    title=f"{title}::{i}" if title else str(i),
                    style=style,
                    weight=weight,
                    ax=axes[i]
                )
            axes[i].set_axis_on()
        else:
            break  # Stop if there are more components than axes

    # Turn off any unused axes
    if axes is not None:
        for j in range(i + 1, len(axes)):
            axes[j].set_axis_off()


def fplot_network(pd_df: pd.DataFrame,
                  source: str = "source",
                  target: str = "target",
                  weight: str = "weight",
                  title: Optional[str] = None,
                  style: StyleTemplate = NETWORK_STYLE_TEMPLATE,
                  sort_by: Optional[str] = None,
                  ascending: bool = False,
                  node_list: Optional[List] = None,
                  figsize: Tuple[float, float] = (19.2, 10.8)) -> Figure:
    """Return a figure with a network graph.

    Args:
        pd_df (pd.DataFrame): DataFrame containing edge data.
        source (str, optional): Column name for source nodes. Defaults to ``"source"``.
        target (str, optional): Column name for target nodes. Defaults to ``"target"``.
        weight (str, optional): Column name for edge weights. Defaults to ``"weight"``.
        title (Optional[str], optional): Plot title. Defaults to ``None``.
        style (StyleTemplate, optional): Style configuration. Defaults to ``NETWORK_STYLE_TEMPLATE``.
        sort_by (Optional[str], optional): Column used to sort the data. Defaults to ``None``.
        ascending (bool, optional): Sort order for the data. Defaults to ``False``.
        node_list (Optional[List], optional): Nodes to include. Defaults to ``None``.
        figsize (Tuple[float, float], optional): Size of the created figure. Defaults to ``(19.2, 10.8)``.

    Returns:
        Figure: Matplotlib figure with the network graph.
    """
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(style.background_color)
    ax = fig.add_subplot()
    ax = aplot_network(pd_df,
                       source=source,
                       target=target,
                       weight=weight,
                       title=title,
                       style=style,
                       sort_by=sort_by,
                       ascending=ascending,
                       node_list=node_list,
                       ax=ax
                       )
    return fig


def fplot_network_components(pd_df: pd.DataFrame,
                             source: str = "source",
                             target: str = "target",
                             weight: str = "weight",
                             title: Optional[str] = None,
                             style: StyleTemplate = NETWORK_STYLE_TEMPLATE,
                             sort_by: Optional[str] = None,
                             ascending: bool = False,
                             node_list: Optional[List] = None,
                             figsize: Tuple[float, float] = (19.2, 10.8),
                             n_cols: Optional[int] = None) -> Figure:
    """Return a figure showing individual network components.

    Args:
        pd_df (pd.DataFrame): DataFrame containing edge data.
        source (str, optional): Column name for source nodes. Defaults to ``"source"``.
        target (str, optional): Column name for target nodes. Defaults to ``"target"``.
        weight (str, optional): Column name for edge weights. Defaults to ``"weight"``.
        title (Optional[str], optional): Plot title. Defaults to ``None``.
        style (StyleTemplate, optional): Style configuration. Defaults to ``NETWORK_STYLE_TEMPLATE``.
        sort_by (Optional[str], optional): Column used to sort the data. Defaults to ``None``.
        ascending (bool, optional): Sort order for the data. Defaults to ``False``.
        node_list (Optional[List], optional): Nodes to include. Defaults to ``None``.
        figsize (Tuple[float, float], optional): Size of the created figure. Defaults to ``(19.2, 10.8)``.
        n_cols (Optional[int], optional): Number of columns for subplots. If None, it's inferred. Defaults to ``None``.

    Returns:
        Figure: Matplotlib figure displaying component plots.
    """
    # First, get the graph and components to determine the layout
    df = pd_df.copy()
    if node_list:
        df = df.loc[(df["source"].isin(node_list)) | (df["target"].isin(node_list))]

    validate_dataframe(df, cols=[source, target, weight], sort_by=sort_by)
    graph = NetworkGraph.from_pandas_edgelist(df, source=source, target=target, weight=weight)
    graph = graph.get_core_subgraph(k=2)
    connected_components = list(nx.connected_components(graph._nx_graph))

    n_components = len(connected_components)
    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(n_components)))
    n_rows = int(np.ceil(n_components / n_cols))

    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.patch.set_facecolor(style.background_color)

    if not isinstance(axes_grid, np.ndarray):
        axes = np.array([axes_grid])
    else:
        axes = axes_grid.flatten()

    aplot_network_components(
        pd_df=pd_df,
        source=source,
        target=target,
        weight=weight,
        title=title,
        style=style,
        sort_by=sort_by,
        ascending=ascending,
        node_list=node_list,
        axes=axes
    )

    if title:
        fig.suptitle(title, color=style.font_color, fontsize=style.font_size * 2.5)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    return fig
