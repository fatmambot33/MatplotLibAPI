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

from MatplotLibAPI.StyleTemplate import (
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
    ) -> 'NetworkGraph':
        """Return a trimmed subgraph limited by nodes and edges.

        Args:
            node_list (Optional[List[str]], optional): Nodes to include. Defaults to ``None``.
            max_edges (int, optional): Maximum edges to retain. Defaults to ``DEFAULT["MAX_EDGES"]``.

        Returns:
            NetworkGraph: Trimmed subgraph.
        """
        if node_list is None:
            node_list = self.nodes.sort("weight")[: DEFAULT["MAX_NODES"]]
        connected_subgraph_nodes = list(self.find_connected_subgraph().nodes)
        node_list = [node for node in node_list if node in connected_subgraph_nodes]

        subgraph = NetworkGraph(nx.subgraph(self._nx_graph, nbunch=node_list))
        edges = subgraph.top_k_edges(attribute="weight", k=5).keys()
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
        nx.draw_networkx_nodes(
            self._nx_graph,
            pos,
            ax=ax,
            node_size=node_sizes,
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
                                node_list: Optional[List] = None,
                                scale: int = DEFAULT["GRAPH_SCALE"],
                                node_scale: int = DEFAULT["MAX_NODE_SIZE"],
                                edge_scale: float = DEFAULT["MAX_EDGE_WIDTH"],
                                max_nodes: int = DEFAULT["MAX_NODES"],
                                max_edges: int = DEFAULT["MAX_EDGES"],
                                plt_title: Optional[str] = "Top keywords"):
        """Plot each connected component of the graph.

        Args:
            node_list (Optional[List], optional): Nodes to include. Defaults to ``None``.
            scale (int, optional): Figure scaling factor. Defaults to ``DEFAULT["GRAPH_SCALE"]``.
            node_scale (int, optional): Scaling for node sizes. Defaults to ``DEFAULT["MAX_NODE_SIZE"]``.
            edge_scale (float, optional): Scaling for edge widths. Defaults to ``DEFAULT["MAX_EDGE_WIDTH"]``.
            max_nodes (int, optional): Maximum nodes per component. Defaults to ``DEFAULT["MAX_NODES"]``.
            max_edges (int, optional): Maximum edges per component. Defaults to ``DEFAULT["MAX_EDGES"]``.
            plt_title (Optional[str], optional): Base title for component plots. Defaults to ``"Top keywords"``.

        Returns:
            List[Axes]: Axes for each plotted component.
        """
        # node_list=self.nodes_circuits(node_list)
        g = self.subgraph(max_edges=max_edges, node_list=node_list)
        connected_components = nx.connected_components(g._nx_graph)
        axes = []
        for connected_component in connected_components:
            if len(connected_component) > 5:
                connected_component_graph = self.subgraph(max_edges=max_edges,
                                                          node_list=connected_component)
                ax = connected_component_graph.plot_network()
                axes.append(ax)
        return axes

    def find_connected_subgraph(self) -> 'NetworkGraph':
        """Return subgraph containing only nodes with degree >= 2.

        Returns:
            NetworkGraph: Connected subgraph.
        """
        logging.info('find_connected_subgraph')
        # Copy the original graph to avoid modifying it
        H = self._nx_graph.copy()

        # Flag to keep track of whether any node with degree < 2 was removed
        removed_node = True

        while removed_node:
            removed_node = False
            # Iterate over the nodes
            for node in list(H.nodes):
                if H.degree[node] < 2:
                    # Remove the node and its incident edges
                    logging.info(
                        f'Removing the {node} node and its incident edges')
                    H.remove_node(node)
                    removed_node = True
                    break

        return NetworkGraph(H)

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
        network_G = NetworkGraph(network_G)
        network_G = network_G.find_connected_subgraph()

        edge_aggregates = network_G.top_k_edges(attribute=weight, k=10)
        node_aggregates = {}
        for (u, v), weight_value in edge_aggregates.items():
            if u not in node_aggregates:
                node_aggregates[u] = 0
            if v not in node_aggregates:
                node_aggregates[v] = 0
            node_aggregates[u] += weight_value
            node_aggregates[v] += weight_value

        nx.set_node_attributes(network_G._nx_graph, node_aggregates, name=weight)

        network_G = network_G.edge_subgraph(
            edges=network_G.top_k_edges(attribute=weight)
        )
        return network_G


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
    if node_list:
        df = pd_df.loc[(pd_df["source"].isin(node_list)) |
                       (pd_df["target"].isin(node_list))]
    else:
        df = pd_df
    validate_dataframe(df, cols=[source, target, weight], sort_by=sort_by)

    graph = NetworkGraph.from_pandas_edgelist(
        df, source=source, target=target, weight=weight
    )
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
                             ax: Optional[Axes] = None) -> List[Axes]:
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
        ax (Optional[Axes], optional): Existing axes to draw on. Defaults to ``None``.

    Returns:
        List[Axes]: Axes for each component plotted.
    """
    if node_list:
        df = pd_df.loc[(pd_df["source"].isin(node_list)) |
                       (pd_df["target"].isin(node_list))]
    else:
        df = pd_df
    validate_dataframe(df, cols=[source, target, weight], sort_by=sort_by)

    graph = NetworkGraph.from_pandas_edgelist(
        df, source=source, target=target, weight=weight
    )
    connected_components = nx.connected_components(graph._nx_graph)
    axes = []
    i = 0
    for connected_component in connected_components:
        if len(connected_component) > 5:
            connected_component_graph = graph.subgraph(
                node_list=connected_component)
            comp_ax = connected_component_graph.plot_network(title=f"{title}::{i}",
                                                             style=style,
                                                             weight=weight,
                                                             ax=ax)
            axes.append(comp_ax)
            i += 1
    return axes


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
                             figsize: Tuple[float, float] = (19.2, 10.8)) -> Figure:
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

    Returns:
        Figure: Matplotlib figure displaying component plots.
    """
    axes = aplot_network_components(pd_df,
                                    source=source,
                                    target=target,
                                    weight=weight,
                                    title=title,
                                    style=style,
                                    sort_by=sort_by,
                                    ascending=ascending,
                                    node_list=node_list
                                    )
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
