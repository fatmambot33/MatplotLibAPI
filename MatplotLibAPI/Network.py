import logging
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from networkx import Graph
from networkx.classes.graph import Graph


from .StyleTemplate import StyleTemplate, string_formatter, format_func, validate_dataframe

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
    return (np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())


def scale_weights(weights, scale_min=0, scale_max=1):
    deciles = np.percentile(weights, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    outs = np.searchsorted(deciles, weights)
    return [out * (scale_max-scale_min)/len(deciles)+scale_min for out in outs]


class NodeView(nx.classes.reportviews.NodeView):
    def sort(self,
             attribute: Optional[str] = 'weight',
             reverse: Optional[bool] = True):
        # Sort the nodes based on the specified attribute
        sorted_nodes = sorted(self,
                              key=lambda node: self[node][attribute],
                              reverse=reverse)
        return sorted_nodes

    def filter(self, attribute: str, value: str):
        # Filter the nodes based on the specified attribute and value
        filtered_nodes = [
            node for node in self if attribute in self[node] and self[node][attribute] == value]
        return filtered_nodes


class AdjacencyView(nx.classes.coreviews.AdjacencyView):
    def sort(self,
             attribute: Optional[str] = 'weight',
             reverse: Optional[bool] = True):
        # Sort the nodes based on the specified attribute
        sorted_nodes = sorted(self,
                              key=lambda node: self[node][attribute],
                              reverse=reverse)
        return sorted_nodes

    def filter(self, attribute: str, value: str):
        # Filter the nodes based on the specified attribute and value
        filtered_nodes = [
            node for node in self if attribute in self[node] and self[node][attribute] == value]
        return filtered_nodes


class EdgeView(nx.classes.reportviews.EdgeView):
    def sort(self,
             reverse: Optional[bool] = True,
             attribute: Optional[str] = 'weight'):
        sorted_edges = sorted(self(data=True),
                              key=lambda t: t[2].get(attribute, 1),
                              reverse=reverse)
        return {(u, v): _ for u, v, _ in sorted_edges}

    def filter(self, attribute: str, value: str):
        # Filter the edges based on the specified attribute and value
        filtered_edges = [
            edge for edge in self if attribute in self[edge] and self[edge][attribute] == value]
        return [(edge[0], edge[1]) for edge in filtered_edges]


class networkGraph():
    """
    Custom graph class based on NetworkX's Graph class.
    """
    _nx_graph: nx.Graph

    def __init__(self, nx_graph: nx.Graph):
        self._nx_graph = nx_graph
        self._scale = 1.0

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float):
        self._scale = value

    @property
    def nodes(self):
        return NodeView(self._nx_graph)

    @nodes.setter
    def scale(self, value: NodeView):
        self.nodes = value

    @property
    def edges(self):
        return EdgeView(self._nx_graph)

    @property
    def adjacency(self):
        return AdjacencyView(list(self._nx_graph))

    def edge_subgraph(self, edges: Iterable) -> 'networkGraph':
        return nx.edge_subgraph(self._nx_graph, edges)

    def layout(self,
               max_node_size: int = DEFAULT["MAX_NODES"],
               min_node_size: int = DEFAULT["MAX_NODES"],
               max_edge_width: int = DEFAULT["MAX_EDGE_WIDTH"],
               max_font_size: int = DEFAULT["MAX_FONT_SIZE"],
               min_font_size: int = DEFAULT["MIN_FONT_SIZE"],
               weight: str = "weight"):
        """
        Calculates the sizes for nodes, edges, and fonts based on node weights and edge weights.

        Parameters:
        - max_node_size (int): Maximum size for nodes (default: 300).
        - max_edge_width (int): Maximum width for edges (default: 10).
        - max_font_size (int): Maximum font size for node labels (default: 18).

        Returns:
        - Tuple[List[int], List[int], Dict[int, List[str]]]: A tuple containing the node sizes, edge widths,
          and font sizes for node labels.
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

    def subgraph(self,
                 node_list: Optional[List[str]] = None,
                 max_edges: int = DEFAULT["MAX_EDGES"]) -> 'networkGraph':
        if node_list is None:
            node_list = self.nodes.sort("weight")[:DEFAULT["MAX_NODES"]]
        connected_subgraph_nodes = list(self.find_connected_subgraph())
        node_list = [
            node for node in node_list if node in connected_subgraph_nodes]

        subgraph = nx.subgraph(self, nbunch=node_list)
        edges = subgraph.top_k_edges(attribute="weight", k=5).keys()
        subgraph = subgraph.edge_subgraph(list(edges)[:max_edges])
        return networkGraph(subgraph)

    def plot_network(self,
                     title: Optional[str] = None,
                     style: StyleTemplate = NETWORK_STYLE_TEMPLATE,
                     weight: str = "weight",
                     ax: Optional[Axes] = None) -> Axes:
        """
        Plots the degree distribution of the graph, including a degree rank plot and a degree histogram.
        """
        degree_sequence = sorted(
            [d for n, d in self._nx_graph.degree()], reverse=True)
        dmax = max(degree_sequence)
        sns.set_palette(style.palette)
        if ax is None:
            ax = plt.gca()

        node_sizes, edge_widths, font_sizes = self.layout(
            min_node_size=DEFAULT["MIN_NODE_SIZE"]/5,
            max_node_size=DEFAULT["MAX_NODE_SIZE"],
            max_edge_width=DEFAULT["MAX_EDGE_WIDTH"],
            min_font_size=style.font_mapping.get(0),
            max_font_size=style.font_mapping.get(4),
            weight=weight)
        pos = nx.spring_layout(self._nx_graph, k=1)
        # nodes
        nx.draw_networkx_nodes(self._nx_graph,
                               pos,
                               ax=ax,
                               node_size=list(node_sizes),
                               node_color=node_sizes,
                               cmap=plt.get_cmap(style.palette))
        # edges
        nx.draw_networkx_edges(self._nx_graph,
                               pos,
                               ax=ax,
                               edge_color=style.font_color,
                               edge_cmap=plt.get_cmap(style.palette),
                               width=edge_widths)
        # labels
        for font_size, nodes in font_sizes.items():
            nx.draw_networkx_labels(
                self,
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
        # node_list=self.nodes_circuits(node_list)
        g = self.subgraph(max_edges=max_edges, node_list=node_list)
        connected_components = nx.connected_components(g)
        axes = []
        for connected_component in connected_components:
            if len(connected_component) > 5:
                connected_component_graph = self.subgraph(max_edges=max_edges,
                                                          node_list=connected_component)
                ax = connected_component_graph.plot_network()
                axes.append(ax)
        return axes

    def find_connected_subgraph(self) -> 'networkGraph':
        logging.info(f'find_connected_subgraph')
        # Copy the original graph to avoid modifying it
        H = self._nx_graph.copy()

        # Flag to keep track of whether any node with degree < 2 was removed
        removed_node = True

        while removed_node:
            removed_node = False
            # Iterate over the nodes
            for node in list(H.nodes):
                if H.degree(node) < 2:
                    # Remove the node and its incident edges
                    logging.info(
                        f'Removing the {node} node and its incident edges')
                    H.remove_node(node)
                    removed_node = True
                    break

        return networkGraph(H)

    def top_k_edges(self, attribute: str, reverse: bool = True, k: int = 5) -> Dict[Any, List[Tuple[Any, Dict]]]:
        """
        Returns the top k edges per node based on the given attribute.

        Parameters:
        attribute (str): The attribute name to be used for sorting.
        reverse (bool): Flag indicating whether to sort in reverse order (default: True).
        k (int): Number of top edges to return per node.

        Returns:
        Dict[Any, List[Tuple[Any, Dict]]]: A dictionary where the key is a node
        and the value is a list of top k edges for that node. Each edge is represented
        as a tuple where the first element is the adjacent node and the second element
        is a dictionary of edge attributes.
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
    def from_pandas_edgelist(df: pd.DataFrame,
                             source: str = "source",
                             target: str = "target",
                             weight: str = "weight") -> 'networkGraph':
        """
        Initialize netX instance with a simple dataframe

        :param df_source: DataFrame containing network data.
        :param source: Name of source nodes column in df_source.
        :param target: Name of target nodes column in df_source.
        :param weight: Name of edges weight column in df_source.

        """
        network_G = nx.from_pandas_edgelist(
            df, source=source, target=target, edge_attr=weight)
        network_G = networkGraph(network_G)
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

        nx.set_node_attributes(network_G, node_aggregates, name=weight)

        network_G = network_G.edge_subgraph(
            edges=network_G.top_k_edges(attribute=weight))
        return networkGraph(network_G)


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
    if node_list:
        df = pd_df[(pd_df["source"].isin(node_list)) |
                   (pd_df["target"].isin(node_list))]
    else:
        df = pd_df
    validate_dataframe(df, cols=[source, target, weight], sort_by=sort_by)

    graph = networkGraph.from_pandas_edgelist(df,
                                              source=source,
                                              target=target,
                                              weight=weight)
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
                             ax: Optional[List[Axes]] = None) -> List[Axes]:
    if node_list:
        df = pd_df[(pd_df["source"].isin(node_list)) |
                   (pd_df["target"].isin(node_list))]
    else:
        df = pd_df
    validate_dataframe(df, cols=[source, target, weight], sort_by=sort_by)

    graph = networkGraph.from_pandas_edgelist(df,
                                              source=source,
                                              target=target,
                                              weight=weight)
    connected_components = nx.connected_components(graph)
    axes = []
    i = 0
    for connected_component in connected_components:
        if len(connected_component) > 5:
            connected_component_graph = graph.subgraph(
                node_list=connected_component)
            ax = connected_component_graph.plot_network(title=f"{title}::{i}",
                                                        style=style,
                                                        weight=weight,
                                                        ax=ax)
            axes.append(ax)
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
