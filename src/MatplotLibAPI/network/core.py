"""Network chart plotting helpers."""

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..base_plot import BasePlot
from .constants import _DEFAULT, _WEIGHT_PERCENTILES
from .scaling import _scale_weights
from ..style_template import (
    NETWORK_STYLE_TEMPLATE,
    FIG_SIZE,
    TITLE_SCALE_FACTOR,
    StyleTemplate,
    string_formatter,
    validate_dataframe,
)

__all__ = [
    "NETWORK_STYLE_TEMPLATE",
    "NetworkGraph",
]


def _compute_deciles(weights: Iterable[float]) -> Optional[np.ndarray]:
    """Return deciles for ``weights`` or ``None`` when empty."""
    weights_arr = np.asarray(list(weights), dtype=float)
    if weights_arr.size == 0:
        return None
    return np.percentile(weights_arr, _WEIGHT_PERCENTILES)


class NodeView(nx.classes.reportviews.NodeView):
    """Extended node view with convenience helpers."""

    def __getitem__(self, n: Any) -> Dict[str, Any]:
        return super().__getitem__(n)

    def sort(self, attribute: str = "weight", reverse: bool = True) -> List[Any]:
        """Return nodes sorted by the specified attribute.

        Parameters
        ----------
        attribute : str, optional
            Node attribute used for sorting. The default is "weight".
        reverse : bool, optional
            Sort order. The default is `True`.

        Returns
        -------
        list[Any]
            Sorted nodes.
        """
        sorted_nodes = sorted(
            self, key=lambda node: self[node].get(attribute, 1), reverse=reverse
        )
        return sorted_nodes

    def filter(self, attribute: str, value: str) -> List[Any]:
        """Return nodes where ``attribute`` equals ``value``.

        Parameters
        ----------
        attribute : str
            Node attribute to compare.
        value : str
            Desired attribute value.

        Returns
        -------
        list
            Nodes matching the condition.
        """
        filtered_nodes = [node for node in self if self[node].get(attribute) == value]
        return filtered_nodes

    def to_dataframe(
        self, node_col: str = "node", weight_col: str = "weight"
    ) -> pd.DataFrame:
        """Convert the node view to a DataFrame.

        Parameters
        ----------
        node_col : str, optional
            Column name for node identifiers. The default is "node".
        weight_col : str, optional
            Column name for node weights. The default is "weight".

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for nodes and their weights.
        """
        data = [
            {node_col: node, weight_col: data.get(weight_col, 1)}
            for node, data in self(data=True)
        ]
        return pd.DataFrame(data)


class AdjacencyView(nx.classes.coreviews.AdjacencyView):
    """Adjacency view with sorting and filtering helpers."""

    def sort(self, attribute: str = "weight", reverse: bool = True) -> List[Any]:
        """Return adjacent nodes sorted by the given attribute.

        Parameters
        ----------
        attribute : str, optional
            Attribute used for sorting. The default is "weight".
        reverse : bool, optional
            Sort order. The default is `True`.

        Returns
        -------
        list[Any]
            Sorted adjacent nodes.
        """
        sorted_nodes = sorted(
            self, key=lambda node: self[node].get(attribute, 1), reverse=reverse
        )
        return sorted_nodes

    def filter(self, attribute: str, value: str) -> List[Any]:
        """Return adjacent nodes where ``attribute`` equals ``value``.

        Parameters
        ----------
        attribute : str
            Node attribute to compare.
        value : str
            Desired attribute value.

        Returns
        -------
        list
            Adjacent nodes matching the value.
        """
        filtered_nodes = [node for node in self if self[node].get(attribute) == value]
        return filtered_nodes

    def to_dataframe(
        self, node_col: str = "node", weight_col: str = "weight"
    ) -> pd.DataFrame:
        """Convert the adjacency view to a DataFrame.

        Parameters
        ----------
        node_col : str, optional
            Column name for node identifiers. The default is "node".
        weight_col : str, optional
            Column name for node weights. The default is "weight".

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for adjacent nodes and their weights.
        """
        data = {node: self[node].get(weight_col, 1) for node in self}
        return pd.DataFrame(data)


class EdgeView(nx.classes.reportviews.EdgeView):
    """Edge view with sorting and filtering helpers."""

    def __getitem__(self, e: Tuple) -> Dict[str, Any]:
        return super().__getitem__(e)

    def sort(
        self, attribute: str = "weight", reverse: bool = True
    ) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
        """Return edges sorted by the given attribute.

        Parameters
        ----------
        attribute : str, optional
            Edge attribute used for sorting. The default is "weight".
        reverse : bool, optional
            Sort order. The default is `True`.

        Returns
        -------
        dict[tuple[Any, Any], dict[str, Any]]
            Mapping of edge tuples to their attributes.
        """
        sorted_edges = sorted(
            self(data=True), key=lambda t: t[2].get(attribute, 1), reverse=reverse
        )
        return {(u, v): data for u, v, data in sorted_edges}

    def filter(self, attribute: str, value: str) -> List[Tuple[Any, Any]]:
        """Return edges where ``attribute`` equals ``value``.

        Parameters
        ----------
        attribute : str
            Edge attribute to compare.
        value : str
            Desired attribute value.

        Returns
        -------
        list[tuple[Any, Any]]
            Edges matching the condition.
        """
        return [
            (edge[0], edge[1]) for edge in self if self[edge].get(attribute) == value
        ]

    def to_dataframe(
        self,
        source_col: str = "source",
        target_col: str = "target",
        weight_col: str = "weight",
    ) -> pd.DataFrame:
        """Convert the edge view to a DataFrame.

        Parameters
        ----------
        source_col : str, optional
            Column name for source nodes. The default is "source".
        target_col : str, optional
            Column name for target nodes. The default is "target".
        weight_col : str, optional
            Column name for edge weights. The default is "weight".

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for source, target and weight.
        """
        data = [
            {source_col: u, target_col: v, weight_col: data.get(weight_col, 1)}
            for u, v, data in self(data=True)
        ]
        return pd.DataFrame(data)


class NetworkGraph(BasePlot):
    """Custom graph class based on NetworkX's ``Graph``.

    Methods
    -------
    compute_positions
        Return node positions computed with a spring layout.
    layout
        Return scaled node sizes, edge widths, and grouped font sizes.
    aplot
        Plot the graph on a provided axis.
    fplot
        Plot the graph and return a new figure.
    aplot_connected_components
        Plot each connected component on a shared axis.
    fplot_connected_components
        Plot each connected component on a new figure.
    get_component_subgraph
        Return the subgraph containing the specified node.
    k_core
        Return the k-core of the graph.
    get_core_subgraph
        Return the 2-core of the graph.
    top_k_edges
        Return the top edges for each node based on an attribute.
    calculate_node_weights_from_edges
        Populate node weights by summing top edge weights.
    trim_edges
        Create a subgraph that retains the top edges per node.
    set_node_attributes
        Set multiple node attributes from a mapping.
    from_pandas_edgelist
        Build a graph from a pandas edge list.
    build_from_dataframes
        Construct a graph from node and edge DataFrames with validation.
    """

    _nx_graph: nx.Graph

    def __init__(
        self,
        pd_df: Optional[pd.DataFrame] = None,
        nx_graph: Optional[nx.Graph] = None,
        source: str = "source",
        target: str = "target",
        weight: str = "weight",
    ):
        """Initialize with an existing NetworkX graph.

        Parameters
        ----------
        nx_graph : nx.Graph
            Graph to wrap.
        """
        self._weight_column = weight
        if isinstance(pd_df, nx.Graph) and nx_graph is None:
            nx_graph = pd_df
            pd_df = None

        if nx_graph is not None:
            self._nx_graph = nx_graph
        elif pd_df is not None:
            self._nx_graph = NetworkGraph.from_pandas_edgelist(
                pd_df, source=source, target=target, edge_weight_col=weight
            )._nx_graph
        else:
            self._nx_graph = nx.Graph()

        self._scale = 1.0

    @property
    def scale(self) -> float:
        """Return scaling factor for plotting sizes."""
        return self._scale

    @scale.setter
    def scale(self, value: float):
        """Set scaling factor for plotting sizes.

        Parameters
        ----------
        value : float
            Scaling factor.
        """
        self._scale = value

    @property
    def node_view(self) -> NodeView:
        """Return a ``NodeView`` over the graph."""
        return NodeView(self._nx_graph)

    @node_view.setter
    def node_view(self, node: Any, attributes: Dict[str, Any]):
        """Set attributes for a specific node.

        Parameters
        ----------
        node : Any
            Node identifier.
        attributes : dict
            Attributes to set for the node.
        """
        self._nx_graph.nodes[node].update(attributes)

    @property
    def edge_view(self) -> EdgeView:
        """Return an ``EdgeView`` over the graph."""
        return EdgeView(self._nx_graph)

    @edge_view.setter
    def edge_view(self, edge: Tuple[Any, Any], attributes: Dict[str, Any]):
        """Set attributes for a specific edge.

        Parameters
        ----------
        edge : tuple[Any, Any]
            Edge defined by source and target node identifiers.
        attributes : dict
            Attributes to set for the edge.
        """
        u, v = edge
        self._nx_graph.edges[u, v].update(attributes)

    @property
    def adjacency_view(self) -> AdjacencyView:
        """Return an ``AdjacencyView`` of the graph."""
        return AdjacencyView(self._nx_graph.adj)

    @adjacency_view.setter
    def adjacency_view(self, node: Any, attributes: Dict[str, Any]):
        """Set attributes for a specific node.

        Parameters
        ----------
        node : Any
            Node identifier.
        attributes : dict
            Attributes to set for the node.
        """
        self._nx_graph.adj[node].update(attributes)

    @property
    def connected_components(self) -> List[set]:
        """Return the connected components of the graph."""
        return list(nx.connected_components(self._nx_graph))

    @property
    def number_of_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        return self._nx_graph.number_of_nodes()

    @property
    def number_of_edges(self) -> int:
        """Return the number of edges in the graph."""
        return self._nx_graph.number_of_edges()

    @property
    def density(self) -> float:
        """Return the density of the graph."""
        return nx.density(self._nx_graph)

    @property
    def is_connected(self) -> bool:
        """Return whether the graph is connected."""
        return nx.is_connected(self._nx_graph)

    @property
    def average_clustering(self) -> float:
        """Return the average clustering coefficient of the graph."""
        return nx.average_clustering(self._nx_graph)

    @property
    def diameter(self) -> int:
        """Return the diameter of the graph."""
        return nx.diameter(self._nx_graph)

    @property
    def radius(self) -> int:
        """Return the radius of the graph."""
        return nx.radius(self._nx_graph)

    @property
    def center(self) -> List[Any]:
        """Return the center nodes of the graph."""
        return nx.center(self._nx_graph)

    @property
    def periphery(self) -> List[Any]:
        """Return the periphery nodes of the graph."""
        return nx.periphery(self._nx_graph)

    @property
    def average_shortest_path_length(self) -> float:
        """Return the average shortest path length of the graph."""
        return nx.average_shortest_path_length(self._nx_graph)

    @property
    def transitivity(self) -> float:
        """Return the transitivity of the graph."""
        return nx.transitivity(self._nx_graph)

    @property
    def clustering_coefficients(self) -> Dict[Any, float]:
        """Return the clustering coefficients of the graph."""
        return nx.clustering(self._nx_graph)  # pyright: ignore[reportReturnType]

    @property
    def degree_assortativity_coefficient(self) -> float:
        """Return the degree assortativity coefficient of the graph."""
        return nx.degree_assortativity_coefficient(self._nx_graph)

    def add_node(self, node: Any, **attributes: Any):
        """Add a node with optional attributes.

        Parameters
        ----------
        node : Any
            Node identifier.
        **attributes : dict
            Arbitrary node attributes as keyword arguments.
        """
        self._nx_graph.add_node(node, **attributes)

    def add_nodes_from(self, nodes: Iterable, **attributes: Any):
        """Add multiple nodes with optional attributes.

        Parameters
        ----------
        nodes : Iterable
            Node identifiers to add.
        **attributes : dict
            Arbitrary node attributes as keyword arguments.
        """
        self._nx_graph.add_nodes_from(nodes, **attributes)

    def add_edge(self, source: Any, target: Any, **attributes: Any):
        """Add an edge with optional attributes.

        Parameters
        ----------
        source : Any
            Source node identifier.
        target : Any
            Target node identifier.
        **attributes : dict
            Arbitrary edge attributes as keyword arguments.
        """
        self._nx_graph.add_edge(source, target, **attributes)

    def add_edges_from(self, edges: Iterable[Tuple[Any, Any]], **attributes: Any):
        """Add multiple edges with optional attributes.

        Parameters
        ----------
        edges : Iterable[tuple[Any, Any]]
            Edge tuples defined by source and target node identifiers.
        **attributes : dict
            Arbitrary edge attributes as keyword arguments.
        """
        self._nx_graph.add_edges_from(edges, **attributes)

    def subgraph_edges(self, edges: Iterable) -> "NetworkGraph":
        """Return a subgraph containing only the specified edges.

        Parameters
        ----------
        edges : Iterable
            Edges to include.

        Returns
        -------
        NetworkGraph
            Subgraph with only ``edges``.
        """
        return NetworkGraph(nx.edge_subgraph(self._nx_graph, edges).copy())

    def layout(
        self,
        max_node_size: int = _DEFAULT["MAX_NODE_SIZE"],
        min_node_size: int = _DEFAULT["MIN_NODE_SIZE"],
        max_edge_width: int = _DEFAULT["MAX_EDGE_WIDTH"],
        max_font_size: int = _DEFAULT["MAX_FONT_SIZE"],
        min_font_size: int = _DEFAULT["MIN_FONT_SIZE"],
        edge_weight_col: str = "weight",
        *,
        node_deciles: Optional[np.ndarray],
        edge_deciles: Optional[np.ndarray],
    ) -> Tuple[List[float], List[float], Dict[int, List[str]]]:
        """Calculate node, edge and font sizes based on weights.

        Parameters
        ----------
        max_node_size : int, optional
            Upper bound for node size. The default is `_DEFAULT["MAX_NODE_SIZE"]`.
        min_node_size : int, optional
            Lower bound for node size. The default is `_DEFAULT["MIN_NODE_SIZE"]`.
        max_edge_width : int, optional
            Upper bound for edge width. The default is `_DEFAULT["MAX_EDGE_WIDTH"]`.
        max_font_size : int, optional
            Upper bound for font size. The default is `_DEFAULT["MAX_FONT_SIZE"]`.
        min_font_size : int, optional
            Lower bound for font size. The default is `_DEFAULT["MIN_FONT_SIZE"]`.
        edge_weight_col : str, optional
            Edge attribute used for weighting. The default is "weight".
        node_deciles : np.ndarray, optional
            Node-weight deciles used to scale node and font sizes.
        edge_deciles : np.ndarray, optional
            Edge-weight deciles used to scale edge widths.

        Returns
        -------
        tuple[list[float], list[float], dict[int, list[str]]]
            Node sizes, edge widths and nodes grouped by font size.
        """
        # Normalize and scale nodes' weights within the desired range of edge widths
        node_weights = [
            data.get(edge_weight_col, 1) for node, data in self.node_view(data=True)
        ]
        node_size = _scale_weights(
            weights=node_weights,
            scale_max=max_node_size,
            scale_min=min_node_size,
            deciles=node_deciles,
        )

        # Normalize and scale edges' weights within the desired range of edge widths
        edge_weights = [
            data.get(edge_weight_col, 1) for _, _, data in self.edge_view(data=True)
        ]
        edges_width = _scale_weights(
            weights=edge_weights,
            scale_max=max_edge_width,
            deciles=edge_deciles,
        )

        # Scale the normalized node weights within the desired range of font sizes
        node_size_dict = dict(
            zip(
                self.node_view,
                _scale_weights(
                    weights=node_weights,
                    scale_max=max_font_size,
                    scale_min=min_font_size,
                    deciles=node_deciles,
                ),
            )
        )
        fonts_size = defaultdict(list)
        for node, width in node_size_dict.items():
            fonts_size[int(width)].append(node)
        fonts_size = dict(fonts_size)

        return node_size, edges_width, fonts_size

    def subgraph(
        self,
        node_list: Optional[List[str]] = None,
        max_edges: int = _DEFAULT["MAX_EDGES"],
        min_degree: int = 2,
        top_k_edges_per_node: int = 5,
    ) -> "NetworkGraph":
        """Return a trimmed subgraph limited by nodes and edges.

        Parameters
        ----------
        node_list : list[str], optional
            Nodes to include.
        max_edges : int, optional
            Maximum edges to retain. The default is `_DEFAULT["MAX_EDGES"]`.
        min_degree : int, optional
            Minimum degree for nodes in the core subgraph. The default is 2.
        top_k_edges_per_node : int, optional
            Number of top edges to keep per node. The default is 5.

        Returns
        -------
        NetworkGraph
            Trimmed subgraph.
        """
        if node_list is None:
            node_list = self.node_view.sort("weight")[: _DEFAULT["MAX_NODES"]]
        core_subgraph_nodes = list(self.k_core(k=min_degree).node_view)
        node_list = [node for node in node_list if node in core_subgraph_nodes]
        _subgraph = nx.subgraph(self._nx_graph, node_list)
        subgraph = NetworkGraph(_subgraph)
        if top_k_edges_per_node > 0:
            edges = subgraph.top_k_edges(
                attribute="weight", k=top_k_edges_per_node
            ).keys()
            subgraph = subgraph.subgraph_edges(list(edges)[:max_edges])
        return subgraph

    def compute_positions(
        self,
        k: Optional[float] = None,
        seed: Optional[int] = _DEFAULT["SPRING_LAYOUT_SEED"],
    ) -> Dict[Any, np.ndarray]:
        """Return spring layout positions for the graph.

        Parameters
        ----------
        k : float, optional
            Optimal distance between nodes. The default is ``_DEFAULT["SPRING_LAYOUT_K"]``.
        seed : int, optional
            Seed for reproducible layouts. The default is ``_DEFAULT["SPRING_LAYOUT_SEED"]``.

        Returns
        -------
        dict[Any, np.ndarray]
            Mapping of nodes to their layout coordinates.
        """
        layout_k = _DEFAULT["SPRING_LAYOUT_K"] if k is None else k
        return nx.spring_layout(self._nx_graph, k=layout_k, seed=seed)

    def subgraph_component(self, node: Any) -> "NetworkGraph":
        """Return the connected component containing ``node``.

        Parameters
        ----------
        node : Any
            Node identifier to anchor the component selection.

        Returns
        -------
        NetworkGraph
            Subgraph made of the nodes in the same connected component as
            ``node``.

        Raises
        ------
        ValueError
            If ``node`` is not present in the graph.
        """
        if node not in self._nx_graph:
            raise ValueError(f"Node {node!r} is not present in the graph.")

        component_nodes = next(
            (
                component
                for component in nx.connected_components(self._nx_graph)
                if node in component
            ),
            None,
        )

        if component_nodes is None:
            return NetworkGraph()

        return NetworkGraph(nx.subgraph(self._nx_graph, component_nodes).copy())

    def aplot(
        self,
        title: Optional[str] = None,
        style: Optional[StyleTemplate] = None,
        edge_weight_col: str = "weight",
        layout_seed: Optional[int] = _DEFAULT["SPRING_LAYOUT_SEED"],
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot the graph using node and edge weights.

        Parameters
        ----------
        title : str, optional
            Plot title.
        style : StyleTemplate, optional
            Style configuration. The default is `NETWORK_STYLE_TEMPLATE`.
        edge_weight_col : str, optional
            Edge attribute used for weighting. The default is "weight".
        layout_seed : int, optional
            Seed for the spring layout used to place nodes. The default is ``_DEFAULT["SPRING_LAYOUT_SEED"]``.
        ax : Axes, optional
            Axes to draw on.

        Returns
        -------
        Axes
            Matplotlib axes with the plotted network.
        """
        if not style:
            style = NETWORK_STYLE_TEMPLATE
        sns.set_palette(style.palette)
        if ax is None:
            ax = cast(Axes, plt.gca())

        isolated_nodes = list(nx.isolates(self._nx_graph))
        graph_nx = self._nx_graph
        if isolated_nodes:

            graph_nx = graph_nx.copy()
            graph_nx.remove_nodes_from(isolated_nodes)

        graph = self if graph_nx is self._nx_graph else NetworkGraph(nx_graph=graph_nx)

        if graph._nx_graph.number_of_nodes() == 0:
            ax.set_axis_off()
            if title:
                ax.set_title(
                    title,
                    color=style.font_color,
                    fontsize=style.font_size * TITLE_SCALE_FACTOR,
                )
            return ax

        mapped_min_font_size = style.font_mapping.get(0)
        mapped_max_font_size = style.font_mapping.get(4)
        node_weights = [
            data.get(edge_weight_col, 1) for _, data in graph.node_view(data=True)
        ]
        edge_weights = [
            data.get(edge_weight_col, 1) for _, _, data in graph.edge_view(data=True)
        ]
        node_deciles = _compute_deciles(node_weights)
        edge_deciles = _compute_deciles(edge_weights)

        node_sizes, edge_widths, font_sizes = graph.layout(
            min_node_size=_DEFAULT["MIN_NODE_SIZE"],
            max_node_size=_DEFAULT["MAX_NODE_SIZE"],
            max_edge_width=_DEFAULT["MAX_EDGE_WIDTH"],
            min_font_size=(
                mapped_min_font_size
                if mapped_min_font_size is not None
                else _DEFAULT["MIN_FONT_SIZE"]
            ),
            max_font_size=(
                mapped_max_font_size
                if mapped_max_font_size is not None
                else _DEFAULT["MAX_FONT_SIZE"]
            ),
            edge_weight_col=edge_weight_col,
            node_deciles=node_deciles,
            edge_deciles=edge_deciles,
        )
        pos = graph.compute_positions(seed=layout_seed)
        # nodes
        node_sizes_int = [int(size) for size in node_sizes]
        nx.draw_networkx_nodes(
            graph._nx_graph,
            pos,
            ax=ax,
            node_size=cast(Any, node_sizes_int),
            node_color=cast(Any, node_sizes),
            cmap=plt.get_cmap(style.palette),
        )
        # edges
        nx.draw_networkx_edges(
            graph._nx_graph,
            pos,
            ax=ax,
            edge_color=style.font_color,
            edge_cmap=plt.get_cmap(style.palette),
            width=cast(Any, edge_widths),
        )
        # labels
        for font_size, nodes in font_sizes.items():
            nx.draw_networkx_labels(
                graph._nx_graph,
                pos,
                ax=ax,
                font_size=font_size,
                font_color=style.font_color,
                labels={n: string_formatter(n) for n in nodes},
            )
        ax.set_facecolor(style.background_color)
        if title:
            ax.set_title(
                title,
                color=style.font_color,
                fontsize=style.font_size * TITLE_SCALE_FACTOR,
            )
        ax.set_axis_off()

        return ax

    def aplot_connected_components(
        self,
        title: Optional[str] = None,
        style: Optional[StyleTemplate] = None,
        edge_weight_col: str = "weight",
        layout_seed: Optional[int] = _DEFAULT["SPRING_LAYOUT_SEED"],
        axes: Optional[np.ndarray] = None,
    ) -> Union[Axes, np.ndarray]:
        """Plot all connected components of the graph.

        Parameters
        ----------
        title : str, optional
            Base title for component subplots. When provided, each axis title is
            suffixed with the component index.
        style : StyleTemplate, optional
            Style configuration. The default is `NETWORK_STYLE_TEMPLATE`.
        edge_weight_col : str, optional
            Edge attribute used for weighting. The default is "weight".
        layout_seed : int, optional
            Seed for the spring layout used to place nodes. The default is ``_DEFAULT["SPRING_LAYOUT_SEED"]``.
        axes : np.ndarray, optional
            Existing axes to draw each component on. If None, a grid is created
            based on the number of components.

        Returns
        -------
        Union[Axes, np.ndarray]
            Matplotlib axes with the plotted network. When ``axes`` is provided
            or created, the flattened array of axes is returned; otherwise, a
            single Axes is returned.
        """
        if not style:
            style = NETWORK_STYLE_TEMPLATE
        sns.set_palette(style.palette)

        graph = self
        isolated_nodes = list(nx.isolates(self._nx_graph))
        if isolated_nodes:
            graph = NetworkGraph(nx_graph=self._nx_graph.copy())
            graph._nx_graph.remove_nodes_from(isolated_nodes)

        connected_components = list(nx.connected_components(graph._nx_graph))

        local_axes = axes
        created_axes = False

        if not connected_components:
            if local_axes is None:
                local_axes = np.array([cast(Axes, plt.gca())])
            for axis in local_axes.flatten():
                axis.set_facecolor(style.background_color)
                axis.set_axis_off()
            return local_axes

        if local_axes is None:
            _, local_axes = _compute_network_grid(connected_components, style)
            created_axes = True

        for i, component in enumerate(connected_components):
            if i >= len(local_axes):
                break
            component_graph = NetworkGraph(
                nx.subgraph(graph._nx_graph, component).copy()
            )
            component_graph.aplot(
                title=f"{title}::{i}" if title else str(i),
                style=style,
                edge_weight_col=edge_weight_col,
                layout_seed=layout_seed,
                ax=cast(Axes, local_axes[i]),
            )
            cast(Axes, local_axes[i]).set_axis_on()

        for axis in local_axes[len(connected_components) :]:
            axis.set_axis_off()

        return local_axes if created_axes or len(local_axes) > 1 else local_axes[0]

    def fplot_connected_components(
        self,
        title: Optional[str] = None,
        style: Optional[StyleTemplate] = None,
        edge_weight_col: str = "weight",
        layout_seed: Optional[int] = _DEFAULT["SPRING_LAYOUT_SEED"],
        figsize: Tuple[float, float] = FIG_SIZE,
    ) -> Figure:
        """Plot all connected components of the graph.

        Parameters
        ----------
        title : str, optional
            Plot title to apply to the first component axis.
        style : StyleTemplate, optional
            Style configuration. The default is `NETWORK_STYLE_TEMPLATE`.
        edge_weight_col : str, optional
            Edge attribute used for weighting. The default is "weight".
        layout_seed : int, optional
            Seed for the spring layout used to place nodes. The default is ``_DEFAULT["SPRING_LAYOUT_SEED"]``.

        Returns
        -------
        Figure
            Matplotlib figure with the plotted network.
        """
        if not style:
            style = NETWORK_STYLE_TEMPLATE

        fig, ax = BasePlot.create_fig(figsize=figsize, style=style)

        self.aplot_connected_components(
            title=title,
            style=style,
            edge_weight_col=edge_weight_col,
            layout_seed=layout_seed,
            axes=np.array([ax]),
        )
        return fig

    def k_core(self, k: int = 2) -> "NetworkGraph":
        """Return the k-core of the graph.

        The k-core is a subgraph containing only nodes with degree >= k.

        Parameters
        ----------
        k : int, optional
            The minimum degree for nodes in the core. The default is 2.

        Returns
        -------
        NetworkGraph
            The k-core subgraph.
        """
        core_graph = nx.k_core(self._nx_graph, k=k)
        return NetworkGraph(core_graph)

    def subgraph_core(self) -> "NetworkGraph":
        """Return the 2-core of the graph.

        Returns
        -------
        NetworkGraph
            The k-core subgraph with minimum degree 2.
        """
        return self.k_core(k=2)

    def top_k_edges(
        self, attribute: str, reverse: bool = True, k: int = 5
    ) -> Dict[Any, List[Tuple[Any, Dict]]]:
        """Return the top ``k`` edges based on a given attribute.

        Parameters
        ----------
        attribute : str
            Attribute name used for sorting.
        reverse : bool, optional
            Whether to sort in descending order. The default is `True`.
        k : int, optional
            Number of top edges to return. The default is 5.

        Returns
        -------
        dict[Any, list[tuple[Any, dict]]]
            Mapping of edge tuples to attribute values.
        """
        top_list = {}
        for node in self.node_view:
            edges = self.edge_view(node, data=True)
            edges_sorted = sorted(
                edges, key=lambda x: x[2].get(attribute, 0), reverse=reverse
            )
            top_k_edges = edges_sorted[:k]
            for u, v, data in top_k_edges:
                edge_key = (u, v)
                top_list[edge_key] = data[attribute]
        return top_list

    def calculate_nodes(
        self,
        edge_weight_col: str = "weight",
        k: int = 10,
    ):
        """Calculate node weights by summing weights of top k edges.

        Parameters
        ----------
        edge_weight_col : str, optional
            Edge attribute to use for weighting. The default is "weight".
        k : int, optional
            Number of top edges to consider for each node. The default is 10.
        """
        top_edges = self.top_k_edges(attribute=edge_weight_col, k=k)
        attributes: dict[Any, dict[str, Any]] = {}
        for (u, v), weight_value in top_edges.items():
            for node in [u, v]:
                if node not in attributes:
                    attributes[node] = {edge_weight_col: 0, "edges": 0}
                attributes[node][edge_weight_col] += weight_value
                attributes[node]["edges"] += 1

        self.set_node_attributes(attributes=attributes)

    def trim_edges(
        self, edge_weight_col: str = "weight", k: int = 10
    ) -> "NetworkGraph":
        """Trim the graph to keep only the top k edges per node.

        Parameters
        ----------
        edge_weight_col : str, optional
            Edge attribute to use for sorting. The default is "weight".
        top_k_per_node : int, optional
            Number of top edges to keep per node. The default is 5.

        Returns
        -------
        NetworkGraph
            A new graph containing only the top edges.
        """
        edges_to_keep = self.top_k_edges(attribute=edge_weight_col, k=k)
        network_X = self.subgraph_edges(edges=edges_to_keep)
        network_X.sanitize_network()
        return network_X

    def set_node_attributes(self, attributes: Dict[Any, Dict[str, Any]]):
        """Set multiple node attributes from a dictionary.

        Parameters
        ----------
        attributes : Dict[Any, Dict[str, Any]]
            Mapping of node identifiers to their attribute dictionaries.
        """
        for node, attrs in attributes.items():
            nx.set_node_attributes(self._nx_graph, {node: attrs})

    @staticmethod
    def from_pandas_edgelist(
        edges_df: pd.DataFrame,
        source: str = "source",
        target: str = "target",
        edge_weight_col: str = "weight",
        k: int = 10,
    ) -> "NetworkGraph":
        """Initialize a NetworkGraph from a simple DataFrame.

        Parameters
        ----------
        edges_df : pd.DataFrame
            DataFrame containing edge data.
        source : str, optional
            Column name for source nodes. The default is "source".
        target : str, optional
            Column name for target nodes. The default is "target".
        edge_weight_col : str, optional
            Column name for edge weights. The default is "weight".

        Returns
        -------
        NetworkGraph
            Initialized network graph.
        """
        validate_dataframe(edges_df, cols=[source, target, edge_weight_col])
        network_Z = NetworkGraph()
        for src, dst, weight in edges_df[[source, target, edge_weight_col]].itertuples(
            index=False, name=None
        ):
            network_Z.add_edge(src, dst, **{edge_weight_col: weight})

        network_Z.calculate_nodes(edge_weight_col=edge_weight_col, k=k)
        network_Z.sanitize_network()
        return network_Z

    def sanitize_network(
        self,
    ):
        """Remove inconsistent nodes and edges to ensure graph consistency."""
        if len(self.node_view) == 0:
            return

        nodes = set(self.node_view)
        edges = list(self.edge_view())
        nodes_in_edges = {endpoint for edge in edges for endpoint in edge}

        for node in list(nodes):
            if node not in nodes_in_edges:
                self._nx_graph.remove_node(node)

        for u, v in list(self.edge_view()):
            if u not in self._nx_graph or v not in self._nx_graph:
                self._nx_graph.remove_edge(u, v)

    @staticmethod
    def sanitize_node_df(
        node_df: pd.DataFrame,
        edge_df: pd.DataFrame,
        node_col: str = "node",
        node_weight_col: str = "weight",
        edge_source_col: str = "source",
        edge_target_col: str = "target",
        edge_weight_col: str = "weight",
    ) -> pd.DataFrame:
        """Private helper returning ``node_df`` rows present in the edge list.

        This method supports internal builders and is not part of the public API.

        Parameters
        ----------
        node_df : pd.DataFrame
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
            Column name for edge weights. The default is "weight". Included for
            signature parity with other sanitization helpers.

        Returns
        -------
        pd.DataFrame
            Filtered ``node_df`` with only nodes that appear as sources or targets.
        """
        validate_dataframe(node_df, cols=[node_col, node_weight_col])
        filtered_node_df = node_df.copy()
        nodes_in_edges = list(
            set(edge_df[edge_source_col]).union(edge_df[edge_target_col])
        )
        return filtered_node_df.loc[filtered_node_df[node_col].isin(nodes_in_edges)]

    @staticmethod
    def sanitize_edge_df(
        node_df: pd.DataFrame,
        edge_df: pd.DataFrame,
        node_col: str = "node",
        node_weight_col: str = "weight",
        edge_source_col: str = "source",
        edge_target_col: str = "target",
        edge_weight_col: str = "weight",
    ) -> pd.DataFrame:
        """Private helper returning a sanitized copy of the edge DataFrame.

        Intended for internal validation when building graphs from dataframes.

        Parameters
        ----------
        node_df : pd.DataFrame
            DataFrame containing node identifiers and weights.
        edge_df : pd.DataFrame
            Edge DataFrame containing source and target columns.
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

        Returns
        -------
        pd.DataFrame
            Sanitized edge DataFrame containing only edges whose nodes appear
            in ``node_df``.
        """
        validate_dataframe(
            edge_df, cols=[edge_source_col, edge_target_col, edge_weight_col]
        )
        validate_dataframe(node_df, cols=[node_col, node_weight_col])
        allowed_nodes = node_df[node_col].tolist()
        edge_df = edge_df.loc[
            edge_df[edge_source_col].isin(allowed_nodes)
            & edge_df[edge_target_col].isin(allowed_nodes)
        ]
        return edge_df

    @staticmethod
    def sanitize_dfs(
        node_df: pd.DataFrame,
        edge_df: pd.DataFrame,
        node_col: str = "node",
        node_weight_col: str = "weight",
        edge_source_col: str = "source",
        edge_target_col: str = "target",
        edge_weight_col: str = "weight",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return sanitized node and edge DataFrames.

        Parameters
        ----------
        node_df : pd.DataFrame
            DataFrame containing node identifiers and weights.
        edge_df : pd.DataFrame
            Edge DataFrame containing source, target, and weights.
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

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the sanitized node and edge DataFrames.
        """
        node_df = NetworkGraph.sanitize_node_df(
            node_df,
            edge_df=edge_df,
            node_col=node_col,
            node_weight_col=node_weight_col,
            edge_source_col=edge_source_col,
            edge_target_col=edge_target_col,
            edge_weight_col=edge_weight_col,
        )
        edge_df = NetworkGraph.sanitize_edge_df(
            node_df,
            edge_df=edge_df,
            node_col=node_col,
            node_weight_col=node_weight_col,
            edge_source_col=edge_source_col,
            edge_target_col=edge_target_col,
            edge_weight_col=edge_weight_col,
        )
        return node_df, edge_df

    @staticmethod
    def from_pandas(
        node_df: pd.DataFrame,
        edge_df: pd.DataFrame,
        node_col: str = "node",
        node_weight_col: str = "weight",
        edge_source_col: str = "source",
        edge_target_col: str = "target",
        edge_weight_col: str = "weight",
    ) -> "NetworkGraph":
        """Build a NetworkGraph from node and edge DataFrames.

        Parameters
        ----------
        node_df : pd.DataFrame
            DataFrame containing node identifiers and weights.
        edge_df : pd.DataFrame
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

        Returns
        -------
        NetworkGraph
            Prepared ``NetworkGraph`` instance with node weights set and edges
            filtered to nodes present in ``node_df``.
        """
        if node_df is not None:
            node_df, edge_df = NetworkGraph.sanitize_dfs(
                node_df,
                edge_df,
                node_col=node_col,
                node_weight_col=node_weight_col,
                edge_source_col=edge_source_col,
                edge_target_col=edge_target_col,
                edge_weight_col=edge_weight_col,
            )
        graph = NetworkGraph.from_pandas_edgelist(
            edge_df,
            source=edge_source_col,
            target=edge_target_col,
            edge_weight_col=edge_weight_col,
        )
        if node_df is None or node_df.empty:
            graph.calculate_nodes(edge_weight_col=edge_weight_col)
        else:
            node_weights = {
                node: {node_weight_col: weight_value}
                for node, weight_value in node_df.set_index(node_col)[
                    node_weight_col
                ].items()
                if node in graph._nx_graph.nodes
            }
            graph.set_node_attributes(node_weights)

        return graph


def _compute_network_grid(
    connected_components: List[set], style: StyleTemplate
) -> Tuple[Figure, np.ndarray]:
    """Compute the grid layout for network component subplots.

    Parameters
    ----------
    connected_components : list[set]
        A list of sets, where each set contains the nodes of a connected component.
    style : StyleTemplate
        The style template used for plotting.

    Returns
    -------
    Tuple[Figure, np.ndarray]
        A tuple containing the Matplotlib figure and the grid of axes.
    """
    n_components = len(connected_components)
    n_cols = int(np.ceil(np.sqrt(n_components)))
    n_rows = int(np.ceil(n_components / n_cols))
    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=FIG_SIZE)
    fig = cast(Figure, fig)
    fig.set_facecolor(style.background_color)
    if not isinstance(axes_grid, np.ndarray):
        axes = np.array([axes_grid])
    else:
        axes = axes_grid.flatten()
    return fig, axes
