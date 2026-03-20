"""Tests for network visualizations."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from MatplotLibAPI.network import (
    NetworkGraph,
    _WEIGHT_PERCENTILES,
    fplot_network,
    aplot_network_node,
    fplot_network_node,
    fplot_network_components,
    _scale_weights,
    _softmax,
)
from MatplotLibAPI.style_template import StyleTemplate, TITLE_SCALE_FACTOR


def test_fplot_network(load_sample_df):
    """Render a network figure from sample data."""

    df = load_sample_df("network.csv")

    fig = fplot_network(
        pd_df=df,
        edge_source_col="city_a",
        edge_target_col="city_b",
        edge_weight_col="distance_km",
    )

    assert isinstance(fig, Figure)


def test_fplot_network_returns_pyplot_managed_figure(load_sample_df):
    """Return a figure that supports direct ``fig.show()`` calls."""

    df = load_sample_df("network.csv")

    fig = fplot_network(
        pd_df=df,
        edge_source_col="city_a",
        edge_target_col="city_b",
        edge_weight_col="distance_km",
    )

    assert fig.canvas.manager is not None
    fig.show()


def test_network_graph_fplot_returns_pyplot_managed_figure(load_sample_df):
    """Return a pyplot-managed figure when calling ``NetworkGraph.fplot``."""

    df = load_sample_df("network.csv")
    graph = NetworkGraph.from_pandas_edgelist(
        df,
        source="city_a",
        target="city_b",
        edge_weight_col="distance_km",
    )

    fig = graph.fplot(edge_weight_col="distance_km")

    assert isinstance(fig, Figure)
    assert fig.canvas.manager is not None
    fig.show()


def test_accessor_fplot_network_components(load_sample_df):
    """Render network component figures via the pandas accessor."""

    df = load_sample_df("network.csv")

    fig = df.mpl.fplot_network_components(
        edge_source_col="city_a",
        edge_target_col="city_b",
        edge_weight_col="distance_km",
    )

    assert isinstance(fig, Figure)


def test_softmax_matches_expected_probabilities():
    """Return softmax probabilities consistent with NumPy operations."""

    values = [0.0, 1.0, 2.0]

    result = _softmax(values)
    expected = np.exp(np.array(values) - np.max(values))
    expected = expected / expected.sum()

    assert np.allclose(result, expected)


def test_scale_weights_empty_handles_gracefully():
    """Return an empty list when no weights are provided."""

    assert _scale_weights([]) == []


def test_layout_handles_graph_without_edges():
    """Compute layout without raising for graphs lacking edges."""

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(["a", "b"])
    graph = NetworkGraph(nx_graph=nx_graph)

    node_sizes, edge_widths, fonts_size = graph.layout()

    assert len(node_sizes) == graph.number_of_nodes
    assert edge_widths == []
    assert fonts_size


def test_network_components_title_respects_scale_factor(load_sample_df):
    """Ensure component plot titles honor the shared title scale factor."""

    df = load_sample_df("network.csv")
    style = StyleTemplate(font_size=11)

    fig = fplot_network_components(
        pd_df=df,
        edge_source_col="city_a",
        edge_target_col="city_b",
        edge_weight_col="distance_km",
        title="Component Title",
        style=style,
    )

    suptitle = getattr(fig, "_suptitle", None)
    expected_size = style.font_size * TITLE_SCALE_FACTOR * 1.25

    assert suptitle is not None
    assert suptitle.get_fontsize() == expected_size


def test_scale_weights_respects_precomputed_deciles():
    """Reuse provided deciles to produce consistent scaling across calls."""

    weights = [1.0, 2.0, 3.0, 4.0]
    deciles = np.percentile(np.array(weights), _WEIGHT_PERCENTILES)

    expected = _scale_weights(weights)
    assert _scale_weights(weights, deciles=deciles) == expected


def test_network_layout_respects_precomputed_deciles() -> None:
    """Reuse provided deciles to produce stable layout scaling."""

    graph = NetworkGraph()
    graph.add_edge("a", "b", weight=1.0)
    graph.add_edge("a", "c", weight=5.0)
    graph.add_edge("b", "c", weight=10.0)
    graph.calculate_nodes(edge_weight_col="weight", k=10)

    node_weights = [data.get("weight", 1) for _, data in graph.node_view(data=True)]
    edge_weights = [data.get("weight", 1) for _, _, data in graph.edge_view(data=True)]
    node_deciles = np.percentile(np.array(node_weights), _WEIGHT_PERCENTILES)
    edge_deciles = np.percentile(np.array(edge_weights), _WEIGHT_PERCENTILES)

    expected = graph.layout(edge_weight_col="weight")
    actual = graph.layout(
        edge_weight_col="weight",
        node_deciles=node_deciles,
        edge_deciles=edge_deciles,
    )
    assert actual == expected


def test_compute_positions_is_reproducible_with_seed():
    """Return identical layouts when seeded and varied layouts when not."""

    nx_graph = nx.Graph()
    nx_graph.add_edges_from([(1, 2), (2, 3), (3, 1)])
    graph = NetworkGraph(nx_graph=nx_graph)

    seeded_positions = graph.compute_positions(seed=7)
    reseeded_positions = graph.compute_positions(seed=7)
    different_seed_positions = graph.compute_positions(seed=8)

    for node in seeded_positions:
        assert np.allclose(seeded_positions[node], reseeded_positions[node])

    assert any(
        not np.allclose(seeded_positions[node], different_seed_positions[node])
        for node in seeded_positions
    )


def test_aplot_network_node_limits_to_component(monkeypatch):
    """Plot only the component containing the requested node."""

    df = pd.DataFrame(
        {
            "source": ["a", "b", "c", "x", "y", "z"],
            "target": ["b", "c", "a", "y", "z", "x"],
            "weight": [1, 1, 1, 2, 2, 2],
        }
    )

    captured_nodes = []

    def fake_plot(self, *args, **kwargs):  # type: ignore[override]
        captured_nodes.append(set(self._nx_graph.nodes))
        return plt.gca()

    monkeypatch.setattr(NetworkGraph, "aplot", fake_plot)

    aplot_network_node(df, node="a")

    assert captured_nodes and captured_nodes[0] == {"a", "b", "c"}


def test_fplot_network_node_returns_figure(load_sample_df):
    """Return a Matplotlib figure when plotting a node component."""

    df = load_sample_df("network.csv")

    fig = fplot_network_node(
        df,
        node="New York",
        edge_source_col="city_a",
        edge_target_col="city_b",
        edge_weight_col="distance_km",
    )

    assert isinstance(fig, Figure)


def test_aplot_network_node_raises_for_missing_node(load_sample_df):
    """Raise a ValueError when the requested node is absent from the graph."""

    df = load_sample_df("network.csv")

    with pytest.raises(ValueError):
        aplot_network_node(
            df,
            node="Atlantis",
            edge_source_col="city_a",
            edge_target_col="city_b",
            edge_weight_col="distance_km",
        )
