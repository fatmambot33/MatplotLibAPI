"""Tests for network visualizations."""

from matplotlib.figure import Figure
import networkx as nx
import numpy as np

import MatplotLibAPI
from MatplotLibAPI.Network import (
    NetworkGraph,
    WEIGHT_PERCENTILES,
    scale_weights,
    softmax,
)
from MatplotLibAPI.StyleTemplate import StyleTemplate, TITLE_SCALE_FACTOR


def test_fplot_network(load_sample_df):
    """Render a network figure from sample data."""

    df = load_sample_df("network.csv")

    fig = MatplotLibAPI.fplot_network(
        pd_df=df, source="city_a", target="city_b", weight="distance_km"
    )

    assert isinstance(fig, Figure)


def test_accessor_fplot_network_components(load_sample_df):
    """Render network component figures via the pandas accessor."""

    df = load_sample_df("network.csv")

    fig = df.mpl.fplot_network_components(
        source="city_a", target="city_b", weight="distance_km"
    )

    assert isinstance(fig, Figure)


def test_softmax_matches_expected_probabilities():
    """Return softmax probabilities consistent with NumPy operations."""

    values = [0.0, 1.0, 2.0]

    result = softmax(values)
    expected = np.exp(np.array(values) - np.max(values))
    expected = expected / expected.sum()

    assert np.allclose(result, expected)


def test_scale_weights_empty_handles_gracefully():
    """Return an empty list when no weights are provided."""

    assert scale_weights([]) == []


def test_layout_handles_graph_without_edges():
    """Compute layout without raising for graphs lacking edges."""

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(["a", "b"])
    graph = NetworkGraph(nx_graph)

    node_sizes, edge_widths, fonts_size = graph.layout()

    assert len(node_sizes) == graph.number_of_nodes
    assert edge_widths == []
    assert fonts_size


def test_network_components_title_respects_scale_factor(load_sample_df):
    """Ensure component plot titles honor the shared title scale factor."""

    df = load_sample_df("network.csv")
    style = StyleTemplate(font_size=11)

    fig = MatplotLibAPI.fplot_network_components(
        pd_df=df,
        source="city_a",
        target="city_b",
        weight="distance_km",
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
    deciles = np.percentile(np.array(weights), WEIGHT_PERCENTILES)

    expected = scale_weights(weights)
    assert scale_weights(weights, deciles=deciles) == expected


def test_compute_positions_is_reproducible_with_seed():
    """Return identical layouts when seeded and varied layouts when not."""

    nx_graph = nx.Graph()
    nx_graph.add_edges_from([(1, 2), (2, 3), (3, 1)])
    graph = NetworkGraph(nx_graph)

    seeded_positions = graph.compute_positions(seed=7)
    reseeded_positions = graph.compute_positions(seed=7)
    different_seed_positions = graph.compute_positions(seed=8)

    for node in seeded_positions:
        assert np.allclose(seeded_positions[node], reseeded_positions[node])

    assert any(
        not np.allclose(seeded_positions[node], different_seed_positions[node])
        for node in seeded_positions
    )
