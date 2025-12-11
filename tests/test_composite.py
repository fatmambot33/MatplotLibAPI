"""Tests for composite plotting helpers."""

from matplotlib.collections import PathCollection
import pandas as pd

from MatplotLibAPI.Composite import plot_wordcloud_network


def test_plot_wordcloud_network_renders_wordcloud_and_network() -> None:
    """Ensure the composite plot shows one word cloud and one network graph."""

    nodes_df = pd.DataFrame({"node": ["alpha", "beta", "gamma"], "weight": [5, 3, 1]})
    edges_df = pd.DataFrame(
        {
            "source": ["alpha", "alpha", "beta"],
            "target": ["beta", "gamma", "gamma"],
            "weight": [2, 1, 3],
        }
    )

    fig = plot_wordcloud_network(nodes_df, edges_df)

    assert len(fig.axes) == 2
    wordcloud_ax, network_ax = fig.axes

    assert wordcloud_ax.images, "Word cloud subplot should render an image."
    assert any(
        isinstance(collection, PathCollection) for collection in network_ax.collections
    ), "Network subplot should render nodes as a PathCollection."


def test_plot_wordcloud_network_handles_zero_weights() -> None:
    """Ensure the word cloud still renders when node weights are zero or missing."""

    nodes_df = pd.DataFrame({"node": ["alpha", "beta", "gamma"], "weight": [0, 0, 0]})
    edges_df = pd.DataFrame(
        {
            "source": ["alpha", "alpha", "beta"],
            "target": ["beta", "gamma", "gamma"],
            "weight": [2, 1, 3],
        }
    )

    fig = plot_wordcloud_network(nodes_df, edges_df)

    wordcloud_ax, network_ax = fig.axes

    assert wordcloud_ax.images, "Word cloud should render even when weights are zero."
    assert not any(
        isinstance(collection, PathCollection)
        for collection in wordcloud_ax.collections
    ), "Word cloud subplot should not contain network node collections."
    assert any(
        isinstance(collection, PathCollection) for collection in network_ax.collections
    ), "Network subplot should still render nodes as a PathCollection."
