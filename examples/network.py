"""
Example module for network graph sample data generation and plotting.

This module provides functions to generate and plot sample network data for testing.
"""

import sys
from pathlib import Path
from matplotlib.figure import Figure
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from MatplotLibAPI.network import NetworkGraph


def generate_sample_network_data() -> pd.DataFrame:
    """Generate and save sample data for a network graph."""
    data = {
        "city_a": ["New York", "London", "Tokyo", "Sydney", "New York"],
        "city_b": ["London", "Tokyo", "Sydney", "New York", "Tokyo"],
        "distance_km": [5585, 9562, 7824, 16027, 10850],
    }
    df = pd.DataFrame(data)
    return df


def plot_sample_network_data() -> Figure:
    """Load a sample DataFrame for testing."""

    pd_df = generate_sample_network_data()
    graph = NetworkGraph.from_pandas_edgelist(
        pd_df, source="city_a", target="city_b", edge_weight_col="distance_km"
    )
    plot_fig = graph.fplot(title="Network Graph")

    return plot_fig


if __name__ == "__main__":
    plot_df = generate_sample_network_data()
    f = plot_df.mpl.fplot_network(
        source="city_a", target="city_b", edge_weight_col="distance_km"
    )
    plot_fig = plot_sample_network_data()
