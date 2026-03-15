"""
Example module for network graph sample data generation and plotting.

This module provides functions to generate and plot sample network data for testing.
"""

import sys
import os
from pathlib import Path
from matplotlib.figure import Figure
import pandas as pd
from sample_data import main as generate_sample_data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def generate_sample_network_data():
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
    from MatplotLibAPI.network import NetworkGraph

    pd_df = generate_sample_network_data()
    graph = NetworkGraph.from_pandas_edgelist(
        pd_df, source="city_a", target="city_b", edge_weight_col="distance_km"
    )
    plot_fig = graph.fplot(title="Network Graph")

    return plot_fig


if __name__ == "__main__":
    plot_fig = plot_sample_network_data()
    plot_fig.savefig("examples/network.png")
