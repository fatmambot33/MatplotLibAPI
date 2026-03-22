"""
Example module for network graph sample data generation and plotting.

This module provides functions to generate and plot sample network data for testing.
"""

import sys
from pathlib import Path
from matplotlib.figure import Figure
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from MatplotLibAPI.bubble import Bubble


def generate_sample_data() -> pd.DataFrame:
    """Generate and save sample data for a bubble chart."""
    data = {
        "country": ["USA", "China", "India", "Brazil", "Nigeria"],
        "population": [331, 1441, 1393, 213, 206],  # in millions
        "gdp_per_capita": [63593, 10500, 2191, 7741, 2229],
        "age": [62, 45, 35, 42, 30],
    }
    df = pd.DataFrame(data)
    return df


def plot_sample_data() -> Figure:
    """Load a sample DataFrame for testing."""

    pd_df = generate_sample_data()
    o_plot = Bubble(
        pd_df=pd_df, label="country", x="population", y="gdp_per_capita", z="age"
    )
    plot_fig = o_plot.fplot(title="Bubble Graph")

    return plot_fig


if __name__ == "__main__":
    plot_fig = plot_sample_data()
    plot_fig.show()
