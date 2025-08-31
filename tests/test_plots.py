"""Smoke tests for plotting functions to ensure they run without errors."""

import pandas as pd
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from MatplotLibAPI.Composite import plot_composite_bubble
from MatplotLibAPI.Network import fplot_network
from MatplotLibAPI.Pivot import plot_pivoted_bars
from MatplotLibAPI.Table import fplot_table
from MatplotLibAPI.Timeserie import fplot_timeserie
from MatplotLibAPI.Treemap import fplot_treemap


def test_plot_pivoted_bars():
    """Test plot_pivoted_bars returns an Axes object."""
    data = {
        "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
        "category": ["A", "B", "A", "B"],
        "value": [10, 20, 15, 25],
    }
    df = pd.DataFrame(data)

    ax = plot_pivoted_bars(data=df, label="category", x="date", y="value")

    assert isinstance(ax, Axes)
    plt.close()


def test_fplot_table():
    """Test fplot_table returns a Figure object."""
    data = {"col1": [1, 2, 3], "col2": ["A", "B", "C"]}
    df = pd.DataFrame(data)

    fig = fplot_table(pd_df=df, cols=["col1", "col2"])

    assert isinstance(fig, Figure)
    plt.close()


def test_fplot_network():
    """Test fplot_network returns a Figure object."""
    data = {
        "source": ["A", "B", "C", "D"],
        "target": ["B", "C", "D", "A"],
        "weight": [1, 1, 1, 1],
    }
    df = pd.DataFrame(data)

    fig = fplot_network(pd_df=df)

    assert isinstance(fig, Figure)
    plt.close()


def test_fplot_timeserie():
    """Test fplot_timeserie returns a Figure object."""
    data = {
        "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        "group": ["A", "A", "B"],
        "value": [1, 2, 3],
    }
    df = pd.DataFrame(data)

    fig = fplot_timeserie(pd_df=df, label="group", x="date", y="value")

    assert isinstance(fig, Figure)
    plt.close()


def test_fplot_treemap():
    """Test fplot_treemap returns a Plotly Figure object."""
    data = {"path": ["A", "B", "C"], "values": [10, 20, 30]}
    df = pd.DataFrame(data)

    fig = fplot_treemap(pd_df=df, path="path", values="values")

    assert isinstance(fig, go.Figure)
    plt.close()


def test_plot_composite_bubble():
    """Test plot_composite_bubble returns a Figure object."""
    data = {
        "country": ["A", "B", "C", "D"],
        "gdp_per_capita": [45000, 42000, 52000, 48000],
        "life_expectancy": [81, 78, 83, 82],
        "population": [10, 20, 5, 30],
    }
    df = pd.DataFrame(data)

    fig = plot_composite_bubble(
        pd_df=df,
        label="country",
        x="gdp_per_capita",
        y="life_expectancy",
        z="population",
    )

    assert isinstance(fig, Figure)
    plt.close()
