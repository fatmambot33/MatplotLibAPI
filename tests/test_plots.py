"""Tests for plotting functions."""

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import MatplotLibAPI

from MatplotLibAPI.Pivot import plot_pivoted_bars
from MatplotLibAPI.Table import fplot_table
from MatplotLibAPI.Timeserie import fplot_timeserie
from MatplotLibAPI.Treemap import fplot_treemap
from MatplotLibAPI.Sunburst import fplot_sunburst
from MatplotLibAPI.Wordcloud import fplot_wordcloud
from MatplotLibAPI import (
    fplot_bar,
    fplot_histogram_kde,
    fplot_correlation_matrix,
    fplot_pie_donut,
)
import plotly.graph_objects as go


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

    fig = MatplotLibAPI.fplot_network(pd_df=df)

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

    fig = MatplotLibAPI.plot_composite_bubble(
        pd_df=df,
        label="country",
        x="gdp_per_capita",
        y="life_expectancy",
        z="population",
    )

    assert isinstance(fig, Figure)
    plt.close()


def test_fplot_wordcloud():
    """Test fplot_wordcloud returns a Figure object."""
    data = {"word": ["alpha", "beta", "gamma", "alpha"], "weight": [2, 1, 3, 1]}
    df = pd.DataFrame(data)

    fig = fplot_wordcloud(
        pd_df=df, text_column="word", weight_column="weight", random_state=42
    )

    assert isinstance(fig, Figure)
    plt.close()


def test_dataframe_accessor_fplot_composite_treemap():
    """Test fplot_composite_treemap via the pandas accessor."""
    data = {"group": ["A", "B", "C"], "value": [10, 20, 30]}
    df = pd.DataFrame(data)

    fig = df.mpl.fplot_composite_treemap(paths=["group"], values="value")

    assert isinstance(fig, go.Figure)


def test_fplot_sunburst():
    """Test fplot_sunburst returns a Plotly Figure object."""
    data = {
        "labels": [
            "Eve",
            "Cain",
            "Seth",
            "Enos",
            "Noam",
            "Abel",
            "Awan",
            "Enoch",
            "Azura",
        ],
        "parents": ["", "Eve", "Eve", "Seth", "Seth", "Eve", "Eve", "Awan", "Eve"],
        "values": [10, 14, 12, 10, 2, 6, 6, 4, 4],
    }
    df = pd.DataFrame(data)

    fig = fplot_sunburst(df, labels="labels", parents="parents", values="values")

    assert isinstance(fig, go.Figure)


def test_fplot_bar():
    """Test fplot_bar returns a Figure object."""
    data = {"category": ["A", "B", "A"], "group": ["X", "Y", "X"], "value": [5, 7, 3]}
    df = pd.DataFrame(data)

    fig = fplot_bar(
        pd_df=df, category="category", value="value", group="group", stacked=True
    )

    assert isinstance(fig, Figure)
    plt.close()


def test_fplot_histogram_kde():
    """Test fplot_histogram_kde returns a Figure object."""
    df = pd.DataFrame({"value": [1, 2, 2, 3, 4, 5]})

    fig = fplot_histogram_kde(pd_df=df, column="value", bins=5, kde=True)

    assert isinstance(fig, Figure)
    plt.close()


def test_fplot_correlation_matrix():
    """Test fplot_correlation_matrix returns a Figure object."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1], "c": [1, 1, 1]})

    fig = fplot_correlation_matrix(pd_df=df)

    assert isinstance(fig, Figure)
    plt.close()


def test_fplot_pie_donut():
    """Test fplot_pie_donut returns a Figure object."""
    df = pd.DataFrame({"category": ["A", "B"], "value": [30, 70]})

    fig = fplot_pie_donut(pd_df=df, category="category", value="value", donut=True)

    assert isinstance(fig, Figure)
    plt.close()
