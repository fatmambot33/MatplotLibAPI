"""Tests for pivot visualizations."""

from matplotlib.axes import Axes

from MatplotLibAPI.Pivot import plot_pivoted_bars


def test_plot_pivoted_bars(load_sample_df):
    """Plot pivoted bars using generated sample data."""

    df = load_sample_df("pivot.csv", parse_dates=["year"])

    ax = plot_pivoted_bars(data=df, label="city", x="year", y="population_increase")

    assert isinstance(ax, Axes)
