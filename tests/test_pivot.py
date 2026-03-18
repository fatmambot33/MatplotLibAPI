"""Tests for pivot visualizations."""

from matplotlib.axes import Axes

from MatplotLibAPI.pivot import aplot_pivoted_bars


def test_plot_pivoted_bars(load_sample_df):
    """Plot pivoted bars using generated sample data."""

    df = load_sample_df("pivot.csv", parse_dates=["year"])

    ax = aplot_pivoted_bars(data=df, label="city", x="year", y="population_increase")

    assert isinstance(ax, Axes)


def test_plot_pivoted_bars_allows_overriding_alpha(load_sample_df):
    """Allow overriding pivoted bar alpha with forwarded kwargs."""

    df = load_sample_df("pivot.csv", parse_dates=["year"])

    ax = aplot_pivoted_bars(
        data=df,
        label="city",
        x="year",
        y="population_increase",
        alpha=0.2,
    )

    assert any(patch.get_alpha() == 0.2 for patch in ax.patches)
