"""Tests for histogram and KDE visualizations."""

from matplotlib.figure import Figure
from matplotlib.colors import to_hex

from MatplotLibAPI.histogram import aplot_histogram, fplot_histogram


def test_fplot_histogram_kde(load_sample_df):
    """Render a histogram with KDE from sample data."""

    df = load_sample_df("histogram.csv")

    fig = fplot_histogram(pd_df=df, column="waiting_time_minutes", bins=10, kde=True)

    assert isinstance(fig, Figure)


def test_aplot_histogram_allows_overriding_color(load_sample_df):
    """Allow overriding histogram color with forwarded kwargs."""

    df = load_sample_df("histogram.csv")

    ax = aplot_histogram(
        pd_df=df,
        column="waiting_time_minutes",
        color="red",
    )

    assert to_hex(ax.patches[0].get_facecolor()) == "#ff0000"
