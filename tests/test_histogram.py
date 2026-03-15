"""Tests for histogram and KDE visualizations."""

from matplotlib.figure import Figure

from MatplotLibAPI.histogram import fplot_histogram


def test_fplot_histogram_kde(load_sample_df):
    """Render a histogram with KDE from sample data."""

    df = load_sample_df("histogram.csv")

    fig = fplot_histogram(pd_df=df, column="waiting_time_minutes", bins=10, kde=True)

    assert isinstance(fig, Figure)
