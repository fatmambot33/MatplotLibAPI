"""Tests for waffle visualizations."""

from matplotlib.figure import Figure

from MatplotLibAPI import fplot_waffle


def test_fplot_waffle(load_sample_df):
    """Render a waffle chart from sample data."""

    df = load_sample_df("waffle.csv")

    fig = fplot_waffle(pd_df=df, category="device", value="sessions", rows=10)

    assert isinstance(fig, Figure)
