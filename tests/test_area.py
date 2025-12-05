"""Tests for area visualizations."""

from matplotlib.figure import Figure

from MatplotLibAPI import fplot_area


def test_fplot_area(load_sample_df):
    """Render a stacked area chart from sample data."""

    df = load_sample_df("area.csv")

    fig = fplot_area(
        pd_df=df, x="quarter", y="subscriptions", label="segment", stacked=True
    )

    assert isinstance(fig, Figure)
