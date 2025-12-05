"""Tests for pie and donut visualizations."""

from matplotlib.figure import Figure

from MatplotLibAPI import fplot_pie_donut


def test_fplot_pie_donut(load_sample_df):
    """Render a donut chart from sample data."""

    df = load_sample_df("pie.csv")

    fig = fplot_pie_donut(pd_df=df, category="device", value="sessions", donut=True)

    assert isinstance(fig, Figure)
