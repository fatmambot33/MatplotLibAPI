"""Tests for pie and donut visualizations."""

from matplotlib.figure import Figure

from MatplotLibAPI.pie import fplot_pie


def test_fplot_pie_donut(load_sample_df):
    """Render a donut chart from sample data."""

    df = load_sample_df("pie.csv")

    fig = fplot_pie(pd_df=df, category="device", value="sessions", donut=True)

    assert isinstance(fig, Figure)
