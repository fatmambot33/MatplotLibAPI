"""Tests for bar and stacked bar visualizations."""

from matplotlib.figure import Figure

from MatplotLibAPI import fplot_bar


def test_fplot_bar(load_sample_df):
    """Render a bar chart from sample data."""

    df = load_sample_df("bar.csv")

    fig = fplot_bar(
        pd_df=df, category="product", value="revenue", group="region", stacked=True
    )

    assert isinstance(fig, Figure)
