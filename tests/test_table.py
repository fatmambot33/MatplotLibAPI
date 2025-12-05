"""Tests for table visualizations."""

from matplotlib.figure import Figure

from MatplotLibAPI.Table import fplot_table


def test_fplot_table(load_sample_df):
    """Render a table figure from sample data."""

    df = load_sample_df("table.csv")

    fig = fplot_table(pd_df=df, cols=df.columns.tolist())

    assert isinstance(fig, Figure)
