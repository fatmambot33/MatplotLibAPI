"""Tests for box and violin visualizations."""

from matplotlib.figure import Figure

from MatplotLibAPI import fplot_box_violin


def test_fplot_box_violin(load_sample_df):
    """Render box and violin plots from sample data."""

    df = load_sample_df("box_violin.csv")

    fig = fplot_box_violin(
        pd_df=df, column="satisfaction_score", by="department", violin=True
    )

    assert isinstance(fig, Figure)
