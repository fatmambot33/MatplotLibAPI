"""Tests for heatmap and correlation matrix visualizations."""

from matplotlib.figure import Figure

from MatplotLibAPI.heatmap import aplot_heatmap, fplot_heatmap, fplot_correlation_matrix


def test_fplot_heatmap(load_sample_df):
    """Render a heatmap from sample data."""

    df = load_sample_df("heatmap.csv")

    fig = fplot_heatmap(pd_df=df, x="month", y="channel", value="engagements")

    assert isinstance(fig, Figure)


def test_fplot_correlation_matrix(load_sample_df):
    """Render a correlation matrix from sample data."""

    df = load_sample_df("heatmap.csv")

    fig = fplot_correlation_matrix(
        pd_df=df, x="month", y="channel", value="engagements"
    )

    assert isinstance(fig, Figure)


def test_aplot_heatmap_allows_overriding_colormap(load_sample_df):
    """Allow overriding heatmap colormap with forwarded kwargs."""

    df = load_sample_df("heatmap.csv")

    ax = aplot_heatmap(
        pd_df=df,
        x="month",
        y="channel",
        value="engagements",
        cmap="magma",
    )

    assert ax.collections
