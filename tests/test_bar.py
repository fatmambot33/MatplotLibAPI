"""Tests for bar and stacked bar visualizations."""

from matplotlib.figure import Figure

from MatplotLibAPI.bar import aplot_bar, fplot_bar


def test_fplot_bar(load_sample_df):
    """Render a bar chart from sample data."""

    df = load_sample_df("bar.csv")

    fig = fplot_bar(
        pd_df=df, category="product", value="revenue", group="region", stacked=True
    )

    assert isinstance(fig, Figure)


def test_aplot_bar_allows_overriding_alpha(load_sample_df):
    """Allow overriding default bar alpha with forwarded kwargs."""

    df = load_sample_df("bar.csv")

    ax = aplot_bar(
        pd_df=df,
        category="product",
        value="revenue",
        group="region",
        stacked=True,
        alpha=0.2,
    )

    assert any(patch.get_alpha() == 0.2 for patch in ax.patches)
