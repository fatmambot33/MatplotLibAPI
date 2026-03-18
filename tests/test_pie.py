"""Tests for pie and donut visualizations."""

from matplotlib.figure import Figure

from MatplotLibAPI.pie import aplot_pie, fplot_pie


def test_fplot_pie_donut(load_sample_df):
    """Render a donut chart from sample data."""

    df = load_sample_df("pie.csv")

    fig = fplot_pie(pd_df=df, category="device", value="sessions", donut=True)

    assert isinstance(fig, Figure)


def test_aplot_pie_allows_overriding_autopct(load_sample_df):
    """Allow overriding pie autopct with forwarded kwargs."""

    df = load_sample_df("pie.csv")

    ax = aplot_pie(
        pd_df=df,
        category="device",
        value="sessions",
        autopct="%1.0f%%",
    )

    autotexts = [text.get_text() for text in ax.texts if "%" in text.get_text()]
    assert all("." not in text for text in autotexts)
