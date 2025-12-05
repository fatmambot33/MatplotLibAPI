"""Tests for word cloud visualizations."""

from matplotlib.figure import Figure

from MatplotLibAPI.Wordcloud import fplot_wordcloud


def test_fplot_wordcloud(load_sample_df):
    """Render a word cloud figure from sample data."""

    df = load_sample_df("wordcloud.csv")

    fig = fplot_wordcloud(
        pd_df=df, text_column="country", weight_column="population", random_state=42
    )

    assert isinstance(fig, Figure)
