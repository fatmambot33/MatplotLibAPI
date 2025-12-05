"""Tests for word cloud visualizations."""

from matplotlib.figure import Figure

from MatplotLibAPI.Wordcloud import create_circular_mask, fplot_wordcloud


def test_fplot_wordcloud(load_sample_df):
    """Render a word cloud figure from sample data."""

    df = load_sample_df("wordcloud.csv")

    fig = fplot_wordcloud(
        pd_df=df, text_column="country", weight_column="population", random_state=42
    )

    assert isinstance(fig, Figure)
    default_mask = create_circular_mask()
    image = fig.axes[0].images[0].get_array()
    assert tuple(image.shape[:2]) == default_mask.shape


def test_fplot_wordcloud_with_mask(load_sample_df):
    """Render a word cloud using a circular mask to constrain placement."""

    df = load_sample_df("wordcloud.csv")
    mask = create_circular_mask(size=200)

    fig = fplot_wordcloud(
        pd_df=df,
        text_column="country",
        weight_column="population",
        random_state=0,
        mask=mask,
    )

    image = fig.axes[0].images[0].get_array()
    assert tuple(image.shape[:2]) == mask.shape
