"""Tests for word cloud visualizations."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from MatplotLibAPI.Wordcloud import (
    aplot_wordcloud,
    create_circular_mask,
    fplot_wordcloud,
)


def test_fplot_wordcloud(load_sample_df):
    """Render a word cloud figure from sample data."""

    df = load_sample_df("wordcloud.csv")

    fig = fplot_wordcloud(
        pd_df=df, text_column="country", weight_column="population", random_state=42
    )

    assert isinstance(fig, Figure)
    fig.canvas.draw()
    ax = fig.axes[0]
    expected_size = int(
        max(ax.get_window_extent().width, ax.get_window_extent().height)
    )
    image = ax.images[0].get_array()
    assert image is not None
    assert image.shape[0] == image.shape[1] >= expected_size


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
    assert image is not None
    assert tuple(image.shape[:2]) == mask.shape


def test_aplot_wordcloud(load_sample_df):
    """Render a word cloud onto an existing axes object."""

    df = load_sample_df("wordcloud.csv")
    fig, ax = plt.subplots()

    result_ax = aplot_wordcloud(
        ax=ax,
        pd_df=df,
        text_column="country",
        weight_column="population",
        random_state=42,
    )

    assert result_ax is not None
    assert ax is result_ax
    fig.canvas.draw()
    expected_size = int(
        max(ax.get_window_extent().width, ax.get_window_extent().height)
    )
    image = result_ax.images[0].get_array()
    assert image is not None
    assert image.shape[0] == image.shape[1] >= expected_size


def test_create_circular_mask():
    """Verify circular mask generation."""

    mask = create_circular_mask(size=100, radius=40)
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert np.sum(mask == 0) > 0
    assert np.sum(mask == 255) > 0
