"""Tests for network visualizations."""

from matplotlib.figure import Figure

import MatplotLibAPI
from MatplotLibAPI.StyleTemplate import StyleTemplate, TITLE_SCALE_FACTOR


def test_fplot_network(load_sample_df):
    """Render a network figure from sample data."""

    df = load_sample_df("network.csv")

    fig = MatplotLibAPI.fplot_network(
        pd_df=df, source="city_a", target="city_b", weight="distance_km"
    )

    assert isinstance(fig, Figure)


def test_accessor_fplot_network_components(load_sample_df):
    """Render network component figures via the pandas accessor."""

    df = load_sample_df("network.csv")

    fig = df.mpl.fplot_network_components(
        source="city_a", target="city_b", weight="distance_km"
    )

    assert isinstance(fig, Figure)


def test_network_components_title_respects_scale_factor(load_sample_df):
    """Ensure component plot titles honor the shared title scale factor."""

    df = load_sample_df("network.csv")
    style = StyleTemplate(font_size=11)

    fig = MatplotLibAPI.fplot_network_components(
        pd_df=df,
        source="city_a",
        target="city_b",
        weight="distance_km",
        title="Component Title",
        style=style,
    )

    suptitle = getattr(fig, "_suptitle", None)
    expected_size = style.font_size * TITLE_SCALE_FACTOR * 1.25

    assert suptitle is not None
    assert suptitle.get_fontsize() == expected_size
