"""Tests for network visualizations."""

from matplotlib.figure import Figure

import MatplotLibAPI


def test_fplot_network(load_sample_df):
    """Render a network figure from sample data."""

    df = load_sample_df("network.csv")

    fig = MatplotLibAPI.fplot_network(
        pd_df=df, source="city_a", target="city_b", weight="distance_km"
    )

    assert isinstance(fig, Figure)
