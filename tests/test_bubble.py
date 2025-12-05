"""Tests for bubble visualizations."""

from matplotlib.figure import Figure

import MatplotLibAPI


def test_plot_composite_bubble(load_sample_df):
    """Render a composite bubble plot from sample data."""

    df = load_sample_df("bubble.csv")
    df["life_expectancy"] = [79, 77, 70, 75, 61]

    fig = MatplotLibAPI.plot_composite_bubble(
        pd_df=df,
        label="country",
        x="gdp_per_capita",
        y="life_expectancy",
        z="population",
    )

    assert isinstance(fig, Figure)
