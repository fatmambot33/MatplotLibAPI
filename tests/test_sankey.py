"""Tests for Sankey visualizations."""

import plotly.graph_objects as go

from MatplotLibAPI import fplot_sankey


def test_fplot_sankey(load_sample_df):
    """Render a Sankey diagram from sample data."""

    df = load_sample_df("sankey.csv")

    fig = fplot_sankey(pd_df=df, source="source", target="target", value="value")

    assert isinstance(fig, go.Figure)
