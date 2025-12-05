"""Tests for sunburst visualizations."""

import plotly.graph_objects as go

from MatplotLibAPI.Sunburst import fplot_sunburst


def test_fplot_sunburst(load_sample_df):
    """Render a sunburst Plotly figure from sample data."""

    df = load_sample_df("sunburst.csv")

    fig = fplot_sunburst(df, labels="name", parents="parent", values="population")

    assert isinstance(fig, go.Figure)
