"""Tests for treemap visualizations."""

import plotly.graph_objects as go

from MatplotLibAPI.Treemap import fplot_treemap


def test_fplot_treemap(load_sample_df):
    """Render a treemap Plotly figure from sample data."""

    df = load_sample_df("treemap.csv")

    fig = fplot_treemap(pd_df=df, path="path", values="population")

    assert isinstance(fig, go.Figure)


def test_accessor_fplot_composite_treemap(load_sample_df):
    """Render a composite treemap via the pandas accessor."""

    df = load_sample_df("treemap.csv")
    df_levels = df.loc[df["parent"] != ""].copy()
    df_levels["continent"] = df_levels["parent"]
    df_levels["country"] = df_levels["location"]

    fig = df_levels.mpl.fplot_composite_treemap(
        paths=["continent", "country"], values="population"
    )

    assert isinstance(fig, go.Figure)
