"""Tests for area visualizations."""

from matplotlib.axes import Axes

import MatplotLibAPI  # noqa: F401
from matplotlib.figure import Figure

from MatplotLibAPI.area import aplot_area, fplot_area


def test_aplot_area(load_sample_df):
    """Render an area chart on existing axes from sample data."""
    df = load_sample_df("area.csv")

    ax = aplot_area(pd_df=df, x="quarter", y="subscriptions", label="segment")

    assert isinstance(ax, Axes)


def test_fplot_area(load_sample_df):
    """Render a stacked area chart from sample data."""

    df = load_sample_df("area.csv")

    fig = fplot_area(
        pd_df=df, x="quarter", y="subscriptions", label="segment", stacked=True
    )

    assert isinstance(fig, Figure)


def test_accessor_aplot_area(load_sample_df):
    """Render an area chart through the pandas accessor on existing axes."""
    df = load_sample_df("area.csv")

    ax = df.mpl.aplot_area(x="quarter", y="subscriptions", label="segment")

    assert isinstance(ax, Axes)


def test_accessor_fplot_area(load_sample_df):
    """Render an area chart through the pandas accessor on a new figure."""
    df = load_sample_df("area.csv")

    fig = df.mpl.fplot_area(x="quarter", y="subscriptions", label="segment")

    assert isinstance(fig, Figure)


def test_aplot_area_allows_overriding_grouped_alpha(load_sample_df):
    """Allow overriding grouped area alpha via forwarded keyword arguments."""
    df = load_sample_df("area.csv")

    ax = aplot_area(
        pd_df=df,
        x="quarter",
        y="subscriptions",
        label="segment",
        alpha=0.2,
    )

    alphas = [collection.get_alpha() for collection in ax.collections]
    assert any(alpha == 0.2 for alpha in alphas)


def test_aplot_area_allows_overriding_single_series_alpha(load_sample_df):
    """Allow overriding single-series fill alpha via forwarded keyword arguments."""
    df = load_sample_df("area.csv")

    ax = aplot_area(
        pd_df=df,
        x="quarter",
        y="subscriptions",
        alpha=0.2,
    )

    assert ax.collections[0].get_alpha() == 0.2
