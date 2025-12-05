"""Tests for time series visualizations."""

from matplotlib.figure import Figure

from MatplotLibAPI.Timeserie import fplot_timeserie


def test_fplot_timeserie(load_sample_df):
    """Render a time series figure from sample data."""

    df = load_sample_df("timeserie.csv", parse_dates=["year"])

    fig = fplot_timeserie(pd_df=df, label="city", x="year", y="population")

    assert isinstance(fig, Figure)
