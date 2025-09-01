"""Smoke tests for the MatplotLibAPI package."""

import MatplotLibAPI

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd


def test_import():
    """Test that the package can be imported."""
    assert MatplotLibAPI is not None


def test_aplot_bubble_returns_axes():
    """Test that aplot_bubble returns a Matplotlib Axes object."""
    # Create a sample DataFrame
    data = {
        "country": ["A", "B", "C", "D"],
        "gdp_per_capita": [45000, 42000, 52000, 48000],
        "life_expectancy": [81, 78, 83, 82],
        "population": [10, 20, 5, 30],
    }
    df = pd.DataFrame(data)

    # Call the plotting function
    ax = MatplotLibAPI.aplot_bubble(
        pd_df=df,
        label="country",
        x="gdp_per_capita",
        y="life_expectancy",
        z="population",
        title="GDP vs. Life Expectancy",
    )

    # Check if the return is an Axes object
    assert isinstance(ax, Axes)
    # Check if the title is set correctly
    assert ax.get_title() == "GDP vs. Life Expectancy"
    plt.close()


def test_dataframe_accessor_aplot_bubble():
    """Test that the pandas accessor for aplot_bubble works."""
    # Create a sample DataFrame
    data = {
        "country": ["A", "B", "C", "D"],
        "gdp_per_capita": [45000, 42000, 52000, 48000],
        "life_expectancy": [81, 78, 83, 82],
        "population": [10, 20, 5, 30],
    }
    df = pd.DataFrame(data)

    # Call the plotting function via the accessor
    ax = df.mpl.aplot_bubble(
        label="country",
        x="gdp_per_capita",
        y="life_expectancy",
        z="population",
        title="GDP vs. Life Expectancy",
    )

    # Check if the return is an Axes object
    assert isinstance(ax, Axes)
    # Check if the title is set correctly
    assert ax.get_title() == "GDP vs. Life Expectancy"
    plt.close()
