import pandas as pd
import pytest
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from MatplotLibAPI.Pivot import plot_pivoted_bars, plot_pivoted_lines, PIVOTBARS_STYLE_TEMPLATE, PIVOTLINES_STYLE_TEMPLATE

@pytest.fixture
def sample_data():
    """Fixture for creating a sample DataFrame."""
    data = {
        'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02']),
        'category': ['A', 'B', 'A', 'B'],
        'value': [10, 20, 15, 25],
        'browser': ['Chrome', 'Firefox', 'Chrome', 'Firefox']
    }
    return pd.DataFrame(data)

def test_plot_pivoted_bars(sample_data):
    """Test plot_pivoted_bars."""
    fig, ax = plt.subplots()
    result_ax = plot_pivoted_bars(sample_data, x='date', y='value', label='category', ax=ax)
    assert isinstance(result_ax, Axes)
    assert result_ax.get_title() == ""
    assert result_ax.get_xlabel() == "Date"
    assert result_ax.get_ylabel() == "Value"
    plt.close(fig)

def test_plot_pivoted_bars_with_title(sample_data):
    """Test plot_pivoted_bars with a title."""
    fig, ax = plt.subplots()
    result_ax = plot_pivoted_bars(sample_data, x='date', y='value', label='category', ax=ax, title="My Title")
    assert result_ax.get_title() == "My Title"
    plt.close(fig)

def test_plot_pivoted_lines(sample_data):
    """Test plot_pivoted_lines."""
    fig, ax = plt.subplots()
    result_ax = plot_pivoted_lines(sample_data, x='date', y='value', label='category', ax=ax)
    assert isinstance(result_ax, Axes)
    assert result_ax.get_title() == ""
    assert result_ax.get_xlabel() == "Date"
    assert result_ax.get_ylabel() == "Value"
    plt.close(fig)

def test_plot_pivoted_lines_with_title(sample_data):
    """Test plot_pivoted_lines with a title."""
    fig, ax = plt.subplots()
    result_ax = plot_pivoted_lines(sample_data, x='date', y='value', label='category', ax=ax, title="My Title")
    assert result_ax.get_title() == "My Title"
    plt.close(fig)
