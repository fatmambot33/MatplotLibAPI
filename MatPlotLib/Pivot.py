from typing import List, Optional, Union

import pandas as pd

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, MonthLocator

from .Utils import (PIVOTBARS_STYLE_TEMPLATE, PIVOTLINES_STYLE_TEMPLATE,
                    DynamicFuncFormatter, StyleTemplate, generate_ticks)


def plot_pivotbar(data, metric, n_top, title):
    # Sort the data by metric column in descending order
    data_sorted = data.sort_values(by=metric, ascending=False)

    # Select the top rows
    top_rows = data_sorted.head(n_top)

    # Plotting the top 50 data points with tag labels
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the 'Used' data points (where Used == 1) in green
    used_data = top_rows[top_rows['used'] == 1]
    ax.bar(used_data.tag, used_data[metric],
           color='green', label='Used', alpha=0.7)

    # Plot the 'Not Used' data points (where Used == 0) in red
    not_used_data = top_rows[top_rows['used'] == 0]
    ax.bar(not_used_data.tag, not_used_data[metric],
           color='red', label='Not Used', alpha=0.7)

    # Set labels and title
    ax.set_ylabel('UVs')
    ax.set_title(f'{title}\nTop {n_top} tags')
    ax.legend()

    ax.tick_params(axis='x', rotation=90)
    return fig


def plot_lines(ax: Axes,
               data: pd.DataFrame,
               x_col: str,
               y_col: Union[str, List[str]],
               style: Optional[StyleTemplate] = None,
               fig_title: Optional[str] = None,
               n_top: int = 4,
               z_col: str = "browser") -> Axes:
    """
    This function plots time series lines for the top n elements in the specified dimension.

    Parameters:
    ax (matplotlib.axes._axes.Axes): The ax to plot on.
    data (pd.DataFrame): The data to plot.
    metrics (Union[str, List[str]]): The column name(s) in data to plot.
    date_col (str): The column name containing the date information.
    ... (other parameters): Various parameters to customize the plot.
    date_format (str): The format of the date to display on the x-axis.
    date_locator (matplotlib.dates.Locator): Locator object to determine the date ticks on the x-axis.

    Returns:
    ax (matplotlib.axes._axes.Axes): The ax with the plot.
    """

    # Validate inputs
    if x_col not in data.columns:
        raise ValueError(f"'{x_col}' column not found in the data")
    if not isinstance(y_col, list) and not isinstance(y_col, str):
        raise TypeError("'metrics' should be a string or a list of strings")
    if isinstance(y_col, list) and not len(y_col) >= 2:
        raise ValueError(
            f"metrics should be 2 of lengths column not found in the data")
    ax.clear()
    if fig_title is not None:
        ax.set_title(fig_title)
    if style is None:
        style = PIVOTLINES_STYLE_TEMPLATE
    ax.figure.set_facecolor(style.fig_background_color)
    ax.figure.set_edgecolor(style.fig_border)

    display_metric = y_col[0]
    sort_metric = y_col[1]
    # Get the top n elements in the specified z
    top_elements = data.groupby(
        z_col)[sort_metric].sum().nlargest(n_top).index.tolist()
    top_elements_df = data[data[z_col].isin(top_elements)]
    y_min = 0
    # Plot the time series lines for each of the top elements
    for element in top_elements:
        subset = top_elements_df[top_elements_df[z_col] == element]
        # Define the line style based on the element name
        if element == "Chrome":
            line_style = '-'
            color = 'green'
        elif element == "Android Webview":
            line_style = '--'
            color = 'green'
        elif element == "Safari":
            line_style = '-'
            color = 'red'
        elif element == "Safari (in-app)":
            line_style = '--'
            color = 'red'
        else:
            line_style = '-'
            color = 'black'
        y_min = min(y_min, subset[display_metric].min())

        ax.plot(subset[x_col], subset[display_metric], label=element)

    # Set x-axis date format and locator
        if style.x_formatter is not None:
            x_min = data[x_col].min()
            x_max = data[x_col].max()

            if style.x_formatter == "year_month_formatter":
                ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
            else:
                ax.xaxis.set_major_formatter(
                    DynamicFuncFormatter(style.x_formatter))
                ax.set_xticks(generate_ticks(
                    x_min, x_max, num_ticks=style.x_ticks))

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Set title and labels
    ax.set_xlabel(x_col)
    y_max = data[display_metric].dropna().quantile(0.95)

    ax.set_ylim(y_min, y_max)
    ax.set_ylabel(display_metric)
    if style.y_formatter is not None:
        ax.yaxis.set_major_formatter(
            DynamicFuncFormatter(style.y_formatter))
        ax.set_yticks(generate_ticks(
            y_min, y_max, num_ticks=style.y_ticks))
    else:
        ylabels = ['{:,.0f}%'.format(y) for y in ax.get_yticks()*100]
        ax.yaxis.set_yticklabels(ylabels)

    # Add legend and grid
    ax.legend()
    ax.grid(True)

    return ax


def plot_bars(ax: Axes,
              data: pd.DataFrame,
              x_col: str,
              y_col: Union[str, List[str]],
              style: Optional[StyleTemplate] = None,
              fig_title: Optional[str] = None,
              z_col: str = "browser",
              n=5,
              agg_func: str = 'sum') -> Axes:

    # Validate inputs

    if not isinstance(y_col, list):
        y_col = [y_col, y_col]  # Ensure y_col is a list

    # Clear axis
    ax.clear()
    if style is None:
        style = PIVOTBARS_STYLE_TEMPLATE

    # Copying the da
    df_plot = data.copy()

    if df_plot.index.name != x_col:
        df_plot.set_index(x_col, inplace=True)
        print(df_plot.head())

    # Set x-axis date format and locator
    if style.x_formatter is not None:
        if style.x_formatter == "year_month_formatter":
            # Ensure the x_col is in datetime format
            if not pd.api.types.is_datetime64_any_dtype(df_plot.index):
                df_plot[x_col] = pd.to_datetime(df_plot[x_col])
            df_plot.index = df_plot.index.to_pydatetime()
            # Plot the data first
            df_plot.plot(kind='bar', stacked=True, ax=ax)

            ax.xaxis.set_major_locator(MonthLocator())
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

            # Convert the Pandas datetime64 objects to strings in 'Year-Month' format
            formatted_dates = df_plot.index.strftime('%Y-%m')
            # Find the index positions where the day is the first of the month
            first_of_month_positions = [
                i for i, date in enumerate(df_plot.index) if date.day == 1]

            # Set x-ticks at the first of the month positions
            ax.set_xticks(first_of_month_positions)
            ax.set_xticklabels([formatted_dates[i]
                               for i in first_of_month_positions], rotation=45)

            # Remove the blank space at the beginning
            ax.set_xlim(left=0, right=len(df_plot.index) - 1)

        else:
            x_min = df_plot[x_col].min()
            x_max = df_plot[x_col].max()
            df_plot.plot(kind='bar', stacked=True, ax=ax)
            ax.xaxis.set_major_formatter(
                DynamicFuncFormatter(style.x_formatter))
            ax.set_xticks(generate_ticks(
                x_min, x_max, num_ticks=style.x_ticks))
    else:
        df_plot.plot(kind='bar', stacked=True, ax=ax)

    # Apply custom y_formatter if provided
    if style and style.y_formatter is not None:
        ax.yaxis.set_major_formatter(DynamicFuncFormatter(style.y_formatter))

    # Set title and labels
    ax.set_title(fig_title if fig_title else "")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col[0])

    return ax
