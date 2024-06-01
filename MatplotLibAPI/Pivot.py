# Hint for Visual Code Python Interactive window
# %%
from typing import List, Optional, Union

import pandas as pd

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, MonthLocator


from . import DynamicFuncFormatter, StyleTemplate, generate_ticks, string_formatter,  percent_formatter, format_func
from .. import validate_dataframe

PIVOTBARS_STYLE_TEMPLATE = StyleTemplate(
    background_color='black',
    fig_border='darkgrey',
    font_color='white',
    palette='magma',
    format_funcs={"y": percent_formatter,
                  "label": string_formatter}
)
PIVOTLINES_STYLE_TEMPLATE = StyleTemplate(
    background_color='white',
    fig_border='lightgrey',
    palette='viridis',
    format_funcs={"y": percent_formatter, "label": string_formatter}
)


def plot_pivotbar(pd_df: pd.DataFrame,
                  label: str,
                  x: str,
                  y: str,
                  agg: str = "sum",
                  style: StyleTemplate = PIVOTBARS_STYLE_TEMPLATE,
                  title: Optional[str] = None,
                  sort_by: Optional[str] = None,
                  ascending: bool = False,
                  ax: Optional[Axes] = None):

    validate_dataframe(pd_df, cols=[label, x, y], sort_by=sort_by)
    style.format_funcs = format_func(style.format_funcs, label=label, x=x, y=y)
    pivot_df = pd.pivot_table(pd_df, values=y, index=[
                              x], columns=[label], aggfunc=agg)
    # Reset index to make x a column again
    pivot_df = pivot_df.reset_index()

    if not ax:
        ax = plt.gca()

    # Plot each label's data
    for column in pivot_df.columns[1:]:
        _label = column
        if style.format_funcs.get(column):
            _label = style.format_funcs[column](column)
        ax.bar(x=pivot_df[x],
               height=pivot_df[column],
               label=_label, alpha=0.7)

    # Set labels and title
    ax.set_ylabel(string_formatter(y))
    ax.set_xlabel(string_formatter(x))
    if title:
        ax.set_title(f'{title}')
    ax.legend(fontsize=style.font_size-2,
              title_fontsize=style.font_size+2,
              labelcolor='linecolor',
              facecolor=style.background_color)

    ax.tick_params(axis='x', rotation=90)
    return ax


def plot_lines(
    data: pd.DataFrame,
    label: str,
    x: str,
    y: Union[str, List[str]],
    title: Optional[str] = None,
    style: Optional[StyleTemplate] = PIVOTBARS_STYLE_TEMPLATE,
    max_values: int = 4,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    ax: Optional[Axes] = None
) -> Axes:

    if title is not None:
        ax.set_title(title)
    ax.figure.set_facecolor(style.background_color)
    ax.figure.set_edgecolor(style.fig_border)
    # Get the top n elements in the specified z
    top_elements = data.groupby(
        label)[y].sum().nlargest(max_values).index.tolist()
    top_elements_df = data[data[label].isin(top_elements)]
    y_min = 0
    # Plot the time series lines for each of the top elements
    for element in top_elements:
        subset = top_elements_df[top_elements_df[label] == element]
        y_min = min(y_min, subset[y].min())
        ax.plot(subset[x], subset[y], label=element)

    # Set x-axis date format and locator
        if style.x_formatter is not None:
            x_min = data[x].min()
            x_max = data[x].max()

            if style.x_formatter == "year_month_formatter":
                ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
            else:
                ax.xaxis.set_major_formatter(
                    DynamicFuncFormatter(style.x_formatter))
                ax.set_xticks(generate_ticks(
                    x_min, x_max, num_ticks=style.x_ticks))

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Set title and labels
    ax.set_xlabel(x)
    y_max = data[y].dropna().quantile(0.95)

    ax.set_ylim(y_min, y_max)
    ax.set_ylabel(y)
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
