
from dataclasses import dataclass

from typing import List, Optional, Union

import numpy as np
import pandas as pd


from matplotlib.axes import Axes
from matplotlib.dates import num2date
from matplotlib.ticker import FuncFormatter


# region Style


@dataclass
class StyleTemplate:
    fig_background_color: str = 'skyblue'
    fig_border: str = 'steelblue'
    font_name: str = 'Arial'
    font_size: int = 10
    font_color: str = 'black'
    palette: str = 'rocket'
    legend: bool = True
    x_formatter: Optional[str] = None
    x_ticks: int = 10
    y_formatter: Optional[str] = None
    y_ticks: int = 5


KILO_STYLE_TEMPLATE = StyleTemplate(
    fig_background_color='white',
    fig_border='lightgrey',
    palette='viridis',
    y_formatter="kilo_formatter"
)

PERCENT_STYLE_TEMPLATE = StyleTemplate(
    fig_background_color='white',
    fig_border='lightgrey',
    palette='viridis',
    y_formatter="percent_formatter"
)
BUBBLE_STYLE_TEMPLATE = StyleTemplate(
    fig_background_color='white',
    fig_border='lightgrey',
    palette='rocket',
    x_formatter="integer_formatter",
    y_formatter="percent_formatter",
    y_ticks=5
)
TIMESERIES_STYLE_TEMPLATE = StyleTemplate(
    fig_background_color='white',
    fig_border='lightgrey',
    palette='viridis',
    x_formatter="year_month_formatter",
    y_formatter="percent_formatter"
)
PIVOTLINES_STYLE_TEMPLATE = StyleTemplate(
    fig_background_color='white',
    fig_border='lightgrey',
    palette='viridis',
    x_formatter="year_month_formatter",
    y_formatter="percent_formatter"
)
PIVOTBARS_STYLE_TEMPLATE = StyleTemplate(
    fig_background_color='white',
    fig_border='lightgrey',
    palette='viridis',
    x_formatter="year_month_formatter",
    y_formatter="kilo_formatter"
)


DARK_STYLE_TEMPLATE = StyleTemplate(
    fig_background_color='black',
    fig_border='darkgrey',
    font_color='white',
    palette='magma'
)


def generate_ticks(min_val, max_val, num_ticks="10"):
    # Identify the type of the input
    try:
        min_val = float(min_val)
        max_val = float(max_val)
        is_date = False
    except ValueError:
        is_date = True

    # Convert string inputs to appropriate numerical or date types
    num_ticks = int(num_ticks)

    if is_date:
        min_val = pd.Timestamp(min_val).to_datetime64()
        max_val = pd.Timestamp(max_val).to_datetime64()
        data_range = (max_val - min_val).astype('timedelta64[D]').astype(int)
    else:
        data_range = max_val - min_val

    # Calculate a nice step size
    step_size = data_range / (num_ticks - 1)

    # If date, convert back to datetime
    if is_date:
        ticks = pd.date_range(
            start=min_val, periods=num_ticks, freq=f"{step_size}D")
    else:
        # Round the step size to a "nice" number
        exponent = np.floor(np.log10(step_size))
        fraction = step_size / 10**exponent
        nice_fraction = round(fraction)

        # Create nice step size
        nice_step = nice_fraction * 10**exponent

        # Generate the tick marks based on the nice step size
        ticks = np.arange(min_val, max_val + nice_step, nice_step)

    return ticks


class DynamicFuncFormatter(FuncFormatter):
    def __init__(self, func_name):
        self.func = globals()[func_name]
        super().__init__(self.func)


def kilo_formatter(value, pos):
    # Format values for better readability
    if value >= 1000000:
        return f"{value/1000000:.2f}M"
    elif value >= 10000:
        return f"{value/110000:.2f}K"
    else:
        return str(value)


def percent_formatter(x, pos):
    return f"{x * 100:.0f}%"


def integer_formatter(x, pos):
    return f"{x:.0f}"


def year_month_formatter(x, pos):
    return num2date(x).strftime('%Y-%m')
# endregion

# region Wrapper


def plot_func(plot_type, ax: Axes,
              data: pd.DataFrame,
              x_col: str,
              y_col: Union[str, List[str]],
              fig_title: Optional[str] = None,
              style:  Optional[StyleTemplate] = None,
              legend: bool = False,
              **kwargs):
    from .Bubble import plot_bubble
    from .Pivot import plot_bars, plot_lines
    from .TimeSeries import plot_timeseries
    if plot_type == "bubble":
        plot_bubble(ax=ax,
                    data=data,
                    x_col=x_col,
                    y_col=y_col,
                    fig_title=fig_title,
                    style=style,
                    legend=legend,
                    **kwargs)
    elif plot_type == "timeseries":
        plot_timeseries(ax=ax,
                        data=data,
                        x_col=x_col,
                        y_col=y_col,
                        fig_title=fig_title,
                        style=style,
                        legend=legend,
                        **kwargs)
    elif plot_type == "lines":
        plot_lines(ax=ax,
                   data=data,
                   x_col=x_col,
                   y_col=y_col,
                   fig_title=fig_title,
                   style=style,
                   **kwargs)
    elif plot_type == "bars":
        plot_bars(ax=ax,
                  data=data,
                  x_col=x_col,
                  y_col=y_col,
                  fig_title=fig_title,
                  style=style,
                  **kwargs)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")
# endregion
