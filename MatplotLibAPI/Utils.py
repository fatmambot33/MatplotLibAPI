
from dataclasses import dataclass

from typing import List, Optional, Union, Dict, Callable

import numpy as np
import pandas as pd

from matplotlib.dates import num2date
from matplotlib.ticker import FuncFormatter

# region Panda

def _validate_panda(pd_df: pd.DataFrame,
                cols: List[str]):
    for col in cols:
        if col not in pd_df.columns:
            raise AttributeError(f"{col} is not a DataFrame's column")

# endregion

# region Style

MAX_RESULTS = 50
X_COL = "index"
Y_COL = "overlap"
Z_COL = "users"
FIG_SIZE = (19.2, 10.8)
BACKGROUND_COLOR = 'black'
TEXT_COLOR = 'white'
PALETTE = "Greys_r"
FONT_SIZE = 14
FONT_SIZE_MAPPING = {0: FONT_SIZE-4, 1: FONT_SIZE -
                     2, 2: FONT_SIZE, 3: FONT_SIZE+2, 4: FONT_SIZE+4}


@dataclass
class StyleTemplate:
    background_color: str = BACKGROUND_COLOR
    fig_border: str = BACKGROUND_COLOR
    font_name: str = 'Arial'
    font_size: int = FONT_SIZE
    font_color: str = TEXT_COLOR
    palette: str = PALETTE
    legend: bool = True
    xscale: Optional[str] = None
    x_ticks: int = 10
    yscale: Optional[str] = None
    y_ticks: int = 5
    format_funcs: Optional[Dict[str, Optional[Callable[[
        Union[int, float, str]], str]]]] = None
    col_widths: Optional[List[float]] = None

    @property
    def font_mapping(self):
        return {0: self.font_size-3,
                1: self.font_size-1,
                2: self.font_size,
                3: self.font_size+1,
                4: self.font_size+3}


class DynamicFuncFormatter(FuncFormatter):
    def __init__(self, func_name):
        super().__init__(func_name)


def percent_formatter(val, pos: Optional[int] = None):
    if val*100 <= 0.1:  # For 0.1%
        return f"{val*100:.2f}%"
    elif val*100 <= 1:  # For 1%
        return f"{val*100:.1f}%"
    else:
        return f"{val*100:.0f}%"


def bmk_formatter(val, pos: Optional[int] = None):
    if val >= 1_000_000_000:  # Billions
        return f"{val / 1_000_000_000:.2f}B"
    elif val >= 1_000_000:  # Millions
        return f"{val / 1_000_000:.1f}M"
    elif val >= 1_000:  # Thousands
        return f"{val / 1_000:.1f}K"
    else:
        return f"{int(val)}"


def integer_formatter(value, pos: Optional[int] = None):
    # Example formatting function: here we simply return the value as a string with some prefix
    return f"{int(value)}"


def string_formatter(value, pos: Optional[int] = None):
    # Example formatting function: here we simply return the value as a string with some prefix
    return str(value).replace("-", " ").replace("_", " ").title()


def year_month_formatter(x, pos: Optional[int] = None):
    return num2date(x).strftime('%Y-%m')


def percent_formatter(x, pos):
    return f"{x * 100:.0f}%"


BUBBLE_STYLE_TEMPLATE = StyleTemplate(
    format_funcs={"label": string_formatter,
                  "x": bmk_formatter,
                  "y": percent_formatter,
                  "label": string_formatter,
                  "z": bmk_formatter, },
    y_ticks=5
)
TIMESERIE_STYLE_TEMPLATE = StyleTemplate(
    palette='rocket',
    format_funcs={"y": bmk_formatter, "label": string_formatter}
)
PIVOTLINES_STYLE_TEMPLATE = StyleTemplate(
    background_color='white',
    fig_border='lightgrey',
    palette='viridis',
    format_funcs={"y": percent_formatter, "label": string_formatter}
)
PIVOTBARS_STYLE_TEMPLATE = StyleTemplate(
    background_color='black',
    fig_border='darkgrey',
    font_color='white',
    palette='magma',
    format_funcs={"y": percent_formatter, "label": string_formatter}
)

TABLE_STYLE_TEMPLATE = StyleTemplate(
    background_color='black',
    fig_border='darkgrey',
    font_color='white',
    palette='magma'
)

NETWORK_STYLE_TEMPLATE = StyleTemplate(
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


# endregion
