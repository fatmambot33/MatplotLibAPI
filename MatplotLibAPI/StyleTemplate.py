"""Common style utilities and formatters for plotting."""

from typing import List, Optional, Dict, Callable, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np
from matplotlib.dates import num2date
from matplotlib.ticker import FuncFormatter

# Type alias for formatter functions compatible with matplotlib tick formatters
FormatterFunc = Callable[[Union[int, float, str], Optional[int]], str]


# region Validation

def validate_dataframe(
    pd_df: pd.DataFrame,
    cols: List[str],
    sort_by: Optional[str] = None
) -> None:
    """Ensure required columns and optional sort column exist in a DataFrame.

    Args:
        pd_df (pd.DataFrame): The DataFrame to validate.
        cols (List[str]): Required column names.
        sort_by (Optional[str]): Optional column used for sorting.

    Raises:
        AttributeError: If any column is missing.
    """
    required_cols = set(cols)
    if sort_by:
        required_cols.add(sort_by)
    missing = required_cols - set(pd_df.columns)
    if missing:
        raise AttributeError(f"Missing columns in DataFrame: {missing}")


# region Format Dispatcher

def format_func(
    format_funcs: Optional[Dict[str, Optional[FormatterFunc]]],
    label: Optional[str] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
    z: Optional[str] = None
) -> Optional[Dict[str, Optional[FormatterFunc]]]:
    """Map shared formatters to specific keys if provided.

    Args:
        format_funcs (Optional[Dict[str, Optional[FormatterFunc]]]): Dictionary of formatting functions.
        label (Optional[str]): Label column name.
        x (Optional[str]): X-axis column name.
        y (Optional[str]): Y-axis column name.
        z (Optional[str]): Z-axis column name.

    Returns:
        Optional[Dict[str, Optional[FormatterFunc]]]: Updated format function dictionary.
    """
    if not format_funcs:
        return None

    for generic, specific in {"label": label, "x": x, "y": y, "z": z}.items():
        if specific and generic in format_funcs:
            format_funcs[specific] = format_funcs[generic]
    return format_funcs


# region Style Constants

FIG_SIZE = (19.2, 10.8)
BACKGROUND_COLOR = 'black'
TEXT_COLOR = 'white'
PALETTE = "Greys_r"
FONT_SIZE = 14


# region Style Template

@dataclass
class StyleTemplate:
    """Configuration container for plot styling options."""

    background_color: str = BACKGROUND_COLOR
    fig_border: str = BACKGROUND_COLOR
    font_name: str = 'Arial'
    font_size: int = FONT_SIZE
    font_color: str = TEXT_COLOR
    palette: str = PALETTE
    legend: bool = True
    xscale: Optional[str] = None
    x_ticks: int = 5
    yscale: Optional[str] = None
    y_ticks: int = 5
    format_funcs: Optional[Dict[str, Optional[FormatterFunc]]] = None
    col_widths: Optional[List[float]] = None

    @property
    def font_mapping(self) -> Dict[int, int]:
        """Map font levels to adjusted font sizes.

        Returns:
            Dict[int, int]: Level to font size mapping.
        """
        return {
            0: self.font_size - 3,
            1: self.font_size - 1,
            2: self.font_size,
            3: self.font_size + 1,
            4: self.font_size + 3,
        }


# region Custom Formatters

class DynamicFuncFormatter(FuncFormatter):
    """A wrapper for dynamic formatting functions."""

    def __init__(self, func_name: FormatterFunc):
        """Initialize the formatter.

        Args:
            func_name (FormatterFunc): A formatting function.
        """
        super().__init__(func_name)


def percent_formatter(val: float, pos: Optional[int] = None) -> str:
    """Format a value as a percentage."""
    val *= 100
    if val <= 0.1:
        return f"{val:.2f}%"
    elif val <= 1:
        return f"{val:.1f}%"
    return f"{val:.0f}%"


def bmk_formatter(val: float, pos: Optional[int] = None) -> str:
    """Format large numbers using B, M, or K suffixes."""
    if val >= 1_000_000_000:
        return f"{val / 1_000_000_000:.2f}B"
    elif val >= 1_000_000:
        return f"{val / 1_000_000:.1f}M"
    elif val >= 1_000:
        return f"{val / 1_000:.1f}K"
    return str(int(val))


def integer_formatter(val: float, pos: Optional[int] = None) -> str:
    """Format a value as an integer."""
    return str(int(val))


def string_formatter(val: Union[str, float], pos: Optional[int] = None) -> str:
    """Format a string to be title-case with spaces."""
    return str(val).replace("-", " ").replace("_", " ").title()


def yy_mm_formatter(x: float, pos: Optional[int] = None) -> str:
    """Format a float date value as YYYY-MM."""
    return num2date(x).strftime('%Y-%m')


def yy_mm_dd_formatter(x: float, pos: Optional[int] = None) -> str:
    """Format a float date value as YYYY-MM-DD."""
    return num2date(x).strftime('%Y-%m-%d')


# region Tick Generator

def generate_ticks(
    min_val: Union[float, str, pd.Timestamp],
    max_val: Union[float, str, pd.Timestamp],
    num_ticks: int = 5
) -> Union[np.ndarray, pd.DatetimeIndex]:
    """Generate evenly spaced ticks between min and max.

    Args:
        min_val (float | str | pd.Timestamp): Minimum value of range.
        max_val (float | str | pd.Timestamp): Maximum value of range.
        num_ticks (int): Number of tick marks.

    Returns:
        Union[np.ndarray, pd.DatetimeIndex]: Tick values.
    """
    min_val_f: float = 0.0
    max_val_f: float = 0.0

    try:
        min_val_f = float(min_val)
        max_val_f = float(max_val)
        is_date = False
    except (ValueError, TypeError):
        is_date = True

    if is_date:
        min_ts = pd.Timestamp(min_val)
        max_ts = pd.Timestamp(max_val)
        step = ((max_ts - min_ts) / (num_ticks - 1)).days
        return pd.date_range(start=min_ts, periods=num_ticks, freq=f"{step}D")

    data_range = max_val_f - min_val_f
    raw_step = data_range / (num_ticks - 1)
    exponent = np.floor(np.log10(raw_step))
    nice_step = round(raw_step / 10**exponent) * 10**exponent
    return np.arange(min_val_f, max_val_f + nice_step, nice_step)


# endregion
