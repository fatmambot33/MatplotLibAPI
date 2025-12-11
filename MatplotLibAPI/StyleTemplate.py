"""Common style utilities and formatters for plotting."""

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
from matplotlib.dates import num2date
from matplotlib.ticker import FuncFormatter

# Type alias for formatter functions compatible with matplotlib tick formatters
FormatterFunc = Callable[[Union[int, float, str], Optional[int]], str]


# region Validation


def validate_dataframe(
    pd_df: pd.DataFrame, cols: List[str], sort_by: Optional[str] = None
) -> None:
    """Ensure required columns and optional sort column exist in a DataFrame.

    Parameters
    ----------
    pd_df : pd.DataFrame
        The DataFrame to validate.
    cols : list[str]
        Required column names.
    sort_by : str, optional
        Optional column used for sorting.

    Raises
    ------
    AttributeError
        If any column is missing.
    """
    required_cols = set(cols)
    if sort_by:
        required_cols.add(sort_by)
    missing = required_cols - set(pd_df.columns)
    if missing:
        raise AttributeError(f"Missing columns in DataFrame: {missing}")


# endregion

# region Format Dispatcher


def format_func(
    format_funcs: Optional[Dict[str, Optional[FormatterFunc]]],
    label: Optional[str] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
    z: Optional[str] = None,
) -> Optional[Dict[str, Optional[FormatterFunc]]]:
    """Map shared formatters to specific keys if provided.

    Parameters
    ----------
    format_funcs : dict[str, FormatterFunc], optional
        Dictionary of formatting functions.
    label : str, optional
        Label column name.
    x : str, optional
        X-axis column name.
    y : str, optional
        Y-axis column name.
    z : str, optional
        Z-axis column name.

    Returns
    -------
    dict[str, FormatterFunc], optional
        Updated format function dictionary.
    """
    if not format_funcs:
        return None

    new_format_funcs = format_funcs.copy()

    for generic, specific in {"label": label, "x": x, "y": y, "z": z}.items():
        if specific and generic in new_format_funcs:
            new_format_funcs[specific] = new_format_funcs[generic]
    return new_format_funcs


# endregion

# region Style Constants

FIG_SIZE = (19.2, 10.8)
BACKGROUND_COLOR = "black"
TEXT_COLOR = "white"
PALETTE = "Greys_r"
FONT_SIZE = 14
TITLE_SCALE_FACTOR = 2
MAX_RESULTS = 50


# endregion

# region Style Template


@dataclass
class StyleTemplate:
    """Configuration container for plot styling options."""

    background_color: str = BACKGROUND_COLOR
    fig_border: str = BACKGROUND_COLOR
    font_name: str = "Arial"
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
        """Compute progressive font sizes based on the base font.

        The mapping spans five emphasis levels, centered around ``font_size``.
        Each step is scaled to 15% of the base font (minimum step of 1) and
        clamped to a size of at least 1 point to avoid non-readable values for
        very small fonts.

        Returns
        -------
        dict[int, int]
            Level-to-font-size mapping where keys increase with size.
        """
        base_size = max(int(self.font_size), 1)
        step = max(int(math.ceil(base_size * 0.15)), 1)
        return {
            idx: max(base_size + offset * step, 1)
            for idx, offset in enumerate(range(-2, 3))
        }


# endregion

# region Custom Formatters


class DynamicFuncFormatter(FuncFormatter):
    """A wrapper for dynamic formatting functions."""

    def __init__(self, func_name: FormatterFunc):
        """Initialize the formatter.

        Parameters
        ----------
        func_name : FormatterFunc
            A formatting function.
        """
        super().__init__(func_name)


def percent_formatter(val: Union[int, float, str], pos: Optional[int] = None) -> str:
    """Format a value as a percentage."""
    if isinstance(val, str):
        val = float(val)
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


def string_formatter(val: Union[int, float, str], pos: Optional[int] = None) -> str:
    """Format a string to be title-case with spaces."""
    return str(val).replace("-", " ").replace("_", " ").title()


def yy_mm_formatter(x: float, pos: Optional[int] = None) -> str:
    """Format a float date value as YYYY-MM."""
    return num2date(x).strftime("%Y-%m")


def yy_mm_dd_formatter(x: float, pos: Optional[int] = None) -> str:
    """Format a float date value as YYYY-MM-DD."""
    return num2date(x).strftime("%Y-%m-%d")


# endregion

# region Tick Generator


def generate_ticks(
    min_val: Union[float, str, pd.Timestamp],
    max_val: Union[float, str, pd.Timestamp],
    num_ticks: int = 5,
) -> Union[np.ndarray, pd.DatetimeIndex]:
    """Generate evenly spaced ticks between min and max.

    Parameters
    ----------
    min_val : float | str | pd.Timestamp
        Minimum value of range.
    max_val : float | str | pd.Timestamp
        Maximum value of range.
    num_ticks : int
        Number of tick marks.

    Returns
    -------
    np.ndarray | pd.DatetimeIndex
        Tick values.
    """
    min_val_f: float = 0.0
    max_val_f: float = 0.0

    if isinstance(min_val, (int, float, str)) and isinstance(
        max_val, (int, float, str)
    ):
        try:
            min_val_f = float(min_val)
            max_val_f = float(max_val)
            is_date = False
        except (ValueError, TypeError):
            is_date = True
    else:
        is_date = True

    if is_date:
        min_ts = pd.Timestamp(min_val)
        max_ts = pd.Timestamp(max_val)
        if pd.isna(min_ts) or pd.isna(max_ts):  # type: ignore
            return pd.to_datetime([])
        if min_ts == max_ts:
            return pd.to_datetime([min_ts])
        return pd.date_range(start=min_ts, end=max_ts, periods=num_ticks)

    data_range = max_val_f - min_val_f
    raw_step = data_range / (num_ticks - 1)
    exponent = np.floor(np.log10(raw_step))
    nice_step = round(raw_step / 10**exponent) * 10**exponent
    return np.arange(min_val_f, max_val_f + nice_step, nice_step)


# endregion

# region Style Presets

BUBBLE_STYLE_TEMPLATE = StyleTemplate(
    format_funcs=cast(
        Dict[str, Optional[FormatterFunc]],
        {
            "label": string_formatter,
            "x": bmk_formatter,
            "y": percent_formatter,
            "z": bmk_formatter,
        },
    ),
    yscale="log",
)

TIMESERIE_STYLE_TEMPLATE = StyleTemplate(
    format_funcs=cast(
        Dict[str, Optional[FormatterFunc]],
        {"x": yy_mm_formatter, "y": bmk_formatter},
    )
)

TABLE_STYLE_TEMPLATE = StyleTemplate()

TREEMAP_STYLE_TEMPLATE = StyleTemplate()

PIVOTBARS_STYLE_TEMPLATE = StyleTemplate(
    format_funcs=cast(
        Dict[str, Optional[FormatterFunc]],
        {"y": percent_formatter, "label": string_formatter},
    ),
)
PIVOTLINES_STYLE_TEMPLATE = StyleTemplate(
    format_funcs=cast(
        Dict[str, Optional[FormatterFunc]],
        {"y": percent_formatter, "label": string_formatter},
    ),
)

NETWORK_STYLE_TEMPLATE = StyleTemplate()
DISTRIBUTION_STYLE_TEMPLATE = StyleTemplate()
HEATMAP_STYLE_TEMPLATE = StyleTemplate()
AREA_STYLE_TEMPLATE = StyleTemplate()
PIE_STYLE_TEMPLATE = StyleTemplate()
SANKEY_STYLE_TEMPLATE = StyleTemplate()
WORDCLOUD_STYLE_TEMPLATE = StyleTemplate()

# endregion
