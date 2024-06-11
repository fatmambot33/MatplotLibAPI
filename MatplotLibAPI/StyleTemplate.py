

from typing import List, Optional, Dict, Callable, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

from matplotlib.dates import num2date
from matplotlib.ticker import FuncFormatter

# region Utils


def validate_dataframe(pd_df: pd.DataFrame,
                       cols: List[str],
                       sort_by: Optional[str] = None):
    """
    Validate that specified columns exist in a pandas DataFrame and optionally check for a sorting column.

    Parameters:
    pd_df (pd.DataFrame): The pandas DataFrame to validate.
    cols (List[str]): A list of column names that must exist in the DataFrame.
    sort_by (Optional[str]): An optional column name that, if provided, must also exist in the DataFrame.

    Raises:
    AttributeError: If any of the specified columns or the sorting column (if provided) do not exist in the DataFrame.

    Example:
    >>> import pandas as pd
    >>> data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    >>> df = pd.DataFrame(data)
    >>> validate_dataframe(df, ['A', 'B'])  # No error
    >>> validate_dataframe(df, ['A', 'C'])  # Raises AttributeError
    >>> validate_dataframe(df, ['A'], sort_by='B')  # No error
    >>> validate_dataframe(df, ['A'], sort_by='C')  # Raises AttributeError
    """
    _columns = cols.copy()
    if sort_by and sort_by not in _columns:
        _columns.append(sort_by)
    for col in _columns:
        if col not in pd_df.columns:
            raise AttributeError(f"{col} is not a DataFrame's column")


def format_func(
        format_funcs: Optional[Dict[str, Optional[Callable[[Union[int, float, str]], str]]]],
        label: Optional[str] = None,
        x: Optional[str] = None,
        y: Optional[str] = None,
        z: Optional[str] = None):
    """
    Update the formatting functions for specified keys if they exist in the provided format functions dictionary.

    Parameters:
    format_funcs (Optional[Dict[str, Optional[Callable[[Union[int, float, str]], str]]]]): 
        A dictionary mapping keys to formatting functions. The keys can be 'label', 'x', 'y', and 'z'.
    label (Optional[str]): 
        The key to update with the 'label' formatting function from the dictionary.
    x (Optional[str]): 
        The key to update with the 'x' formatting function from the dictionary.
    y (Optional[str]): 
        The key to update with the 'y' formatting function from the dictionary.
    z (Optional[str]): 
        The key to update with the 'z' formatting function from the dictionary.

    Returns:
    Optional[Dict[str, Optional[Callable[[Union[int, float, str]], str]]]]:
        The updated dictionary with the specified keys pointing to their corresponding formatting functions.

    Example:
    >>> format_funcs = {
    ...     "label": lambda x: f"Label: {x}",
    ...     "x": lambda x: f"X-axis: {x}",
    ...     "y": lambda y: f"Y-axis: {y}",
    ... }
    >>> updated_funcs = format_func(format_funcs, label="new_label", x="new_x")
    >>> print(updated_funcs)
    {
        "label": lambda x: f"Label: {x}",
        "x": lambda x: f"X-axis: {x}",
        "y": lambda y: f"Y-axis: {y}",
        "new_label": lambda x: f"Label: {x}",
        "new_x": lambda x: f"X-axis: {x}",
    }
    """

    if label and "label" in format_funcs:
        format_funcs[label] = format_funcs["label"]
    if x and "x" in format_funcs:
        format_funcs[x] = format_funcs["x"]
    if y and "y" in format_funcs:
        format_funcs[y] = format_funcs["y"]
    if z and "z" in format_funcs:
        format_funcs[z] = format_funcs["z"]
    return format_funcs

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
    x_ticks: int = 5
    yscale: Optional[str] = None
    y_ticks: int = 5
    format_funcs: Optional[Dict[str, Optional[Callable[[
        Union[int, float, str]], str]]]] = None
    col_widths: Optional[List[float]] = None
    """
    A class to define style templates for data visualization with customizable attributes.

    Attributes:
    background_color (str): 
        The background color for the visualizations. Default is BACKGROUND_COLOR.
    fig_border (str): 
        The border color for the figures. Default is BACKGROUND_COLOR.
    font_name (str): 
        The name of the font to use. Default is 'Arial'.
    font_size (int): 
        The base size of the font. Default is FONT_SIZE.
    font_color (str): 
        The color of the font. Default is TEXT_COLOR.
    palette (str): 
        The color palette to use. Default is PALETTE.
    legend (bool): 
        A flag to determine if the legend should be displayed. Default is True.
    xscale (Optional[str]): 
        The scale type for the x-axis. Default is None.
    x_ticks (int): 
        The number of ticks on the x-axis. Default is 10.
    yscale (Optional[str]): 
        The scale type for the y-axis. Default is None.
    y_ticks (int): 
        The number of ticks on the y-axis. Default is 5.
    format_funcs (Optional[Dict[str, Optional[Callable[[Union[int, float, str]], str]]]]): 
        A dictionary mapping data keys to formatting functions. Default is None.
    col_widths (Optional[List[float]]): 
        A list of column widths. Default is None.

    Properties:
    font_mapping (dict): 
        A dictionary mapping font size levels to specific font sizes.

    Example:
    >>> template = StyleTemplate()
    >>> template.font_mapping
    {0: FONT_SIZE-3, 1: FONT_SIZE-1, 2: FONT_SIZE, 3: FONT_SIZE+1, 4: FONT_SIZE+3}
    """
    @property
    def font_mapping(self):
        return {0: self.font_size-3,
                1: self.font_size-1,
                2: self.font_size,
                3: self.font_size+1,
                4: self.font_size+3}


class DynamicFuncFormatter(FuncFormatter):
    """
    A class to create a dynamic function formatter for matplotlib plots.

    Inherits from:
    FuncFormatter: A base class from matplotlib for formatting axis ticks.

    Parameters:
    func_name (Callable): The function to be used for formatting.

    Example:
    >>> formatter = DynamicFuncFormatter(percent_formatter)
    """
    def __init__(self, func_name):
        super().__init__(func_name)


def percent_formatter(val, pos: Optional[int] = None):
    """
    Format a value as a percentage.

    Parameters:
    val (float): The value to format.
    pos (Optional[int]): The position (not used).

    Returns:
    str: The formatted percentage string.

    Example:
    >>> percent_formatter(0.005)
    '1%'
    """
    if val*100 <= 0.1:  # For 0.1%
        return f"{val*100:.2f}%"
    elif val*100 <= 1:  # For 1%
        return f"{val*100:.1f}%"
    else:
        return f"{val*100:.0f}%"


def bmk_formatter(val, pos: Optional[int] = None):
    """
    Format a value as billions, millions, or thousands.

    Parameters:
    val (float): The value to format.
    pos (Optional[int]): The position (not used).

    Returns:
    str: The formatted string with B, M, or K suffix.

    Example:
    >>> bmk_formatter(1500000)
    '1.5M'
    """
    if val >= 1_000_000_000:  # Billions
        return f"{val / 1_000_000_000:.2f}B"
    elif val >= 1_000_000:  # Millions
        return f"{val / 1_000_000:.1f}M"
    elif val >= 1_000:  # Thousands
        return f"{val / 1_000:.1f}K"
    else:
        return f"{int(val)}"


def integer_formatter(value, pos: Optional[int] = None):
    """
    Format a value as an integer.

    Parameters:
    value (float): The value to format.
    pos (Optional[int]): The position (not used).

    Returns:
    str: The formatted integer string.

    Example:
    >>> integer_formatter(42.9)
    '42'
    """
    return f"{int(value)}"


def string_formatter(value, pos: Optional[int] = None):
    """
    Format a string by replacing '-' and '_' with spaces and capitalizing words.

    Parameters:
    value (str): The string to format.
    pos (Optional[int]): The position (not used).

    Returns:
    str: The formatted string.

    Example:
    >>> string_formatter("example-string_formatter")
    'Example String Formatter'
    """
    return str(value).replace("-", " ").replace("_", " ").title()


def yy_mm__formatter(x, pos: Optional[int] = None):
    """
    Format a date as 'YYYY-MM'.

    Parameters:
    x (float): The value to format.
    pos (Optional[int]): The position (not used).

    Returns:
    str: The formatted date string.

    Example:
    >>> yy_mm__formatter(737060)
    '2020-01'
    """
    return num2date(x).strftime('%Y-%m')


def yy_mm_dd__formatter(x, pos: Optional[int] = None):
    """
    Format a date as 'YYYY-MM-DD'.

    Parameters:
    x (float): The value to format.
    pos (Optional[int]): The position (not used).

    Returns:
    str: The formatted date string.

    Example:
    >>> yy_mm_dd__formatter(737060)
    '2020-01-01'
    """
    return num2date(x).strftime('%Y-%m-%D')


def generate_ticks(min_val, max_val, num_ticks:int=5):
    """
    Generate tick marks for a given range.

    Parameters:
    min_val (Union[float, str]): The minimum value of the range.
    max_val (Union[float, str]): The maximum value of the range.
    num_ticks (int): The number of ticks to generate. Default is 10.

    Returns:
    np.ndarray: An array of tick marks.

    Example:
    >>> generate_ticks(0, 100, 5)
    array([  0.,  25.,  50.,  75., 100.])
    """
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
