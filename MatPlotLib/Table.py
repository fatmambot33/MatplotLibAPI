import logging
from typing import List, Optional, Dict, Callable
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_table(ax: Axes,
               data: pd.DataFrame,
               mappings: Dict[str, Callable[[pd.Series], pd.Series]],
               sort_column: str = "INDEX",
               sort_ascending: bool = False,
               num_rows: int = None,
               fig_background_color: str = 'black',
               fig_border: str = 'white',
               font_name: str = 'Arial',
               font_size: int = 10,
               font_color="black",
               fig_title: Optional[str] = None,
               col_widths: Optional[List[float]] = None) -> Axes:
    """
    Plots a table using Matplotlib in the provided axis.

    Parameters:
        ax (Axes): The Matplotlib axis to plot the table in.
        data (pd.DataFrame): The pandas DataFrame containing the table data.
        mappings (dict): Dictionary mapping column names to functions that transform the column data.
        sort_column (str, optional): Column to sort the data by. Default is "INDEX".
        sort_ascending (bool, optional): Whether to sort in ascending order. Default is False.
        num_rows (int, optional): Number of rows to display. Default is 10.
        fig_background_color (str, optional): Background color of the figure. Default is 'skyblue'.
        fig_border (str, optional): Border color of the figure. Default is 'steelblue'.
        font_name (str, optional): Font name for the table cells. Default is 'Arial'.
        font_size (int, optional): Font size for the table cells. Default is 10.
        col_widths (list, optional): List of relative column widths. Default is None.

    Returns:
        Axes: The Matplotlib axis with the plotted table.
    """

    if num_rows is None:
        num_rows = len(data.index)
    cols = list(mappings.keys())
    plot_data = data[cols].copy().sort_values(
        by=sort_column, ascending=sort_ascending).head(num_rows).reset_index(drop=True)

    for col, func in mappings.items():
        plot_data[col] = plot_data[col].apply(func)
    if fig_title is not None:
        ax.text(0.5, 1.05,
                fig_title,
                va='top',
                ha='center',
                fontsize=font_size*1.5,
                fontname=font_name,
                color=font_color,
                transform=ax.transAxes)
    table = ax.table(cellText=plot_data.values, colLabels=plot_data.columns,
                     cellLoc='center', colWidths=col_widths, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    for key, cell in table.get_celld().items():
        cell.get_text().set_fontname(font_name)
        cell.get_text().set_color(font_color)
    table.scale(1, 4)
    table.auto_set_column_width(col=list(range(len(plot_data.columns))))
    ax.axis('off')

    return ax
