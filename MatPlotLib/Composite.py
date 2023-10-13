# Hint for Visual Code Python Interactive window
# %%
import matplotlib.pyplot as plt
import pandas as pd
from .Bubble import plot_bubble
from .Table import plot_table
from typing import Callable, Dict, Optional, List
from matplotlib.axes import Axes


def plot_composite(data: pd.DataFrame,
                   sort_column: str,
                   mappings: Dict[str, Callable],
                   num_rows: int = 10,
                   font_size: int = 12,
                   fig_title: str = 'Bubble Plot',
                   fig_background_color: str = 'skyblue',
                   fig_border: str = 'steelblue',
                   font_name: str = 'Arial',
                   font_color: str = 'black') -> None:

    data['uniques_quintile'] = pd.qcut(data['uniques'], 5, labels=False)
    text_size_mapping = {0: 8, 1: 9, 2: 10, 3: 12, 4: 14}
    data["font_size"] = data['uniques_quintile'].map(text_size_mapping)

    data['audience_quintile'] = pd.qcut(data['audience'], 5, labels=False)
    data['INDEX_quintile'] = pd.qcut(data['INDEX'], 5, labels=False)
    # Adjust font size for better readability
    plt.rc('font', size=font_size)

    fig = plt.figure("Graph", figsize=(10, 10))

    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    plot_bubble(ax=ax0,
                data=data,
                font_size=font_size,
                fig_background_color=fig_background_color,
                fig_border=fig_border,
                font_name=font_name)
    ax0.set_title(fig_title)  # Add title
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    top_10 = data.sort_values(by="INDEX", ascending=False).head(10)

    plot_table(ax=ax1,
               data=top_10,
               mappings=mappings,
               sort_column=sort_column,
               num_rows=num_rows,
               fig_background_color=fig_background_color,
               fig_border=fig_border,
               font_name=font_name,
               font_size=font_size,
               font_color=font_color)
    ax1.set_title('Top Items')  # Add title
    ax1.set_axis_off()

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    worst_10 = data.sort_values(by="INDEX").head(10)
    plot_table(ax=ax2,
               data=worst_10,
               mappings=mappings,
               sort_column=sort_column,
               num_rows=num_rows,
               fig_background_color=fig_background_color,
               fig_border=fig_border,
               font_name=font_name,
               font_size=font_size,
               sort_ascending=True)
    ax2.set_title('Worst Items')  # Add title
    ax2.set_axis_off()

    fig.tight_layout()


def plot_composite_12(plot_func1, plot_func2, plot_func3,
                      data1, data2, data3,
                      metrics1, metrics2, metrics3,
                      highlights: Optional[List[str]] = None,
                      font_size: int = 12,
                      fig_title: str = 'Bubble Plot',
                      fig_background_color: str = 'skyblue',
                      fig_border: str = 'steelblue',
                      font_name: str = 'Arial',
                      font_color: str = 'black') -> None:

    # Create a new figure and define the grid
    fig = plt.figure(fig_title, figsize=(10, 10))
    axgrid = fig.add_gridspec(5, 4)

    # Create individual axes based on the grid
    ax0 = fig.add_subplot(axgrid[0:3, :])
    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax2 = fig.add_subplot(axgrid[3:, 2:])

    # Call the individual plot functions with the respective axes and data
    plot_func1(ax=ax0, data=data1, metrics=metrics1, highlights=highlights, font_size=font_size, fig_background_color=fig_background_color,
               fig_border=fig_border, font_name=font_name, font_color=font_color)
    plot_func2(ax=ax1, data=data2, metrics=metrics2, highlights=highlights, font_size=font_size, fig_background_color=fig_background_color,
               fig_border=fig_border, font_name=font_name, font_color=font_color)
    plot_func3(ax=ax2, data=data3, metrics=metrics3, highlights=highlights, font_size=font_size, fig_background_color=fig_background_color,
               fig_border=fig_border, font_name=font_name, font_color=font_color)

    fig.suptitle(fig_title, fontsize=16)
    fig.tight_layout()
    return fig
