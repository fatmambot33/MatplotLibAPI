from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from .Utils import (TABLE_STYLE_TEMPLATE, StyleTemplate)


def plot_table(pd_df: pd.DataFrame,
               cols: List[str],
               title: str = "test",
               style: StyleTemplate = TABLE_STYLE_TEMPLATE,
               max_values: int = 20,
               sort_by: Optional[str] = None,
               ascending: bool = False
               ) -> Axes:
    if not sort_by:
        sort_by = cols[0]
    plot_df = pd_df[cols].sort_values(
        by=sort_by, ascending=ascending).head(max_values)

    col_labels = cols

    if style.format_funcs:
        for col, func in style.format_funcs.items():
            plot_df[col] = plot_df[col].apply(func)

    def format_table(table):
        table.auto_set_font_size(False)
        table.set_fontsize(style.font_size)
        table.scale(1.2, 1.2)

        for key, cell in table.get_celld().items():
            cell.set_fontsize(style.font_size)
            cell.set_facecolor(style.background_color)
            cell.get_text().set_color(style.font_color)
        table.auto_set_font_size(False)
        table.set_fontsize(style.font_size)
        table.scale(1.2, 1.2)

    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # Table for Top 10
    table = plt.table(cellText=plot_df.values,
                      colLabels=col_labels,
                      cellLoc='center',
                      loc='center',
                      bbox=[0.05, 0.1, 0.4, 0.8],
                      colWidths=style.col_widths)
    format_table(table)

    ax.set_title(title,
                 color=style.font_color,
                 fontsize=style.font_size*2)
    return ax
