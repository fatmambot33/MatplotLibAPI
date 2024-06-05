from typing import List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .StyleTemplate import StyleTemplate,  string_formatter, validate_dataframe

TABLE_STYLE_TEMPLATE = StyleTemplate(
    background_color='black',
    fig_border='darkgrey',
    font_color='white',
    palette='magma'
)


def aplot_table(pd_df: pd.DataFrame,
               cols: List[str],
               title: Optional[str] = None,
               style: StyleTemplate = TABLE_STYLE_TEMPLATE,
               max_values: int = 20,
               sort_by: Optional[str] = None,
               ascending: bool = False,
               ax: Optional[Axes] = None
               ) -> Axes:
    validate_dataframe(pd_df, cols=cols, sort_by=sort_by)

    if not sort_by:
        sort_by = cols[0]

    plot_df = pd_df[cols].sort_values(
        by=sort_by, ascending=ascending).head(max_values)

    col_labels = cols

    if style.format_funcs:
        for col, func in style.format_funcs.items():
            if col in plot_df.columns:
                plot_df[col] = plot_df[col].apply(func)

    def format_table(table):
        table.auto_set_font_size(False)
        table.set_fontsize(style.font_size)
        table.scale(1.2, 1.2)

        for key, cell in table.get_celld().items():
            cell.set_fontsize(style.font_size)
            cell.set_facecolor(style.background_color)
            cell.get_text().set_color(style.font_color)

    if ax is None:
        ax = plt.gca()

    table_plot = ax.table(
        cellText=plot_df.values,
        colLabels=[string_formatter(colLabel) for colLabel in col_labels],
        cellLoc='center',
        colWidths=style.col_widths,
        bbox=[0, -0.3, 1, 1.3])
    format_table(table_plot)
    ax.set_facecolor(style.background_color)
    ax.set_axis_off()
    ax.grid(False)
    if title:
        ax.set_title(title, color=style.font_color, fontsize=style.font_size*2)
        ax.title.set_position([0.5, 1.05])
    return ax


def fplot_table(pd_df: pd.DataFrame,
               cols: List[str],
               title: Optional[str] = None,
               style: StyleTemplate = TABLE_STYLE_TEMPLATE,
               max_values: int = 20,
               sort_by: Optional[str] = None,
               ascending: bool = False,
               figsize: Tuple[float, float] = (19.2, 10.8)
               ) -> Figure:
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(style.background_color)
    ax = fig.add_subplot()
    ax = aplot_table(pd_df,
                    cols,
                    title,
                    style,
                    max_values,
                    sort_by,
                    ascending,
                    ax
                    )
    return fig
