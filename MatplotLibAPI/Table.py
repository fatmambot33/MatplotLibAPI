"""Table plotting helpers."""

from typing import List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from .StyleTemplate import StyleTemplate, string_formatter, validate_dataframe

TABLE_STYLE_TEMPLATE = StyleTemplate(
    background_color="black", fig_border="darkgrey", font_color="white", palette="magma"
)


def _prepare_table_data(
    pd_df: pd.DataFrame,
    cols: List[str],
    sort_by: Optional[str],
    ascending: bool,
    max_values: int,
    style: StyleTemplate,
) -> pd.DataFrame:
    """Prepare data for table plotting."""
    validate_dataframe(pd_df, cols=cols, sort_by=sort_by)

    if not sort_by:
        sort_by = cols[0]

    plot_df = (
        pd_df[cols].sort_values(by=sort_by, ascending=ascending).head(max_values)
    )  # type: ignore

    if style.format_funcs:
        for col, func in style.format_funcs.items():
            if col in plot_df.columns and func is not None:
                plot_df[col] = plot_df[col].apply(func)
    return plot_df


def _format_table(table, style: StyleTemplate):
    """Format the table cells and font."""
    table.auto_set_font_size(False)
    table.set_fontsize(style.font_size)
    table.scale(1.2, 1.2)

    for key, cell in table.get_celld().items():
        cell.set_fontsize(style.font_size)
        cell.set_facecolor(style.background_color)
        cell.get_text().set_color(style.font_color)


def aplot_table(
    pd_df: pd.DataFrame,
    cols: List[str],
    title: Optional[str] = None,
    style: StyleTemplate = TABLE_STYLE_TEMPLATE,
    max_values: int = 20,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    ax: Optional[Axes] = None,
) -> Axes:
    """Render a table into the provided axes.

    Parameters
    ----------
    pd_df : pd.DataFrame
        DataFrame containing the data to display.
    cols : list[str]
        Columns to include in the table.
    title : str, optional
        Table title. Default is None.
    style : StyleTemplate, optional
        Style configuration. Default is TABLE_STYLE_TEMPLATE.
    max_values : int, optional
        Maximum number of rows to display. Default is 20.
    sort_by : str, optional
        Column used for sorting. Default is None.
    ascending : bool, optional
        Sort order for the data. Default is False.
    ax : Axes, optional
        Axes to draw on. Default is None.

    Returns
    -------
    Axes
        Matplotlib axes containing the rendered table.
    """
    if ax is None:
        ax = plt.gca()

    plot_df = _prepare_table_data(pd_df, cols, sort_by, ascending, max_values, style)

    table_plot = ax.table(
        cellText=plot_df.values.tolist(),
        colLabels=[string_formatter(colLabel) for colLabel in cols],
        cellLoc="center",
        colWidths=style.col_widths,
        bbox=Bbox.from_bounds(0, -0.3, 1, 1.3),
    )

    _format_table(table_plot, style)

    ax.set_facecolor(style.background_color)
    ax.set_axis_off()
    ax.grid(False)
    if title:
        ax.set_title(title, color=style.font_color, fontsize=style.font_size * 2)
        ax.title.set_position((0.5, 1.05))
    return ax


def fplot_table(
    pd_df: pd.DataFrame,
    cols: List[str],
    title: Optional[str] = None,
    style: StyleTemplate = TABLE_STYLE_TEMPLATE,
    max_values: int = 20,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    figsize: Tuple[float, float] = (19.2, 10.8),
) -> Figure:
    """Return a new figure containing a formatted table.

    Parameters
    ----------
    pd_df : pd.DataFrame
        DataFrame containing the data to display.
    cols : list[str]
        Columns to include in the table.
    title : str, optional
        Table title. Default is None.
    style : StyleTemplate, optional
        Style configuration. Default is TABLE_STYLE_TEMPLATE.
    max_values : int, optional
        Maximum number of rows to display. Default is 20.
    sort_by : str, optional
        Column used for sorting. Default is None.
    ascending : bool, optional
        Sort order for the data. Default is False.
    figsize : tuple[float, float], optional
        Size of the created figure. Default is (19.2, 10.8).

    Returns
    -------
    Figure
        Matplotlib figure containing the table.
    """
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(style.background_color)
    ax = fig.add_subplot()
    ax = aplot_table(pd_df, cols, title, style, max_values, sort_by, ascending, ax)
    return fig
