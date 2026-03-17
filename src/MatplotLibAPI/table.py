"""Table plotting helpers."""

from typing import Any, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from matplotlib.table import Table

from .base_plot import BasePlot

from .utils import _get_axis

from .style_template import (
    FIG_SIZE,
    TITLE_SCALE_FACTOR,
    TABLE_STYLE_TEMPLATE,
    StyleTemplate,
    string_formatter,
    validate_dataframe,
)

__all__ = ["TABLE_STYLE_TEMPLATE", "aplot_table", "fplot_table"]


def _format_table(table: Table, style: StyleTemplate):
    """Format the table cells and font.

    Parameters
    ----------
    table : matplotlib.table.Table
        The table object to format.
    style : StyleTemplate
        The style configuration to apply.
    """
    table.auto_set_font_size(False)
    table.set_fontsize(style.font_size)
    table.scale(1.2, 1.2)

    for key, cell in table.get_celld().items():
        cell.set_fontsize(style.font_size)
        cell.set_facecolor(style.background_color)
        cell.get_text().set_color(style.font_color)


class TablePlot(BasePlot):
    """Class for plotting tables."""

    def __init__(self, pd_df: pd.DataFrame, cols: List[str]):
        validate_dataframe(pd_df, cols=cols)
        super().__init__(pd_df=pd_df)
        self.cols = cols

    def aplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = TABLE_STYLE_TEMPLATE,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        max_values: int = 20,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Axes:
        plot_ax = _get_axis(ax)

        if sort_by is None:
            sort_by = self.cols[0]

        plot_df = (
            self._obj[self.cols]
            .sort_values(by=[sort_by], ascending=ascending)  # type: ignore
            .head(max_values)
            .copy()
        )

        if style.format_funcs:
            for col, func in style.format_funcs.items():
                if col in plot_df.columns and func is not None:
                    plot_df[col] = plot_df[col].apply(func)

        table_plot = plot_ax.table(
            cellText=plot_df.values.tolist(),
            colLabels=[string_formatter(colLabel) for colLabel in self.cols],
            cellLoc="center",
            colWidths=style.col_widths,
            bbox=Bbox.from_bounds(0, -0.3, 1, 1.3),
        )

        _format_table(table_plot, style)

        plot_ax.set_facecolor(style.background_color)
        plot_ax.set_axis_off()
        plot_ax.grid(False)
        if title:
            plot_ax.set_title(
                title,
                color=style.font_color,
                fontsize=style.font_size * TITLE_SCALE_FACTOR,
            )
            plot_ax.title.set_position((0.5, 1.05))
        return plot_ax

    def fplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = TABLE_STYLE_TEMPLATE,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        max_values: int = 20,
        figsize: Tuple[float, float] = FIG_SIZE,
    ) -> Figure:
        fig = Figure(
            figsize=figsize,
            facecolor=style.background_color,
            edgecolor=style.background_color,
        )
        ax = Axes(fig=fig, facecolor=style.background_color)
        self.aplot(
            title=title,
            style=style,
            sort_by=sort_by,
            ascending=ascending,
            max_values=max_values,
            ax=ax,
        )
        return fig


def aplot_table(
    pd_df: pd.DataFrame,
    cols: List[str],
    title: Optional[str] = None,
    style: StyleTemplate = TABLE_STYLE_TEMPLATE,
    max_values: int = 20,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Render a table into the provided axes.

    Parameters
    ----------
    pd_df : pandas.DataFrame
        DataFrame containing the data to display.
    cols : list of str
        Columns to include in the table.
    title : str, optional
        Table title, by default ``None``.
    style : StyleTemplate, optional
        Style configuration, by default ``TABLE_STYLE_TEMPLATE``.
    max_values : int, optional
        Maximum number of rows to display, by default ``20``.
    sort_by : str, optional
        Column used for sorting, by default ``None``.
    ascending : bool, optional
        Sort order for the data, by default ``False``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on, by default ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        Matplotlib axes containing the rendered table.

    Raises
    ------
    AttributeError
        If required columns are not in the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from MatplotLibAPI.Table import aplot_table
    >>> data = {'col1': [1, 2], 'col2': [3, 4]}
    >>> df = pd.DataFrame(data)
    >>> fig, ax = plt.subplots()
    >>> aplot_table(df, cols=['col1', 'col2'], ax=ax)
    """
    return TablePlot(pd_df=pd_df, cols=cols).aplot(
        title=title,
        style=style,
        sort_by=sort_by,
        ascending=ascending,
        max_values=max_values,
        ax=ax,
        **kwargs,
    )


def fplot_table(
    pd_df: pd.DataFrame,
    cols: List[str],
    title: Optional[str] = None,
    style: StyleTemplate = TABLE_STYLE_TEMPLATE,
    max_values: int = 20,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    figsize: Tuple[float, float] = FIG_SIZE,
) -> Figure:
    """Return a new figure containing a formatted table.

    Parameters
    ----------
    pd_df : pandas.DataFrame
        DataFrame containing the data to display.
    cols : list of str
        Columns to include in the table.
    title : str, optional
        Table title, by default ``None``.
    style : StyleTemplate, optional
        Style configuration, by default ``TABLE_STYLE_TEMPLATE``.
    max_values : int, optional
        Maximum number of rows to display, by default ``20``.
    sort_by : str, optional
        Column used for sorting, by default ``None``.
    ascending : bool, optional
        Sort order for the data, by default ``False``.
    figsize : tuple of float, optional
        Size of the created figure, by default ``FIG_SIZE``.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure containing the table.

    Raises
    ------
    AttributeError
        If required columns are not in the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> from MatplotLibAPI.Table import fplot_table
    >>> data = {'col1': [1, 2], 'col2': [3, 4]}
    >>> df = pd.DataFrame(data)
    >>> fig = fplot_table(df, cols=['col1', 'col2'])
    """
    return TablePlot(pd_df=pd_df, cols=cols).fplot(
        title=title,
        style=style,
        sort_by=sort_by,
        ascending=ascending,
        max_values=max_values,
        figsize=figsize,
    )
