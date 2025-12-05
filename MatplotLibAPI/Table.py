"""Table plotting helpers."""

from typing import Any, Dict, List, Optional, Tuple, cast
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from matplotlib.table import Table

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
    """Prepare data for table plotting.

    Parameters
    ----------
    pd_df : pd.DataFrame
        Input DataFrame.
    cols : List[str]
        Columns to include in the table.
    sort_by : Optional[str]
        Column to sort by.
    ascending : bool
        Sort order.
    max_values : int
        Maximum number of rows to display.
    style : StyleTemplate
        Styling for the plot.

    Returns
    -------
    pd.DataFrame
        Prepared DataFrame for plotting.

    Raises
    ------
    AttributeError
        If required columns are missing from the DataFrame.
    """
    validate_dataframe(pd_df, cols=cols, sort_by=sort_by)

    if sort_by is None:
        sort_by = cols[0]

    plot_df = (
        pd_df[cols]
        .sort_values(by=[sort_by], ascending=ascending)  # type: ignore
        .head(max_values)
        .copy()
    )

    if style.format_funcs:
        for col, func in style.format_funcs.items():
            if col in plot_df.columns and func is not None:
                plot_df[col] = plot_df[col].apply(func)
    return plot_df


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
    if ax is None:
        ax = cast(Axes, plt.gca())

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
    save_path: Optional[str] = None,
    savefig_kwargs: Optional[Dict[str, Any]] = None,
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
        Size of the created figure, by default ``(19.2, 10.8)``.

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
    fig = cast(Figure, plt.figure(figsize=figsize))
    fig.patch.set_facecolor(style.background_color)
    ax = fig.add_subplot()
    ax = aplot_table(pd_df, cols, title, style, max_values, sort_by, ascending, ax)
    if save_path:
        fig.savefig(save_path, **(savefig_kwargs or {}))
    return fig
