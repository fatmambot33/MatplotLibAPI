"""Box and violin plot helpers."""

from typing import Any, Optional, Tuple

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .StyleTemplate import (
    DISTRIBUTION_STYLE_TEMPLATE,
    StyleTemplate,
    string_formatter,
    validate_dataframe,
)
from ._visualization_utils import _get_axis, _wrap_aplot


def aplot_box_violin(
    pd_df: pd.DataFrame,
    column: str,
    by: Optional[str] = None,
    violin: bool = False,
    title: Optional[str] = None,
    style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Plot box or violin charts to summarize distributions."""
    cols = [column]
    if by:
        cols.append(by)
    validate_dataframe(pd_df, cols=cols)
    plot_ax = _get_axis(ax)

    if violin:
        sns.violinplot(data=pd_df, x=by, y=column, palette=style.palette, ax=plot_ax)
    else:
        sns.boxplot(data=pd_df, x=by, y=column, palette=style.palette, ax=plot_ax)

    plot_ax.set_facecolor(style.background_color)
    plot_ax.set_ylabel(string_formatter(column))
    if by:
        plot_ax.set_xlabel(string_formatter(by))
    if title:
        plot_ax.set_title(title)
    return plot_ax


def fplot_box_violin(
    pd_df: pd.DataFrame,
    column: str,
    by: Optional[str] = None,
    violin: bool = False,
    title: Optional[str] = None,
    style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
    figsize: Tuple[float, float] = (10, 6),
) -> Figure:
    """Plot box or violin charts on a new figure."""
    return _wrap_aplot(
        aplot_box_violin,
        pd_df=pd_df,
        figsize=figsize,
        column=column,
        by=by,
        violin=violin,
        title=title,
        style=style,
    )
