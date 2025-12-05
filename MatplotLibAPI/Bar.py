"""Bar and stacked bar chart helpers."""

from typing import Optional, Tuple

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


def aplot_bar(
    pd_df: pd.DataFrame,
    category: str,
    value: str,
    group: Optional[str] = None,
    stacked: bool = False,
    title: Optional[str] = None,
    style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot bar or stacked bar charts for categorical comparisons."""
    cols = [category, value]
    if group:
        cols.append(group)
    validate_dataframe(pd_df, cols=cols)

    plot_ax = _get_axis(ax)
    plot_df = pd_df.copy()

    if group:
        pivot_df = plot_df.pivot_table(
            index=category, columns=group, values=value, aggfunc="sum"
        )
        pivot_df.plot(kind="bar", stacked=stacked, ax=plot_ax, alpha=0.85)
    else:
        sns.barplot(
            data=plot_df, x=category, y=value, palette=style.palette, ax=plot_ax
        )

    plot_ax.set_facecolor(style.background_color)
    plot_ax.set_xlabel(string_formatter(category))
    plot_ax.set_ylabel(string_formatter(value))
    if title:
        plot_ax.set_title(title)
    plot_ax.tick_params(axis="x", labelrotation=45)
    return plot_ax


def fplot_bar(
    pd_df: pd.DataFrame,
    category: str,
    value: str,
    group: Optional[str] = None,
    stacked: bool = False,
    title: Optional[str] = None,
    style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
    figsize: Tuple[float, float] = (10, 6),
) -> Figure:
    """Plot bar or stacked bar charts on a new figure."""
    return _wrap_aplot(
        aplot_bar,
        pd_df=pd_df,
        figsize=figsize,
        category=category,
        value=value,
        group=group,
        stacked=stacked,
        title=title,
        style=style,
    )
