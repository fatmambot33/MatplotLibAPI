"""Area chart helpers."""

from typing import Optional, Tuple

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .StyleTemplate import AREA_STYLE_TEMPLATE, StyleTemplate, string_formatter, validate_dataframe
from ._visualization_utils import _get_axis, _wrap_aplot


def aplot_area(
    pd_df: pd.DataFrame,
    x: str,
    y: str,
    label: Optional[str] = None,
    stacked: bool = True,
    title: Optional[str] = None,
    style: StyleTemplate = AREA_STYLE_TEMPLATE,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot area charts, optionally stacked for part-to-whole trends."""

    cols = [x, y]
    if label:
        cols.append(label)
    validate_dataframe(pd_df, cols=cols)
    plot_ax = _get_axis(ax)

    if label:
        pivot_df = pd_df.pivot_table(index=x, columns=label, values=y, aggfunc="sum").sort_index()
        pivot_df.plot(kind="area", stacked=stacked, alpha=0.7, ax=plot_ax)
    else:
        sorted_df = pd_df.sort_values(by=x)
        plot_ax.fill_between(sorted_df[x], sorted_df[y], color=style.font_color, alpha=0.4)
        plot_ax.plot(sorted_df[x], sorted_df[y], color=style.font_color)

    plot_ax.set_xlabel(string_formatter(x))
    plot_ax.set_ylabel(string_formatter(y))
    if title:
        plot_ax.set_title(title)
    return plot_ax


def fplot_area(
    pd_df: pd.DataFrame,
    x: str,
    y: str,
    label: Optional[str] = None,
    stacked: bool = True,
    title: Optional[str] = None,
    style: StyleTemplate = AREA_STYLE_TEMPLATE,
    figsize: Tuple[float, float] = (10, 6),
) -> Figure:
    """Plot area charts on a new figure."""

    return _wrap_aplot(
        aplot_area,
        pd_df=pd_df,
        figsize=figsize,
        x=x,
        y=y,
        label=label,
        stacked=stacked,
        title=title,
        style=style,
    )
