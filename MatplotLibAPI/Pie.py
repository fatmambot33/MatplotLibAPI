"""Pie and donut chart helpers."""

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .StyleTemplate import PIE_STYLE_TEMPLATE, StyleTemplate, validate_dataframe
from ._visualization_utils import _get_axis, _wrap_aplot


def aplot_pie_donut(
    pd_df: pd.DataFrame,
    category: str,
    value: str,
    donut: bool = False,
    title: Optional[str] = None,
    style: StyleTemplate = PIE_STYLE_TEMPLATE,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Plot pie or donut charts for categorical share visualization."""
    validate_dataframe(pd_df, cols=[category, value])
    plot_ax = _get_axis(ax)
    labels = pd_df[category].astype(str).tolist()
    sizes = pd_df[value]

    wedgeprops: Optional[Dict[str, Any]] = None
    if donut:
        wedgeprops = {"width": 0.3}
    wedges, *_ = plot_ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=sns.color_palette(style.palette),
        wedgeprops=wedgeprops,
    )
    plot_ax.axis("equal")
    if title:
        plot_ax.set_title(title)
    return plot_ax


def fplot_pie_donut(
    pd_df: pd.DataFrame,
    category: str,
    value: str,
    donut: bool = False,
    title: Optional[str] = None,
    style: StyleTemplate = PIE_STYLE_TEMPLATE,
    figsize: Tuple[float, float] = (8, 8),
    save_path: Optional[str] = None,
    savefig_kwargs: Optional[Dict[str, Any]] = None,
) -> Figure:
    """Plot pie or donut charts on a new figure."""
    return _wrap_aplot(
        aplot_pie_donut,
        pd_df=pd_df,
        figsize=figsize,
        category=category,
        value=value,
        donut=donut,
        title=title,
        style=style,
        save_path=save_path,
        savefig_kwargs=savefig_kwargs,
    )
