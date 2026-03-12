"""Heatmap and correlation matrix helpers."""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .style_template import (
    HEATMAP_STYLE_TEMPLATE,
    StyleTemplate,
    string_formatter,
    validate_dataframe,
)
from .utils import _get_axis, _wrap_aplot
from .typing import CorrelationMethod

__all__ = [
    "HEATMAP_STYLE_TEMPLATE",
    "aplot_heatmap",
    "aplot_correlation_matrix",
    "fplot_heatmap",
    "fplot_correlation_matrix",
]


def _prepare_treemap_data(
    pd_df: pd.DataFrame,
    x: str,
    y: str,
    value: str,
) -> pd.DataFrame:
    """Prepare data for treemap plotting."""
    validate_dataframe(pd_df, cols=[x, y, value])
    plot_df = pd_df[[x, y, value]].pivot_table(
        index=y, columns=x, values=value, aggfunc="mean"
    )
    return plot_df


def aplot_heatmap(
    pd_df: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    title: Optional[str] = None,
    style: StyleTemplate = HEATMAP_STYLE_TEMPLATE,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Plot a matrix heatmap for multivariate pattern detection."""
    plot_ax = _get_axis(ax)
    pivot_df = _prepare_treemap_data(pd_df, x, y, value)
    sns.heatmap(pivot_df, cmap=style.palette, ax=plot_ax)

    plot_ax.set_xlabel(string_formatter(x))
    plot_ax.set_ylabel(string_formatter(y))
    if title:
        plot_ax.set_title(title)
    return plot_ax


def aplot_correlation_matrix(
    pd_df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    method: CorrelationMethod = "pearson",
    title: Optional[str] = None,
    style: StyleTemplate = HEATMAP_STYLE_TEMPLATE,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Plot a correlation matrix heatmap for numeric columns."""
    subset = (
        columns
        if columns is not None
        else pd_df.select_dtypes(include=[np.number]).columns
    )
    if len(subset) == 0:
        raise AttributeError("No numeric columns available for correlation matrix")

    validate_dataframe(pd_df, cols=list(subset))
    plot_ax = _get_axis(ax)

    selected: pd.DataFrame = pd_df.loc[:, list(subset)]
    corr = selected.corr(method=method)  # pyright: ignore[reportArgumentType]
    sns.heatmap(corr, cmap=style.palette, annot=True, fmt=".2f", ax=plot_ax)
    if title:
        plot_ax.set_title(title)
    return plot_ax


def fplot_heatmap(
    pd_df: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    title: Optional[str] = None,
    style: StyleTemplate = HEATMAP_STYLE_TEMPLATE,
    figsize: Tuple[float, float] = (10, 6),
) -> Figure:
    """Plot a matrix heatmap on a new figure."""
    return _wrap_aplot(
        aplot_heatmap,
        pd_df=pd_df,
        figsize=figsize,
        x=x,
        y=y,
        value=value,
        title=title,
        style=style,
    )


def fplot_correlation_matrix(
    pd_df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    method: CorrelationMethod = "pearson",
    title: Optional[str] = None,
    style: StyleTemplate = HEATMAP_STYLE_TEMPLATE,
    figsize: Tuple[float, float] = (10, 6),
) -> Figure:
    """Plot a correlation matrix heatmap on a new figure."""
    return _wrap_aplot(
        aplot_correlation_matrix,
        pd_df=pd_df,
        figsize=figsize,
        columns=columns,
        method=method,
        title=title,
        style=style,
    )
