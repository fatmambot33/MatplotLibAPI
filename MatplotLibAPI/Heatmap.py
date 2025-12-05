"""Heatmap and correlation matrix helpers."""

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas._typing import CorrelationMethod

from .StyleTemplate import (
    HEATMAP_STYLE_TEMPLATE,
    StyleTemplate,
    string_formatter,
    validate_dataframe,
)
from ._visualization_utils import _get_axis, _wrap_aplot


def aplot_heatmap(
    pd_df: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    title: Optional[str] = None,
    style: StyleTemplate = HEATMAP_STYLE_TEMPLATE,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot a matrix heatmap for multivariate pattern detection."""
    validate_dataframe(pd_df, cols=[x, y, value])
    plot_ax = _get_axis(ax)

    pivot_df = pd_df.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
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
    corr = selected.corr(method=method)
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
    method: str = "pearson",
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
