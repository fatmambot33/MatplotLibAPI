"""Heatmap and correlation matrix helpers."""

from typing import Any, Optional, Sequence, Tuple

import pandas as pd
from pandas.api.extensions import register_dataframe_accessor
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_plot import BasePlot

from .style_template import (
    HEATMAP_STYLE_TEMPLATE,
    StyleTemplate,
    string_formatter,
    validate_dataframe,
)
from .utils import _get_axis, _merge_kwargs
from .typing import CorrelationMethod

__all__ = [
    "HEATMAP_STYLE_TEMPLATE",
    "aplot_heatmap",
    "aplot_correlation_matrix",
    "fplot_heatmap",
    "fplot_correlation_matrix",
]


@register_dataframe_accessor("heatmap")
class Heatmap(BasePlot):
    """Class for plotting heatmaps and correlation matrices."""

    def __init__(
        self,
        pd_df: pd.DataFrame,
        x: str,
        y: str,
        value: str,
    ):
        self._obj = _prepare_data(pd_df, x, y, value)
        self.x = x
        self.y = y
        self.value = value

    @property
    def correlation_matrix(self) -> pd.DataFrame:
        """Compute the correlation matrix for the underlying DataFrame."""
        return self._obj.corr()

    def aplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = HEATMAP_STYLE_TEMPLATE,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Axes:
        plot_ax = _get_axis(ax)
        heatmap_kwargs: dict[str, Any] = {
            "data": self._obj,
            "cmap": style.palette,
            "ax": plot_ax,
        }
        sns.heatmap(**_merge_kwargs(heatmap_kwargs, kwargs))

        plot_ax.set_xlabel(string_formatter(self.x))
        plot_ax.set_ylabel(string_formatter(self.y))
        if title:
            plot_ax.set_title(title)
        return plot_ax

    def fplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = HEATMAP_STYLE_TEMPLATE,
        figsize: Tuple[float, float] = (10, 6),
    ) -> Figure:
        fig = Figure(
            figsize=figsize,
            facecolor=style.background_color,
            edgecolor=style.background_color,
        )
        ax = fig.add_subplot(111)
        ax.set_facecolor(style.background_color)
        self.aplot(title=title, style=style, ax=ax)
        return fig

    def aplot_correlation_matrix(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = HEATMAP_STYLE_TEMPLATE,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Axes:
        plot_ax = _get_axis(ax)
        heatmap_kwargs: dict[str, Any] = {
            "data": self.correlation_matrix,
            "cmap": style.palette,
            "annot": True,
            "fmt": ".2f",
            "ax": plot_ax,
        }
        sns.heatmap(**_merge_kwargs(heatmap_kwargs, kwargs))
        if title:
            plot_ax.set_title(title)
        return plot_ax

    def fplot_correlation_matrix(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = HEATMAP_STYLE_TEMPLATE,
        figsize: Tuple[float, float] = (10, 6),
    ) -> Figure:
        fig = Figure(
            figsize=figsize,
            facecolor=style.background_color,
            edgecolor=style.background_color,
        )
        ax = fig.add_subplot(111)
        ax.set_facecolor(style.background_color)
        self.aplot(
            title=title,
            style=style,
            ax=ax,
        )
        return fig


def _prepare_data(
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
    return Heatmap(
        pd_df=pd_df,
        x=x,
        y=y,
        value=value,
    ).aplot(
        title=title,
        style=style,
        ax=ax,
        **kwargs,
    )


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
    return Heatmap(
        pd_df=pd_df,
        x="",  # Placeholder since correlation matrix is square
        y="",  # Placeholder since correlation matrix is square
        value="",  # Placeholder since correlation matrix is computed internally
    ).aplot_correlation_matrix(
        method=method,
        title=title,
        style=style,
        ax=ax,
        columns=columns,
        **kwargs,
    )


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
    return Heatmap(
        pd_df=pd_df,
        x=x,
        y=y,
        value=value,
    ).fplot(
        title=title,
        style=style,
        figsize=figsize,
    )


def fplot_correlation_matrix(
    pd_df: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    title: Optional[str] = None,
    style: StyleTemplate = HEATMAP_STYLE_TEMPLATE,
    figsize: Tuple[float, float] = (10, 6),
) -> Figure:
    """Plot a correlation matrix heatmap on a new figure."""
    return Heatmap(
        pd_df=pd_df,
        x=x,  # Placeholder since correlation matrix is square
        y=y,  # Placeholder since correlation matrix is square
        value=value,  # Placeholder since correlation matrix is computed internally
    ).fplot_correlation_matrix(title=title, style=style, figsize=figsize)
