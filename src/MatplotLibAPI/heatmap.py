"""Heatmap and correlation matrix helpers."""

from typing import Any, Optional, Sequence, Tuple, cast

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas.api.extensions import register_dataframe_accessor

from .base_plot import BasePlot
from .style_template import (
    HEATMAP_STYLE_TEMPLATE,
    StyleTemplate,
    string_formatter,
    validate_dataframe,
)
from .typing import CorrelationMethod
from .utils import _get_axis, _merge_kwargs

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
        """Plot a heatmap on an existing Matplotlib axes."""
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
        """Plot a heatmap on a new Matplotlib figure."""
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
        """Plot a correlation matrix heatmap on existing axes."""
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
        """Plot a correlation matrix heatmap on a new figure."""
        fig = Figure(
            figsize=figsize,
            facecolor=style.background_color,
            edgecolor=style.background_color,
        )
        ax = fig.add_subplot(111)
        ax.set_facecolor(style.background_color)
        self.aplot_correlation_matrix(
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
    """Prepare data for heatmap plotting."""
    validate_dataframe(pd_df, cols=[x, y, value])
    plot_df = pd_df[[x, y, value]].pivot_table(
        index=y, columns=x, values=value, aggfunc="mean"
    )
    return plot_df


def _compute_correlation_matrix(
    pd_df: pd.DataFrame,
    columns: Optional[Sequence[str]],
    method: CorrelationMethod,
) -> pd.DataFrame:
    """Compute a correlation matrix from numeric dataframe columns."""
    source_df = pd_df[list(columns)] if columns else pd_df
    numeric_df = source_df.select_dtypes(include="number")
    if numeric_df.empty:
        raise ValueError("No numeric columns available to compute correlation matrix.")
    return numeric_df.corr(method=cast(Any, method))


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
    corr_df = _compute_correlation_matrix(pd_df=pd_df, columns=columns, method=method)
    plot_ax = _get_axis(ax)
    heatmap_kwargs: dict[str, Any] = {
        "data": corr_df,
        "cmap": style.palette,
        "annot": True,
        "fmt": ".2f",
        "ax": plot_ax,
    }
    sns.heatmap(**_merge_kwargs(heatmap_kwargs, kwargs))
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
        x=x,
        y=y,
        value=value,
    ).fplot_correlation_matrix(title=title, style=style, figsize=figsize)
