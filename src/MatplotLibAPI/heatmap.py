"""Heatmap and correlation matrix helpers."""

from typing import Any, Optional, Sequence, Tuple, cast, Literal

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
from .utils import _get_axis, _merge_kwargs, create_fig

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

    def correlation_matrix(
        self,
        correlation_method: CorrelationMethod = "pearson",
    ) -> pd.DataFrame:
        """Compute the correlation matrix for the underlying DataFrame."""
        return self._obj.corr(method=correlation_method)

    def aplot(
        self,
        title: Optional[str] = None,
        style: Optional[StyleTemplate] = None,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot a heatmap on an existing Matplotlib axes."""
        if not style:
            style = HEATMAP_STYLE_TEMPLATE
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

    def aplot_correlation_matrix(
        self,
        title: Optional[str] = None,
        style: Optional[StyleTemplate] = None,
        correlation_method: CorrelationMethod = "pearson",
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot a correlation matrix heatmap on existing axes."""
        if not style:
            style = HEATMAP_STYLE_TEMPLATE
        plot_ax = _get_axis(ax)
        heatmap_kwargs: dict[str, Any] = {
            "data": self.correlation_matrix(correlation_method),
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
        style: Optional[StyleTemplate] = None,
        figsize: Tuple[float, float] = (10, 6),
        correlation_method: CorrelationMethod = "pearson",
    ) -> Figure:
        """Plot a correlation matrix heatmap on a new figure."""
        if not style:
            style = HEATMAP_STYLE_TEMPLATE
        fig, ax = create_fig(figsize=figsize, style=style)
        self.aplot_correlation_matrix(
            title=title,
            style=style,
            correlation_method=correlation_method,
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
    style: Optional[StyleTemplate] = None,
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
    x: str,
    y: str,
    value: str,
    correlation_method: CorrelationMethod = "pearson",
    title: Optional[str] = None,
    style: Optional[StyleTemplate] = None,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    return Heatmap(
        pd_df=pd_df,
        x=x,
        y=y,
        value=value,
    ).aplot_correlation_matrix(
        title=title, style=style, correlation_method=correlation_method, ax=ax
    )


def fplot_heatmap(
    pd_df: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    title: Optional[str] = None,
    style: Optional[StyleTemplate] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> Figure:
    """Plot a matrix heatmap on a new figure."""
    return Heatmap(
        pd_df=pd_df,
        x=x,
        y=y,
        value=value,
    ).fplot_w(
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
    style: Optional[StyleTemplate] = None,
    figsize: Tuple[float, float] = (10, 6),
    correlation_method: CorrelationMethod = "pearson",
) -> Figure:
    """Plot a correlation matrix heatmap on a new figure."""
    return Heatmap(
        pd_df=pd_df,
        x=x,
        y=y,
        value=value,
    ).fplot_correlation_matrix(
        title=title, style=style, figsize=figsize, correlation_method=correlation_method
    )
