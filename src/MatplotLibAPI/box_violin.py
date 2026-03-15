"""Box and violin plot helpers."""

from typing import Any, Dict, Optional, Tuple

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_plot import BasePlot

from .style_template import (
    DISTRIBUTION_STYLE_TEMPLATE,
    StyleTemplate,
    string_formatter,
    validate_dataframe,
)
from .utils import _get_axis, _wrap_aplot

__all__ = ["DISTRIBUTION_STYLE_TEMPLATE", "aplot_box_violin", "fplot_box_violin"]


class BoxViolinPlot(BasePlot):
    """Class for plotting box and violin charts."""

    def __init__(
        self,
        pd_df: pd.DataFrame,
        column: str,
        by: Optional[str] = None,
        violin: bool = False,
    ):
        super().__init__(pd_df=pd_df)
        self.column = column
        self.by = by
        self.violin = violin

        cols = [self.column]
        if self.by:
            cols.append(self.by)
        validate_dataframe(self._obj, cols=cols)

    def aplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Axes:

        plot_ax = _get_axis(ax)

        common_kwargs = {
            "data": self._obj,
            "x": self.by,
            "y": self.column,
            "palette": style.palette,
        }

        if self.violin:
            sns.violinplot(**common_kwargs, hue=self.by, legend=False, ax=plot_ax)
        else:
            sns.boxplot(**common_kwargs, hue=self.by, legend=False, ax=plot_ax)

        plot_ax.set_facecolor(style.background_color)
        plot_ax.set_ylabel(string_formatter(self.column))
        if self.by:
            plot_ax.set_xlabel(string_formatter(self.by))
        if title:
            plot_ax.set_title(title)
        return plot_ax

    def fplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
        figsize: Tuple[float, float] = (10, 6),
    ) -> Figure:
        return _wrap_aplot(
            aplot_box_violin,
            pd_df=self._obj,
            figsize=figsize,
            column=self.column,
            by=self.by,
            violin=self.violin,
            title=title,
            style=style,
        )


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
    return BoxViolinPlot(
        pd_df=pd_df,
        column=column,
        by=by,
        violin=violin,
    ).aplot(
        title=title,
        style=style,
        ax=ax,
        **kwargs,
    )


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
    return BoxViolinPlot(
        pd_df=pd_df,
        column=column,
        by=by,
        violin=violin,
    ).fplot(
        title=title,
        style=style,
        figsize=figsize,
    )
