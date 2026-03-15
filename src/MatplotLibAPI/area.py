"""Area chart helpers."""

from typing import Any, Dict, Optional, Tuple

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_plot import BasePlot

from .style_template import (
    AREA_STYLE_TEMPLATE,
    StyleTemplate,
    string_formatter,
    validate_dataframe,
)
from .utils import _get_axis, _wrap_aplot

__all__ = ["AREA_STYLE_TEMPLATE", "aplot_area", "fplot_area"]


class AreaChart(BasePlot):
    """Class for plotting area charts."""

    def __init__(
        self,
        pd_df: pd.DataFrame,
        x: str,
        y: str,
        label: Optional[str] = None,
        stacked: bool = True,
    ):
        cols = [self.x, self.y]
        if self.label:
            cols.append(self.label)
        validate_dataframe(self._obj, cols=cols)

        super().__init__(pd_df=pd_df)
        self.x = x
        self.y = y
        self.label = label
        self.stacked = stacked

    def aplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = AREA_STYLE_TEMPLATE,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Axes:
        plot_ax = _get_axis(ax)

        if self.label:
            pivot_df = self._obj.pivot_table(
                index=self.x, columns=self.label, values=self.y, aggfunc="sum"
            ).sort_index()
            pivot_df.plot(kind="area", stacked=self.stacked, alpha=0.7, ax=plot_ax)
        else:
            sorted_df = self._obj.sort_values(by=self.x)
            plot_ax.fill_between(
                sorted_df[self.x], sorted_df[self.y], color=style.font_color, alpha=0.4
            )
            plot_ax.plot(sorted_df[self.x], sorted_df[self.y], color=style.font_color)

        plot_ax.set_xlabel(string_formatter(self.x))
        plot_ax.set_ylabel(string_formatter(self.y))
        if title:
            plot_ax.set_title(title)
        return plot_ax

    def fplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = AREA_STYLE_TEMPLATE,
        figsize: Tuple[float, float] = (10, 6),
        **kwargs: Any,
    ) -> Figure:
        return _wrap_aplot(
            self.aplot,
            pd_df=self._obj,
            figsize=figsize,
            title=title,
            style=style,
            **kwargs,
        )


def aplot_area(
    pd_df: pd.DataFrame,
    x: str,
    y: str,
    label: Optional[str] = None,
    stacked: bool = True,
    title: Optional[str] = None,
    style: StyleTemplate = AREA_STYLE_TEMPLATE,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Plot area charts, optionally stacked for part-to-whole trends."""
    return AreaChart(
        pd_df=pd_df,
        x=x,
        y=y,
        label=label,
        stacked=stacked,
    ).aplot(
        title=title,
        style=style,
        ax=ax,
        **kwargs,
    )


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
    return AreaChart(
        pd_df=pd_df,
        x=x,
        y=y,
        label=label,
        stacked=stacked,
    ).fplot(
        title=title,
        style=style,
        figsize=figsize,
    )
