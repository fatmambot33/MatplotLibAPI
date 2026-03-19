"""Abstract base class for all plot types."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .style_template import StyleTemplate, FIG_SIZE


class BasePlot(ABC):
    """Base class defining the interface for all plot types.

    This abstract base class ensures consistency across all plot implementations
    by requiring both accessor-based (aplot_*) and figure-based (fplot_*) methods.

    Methods
    -------
    aplot
        Plot on an existing Matplotlib axes.
    fplot
        Plot on a new Matplotlib figure.
    """

    def __init__(self, pd_df: pd.DataFrame):
        """Initialize the BasePlot."""
        super().__init__()
        self._obj = pd_df

    @abstractmethod
    def aplot(self, *args: Any, **kwargs: Any) -> Axes:
        """Plot on an existing Matplotlib axes.

        Subclasses should implement plot-specific parameters as needed.
        Common parameters include title, style, and ax.

        Parameters
        ----------
        *args : Any
            Plot-specific positional arguments.
        **kwargs : Any
            Plot-specific keyword arguments. May include:
            - title : str, optional
                Chart title.
            - style : StyleTemplate, optional
                Styling template.
            - ax : Axes, optional
                Matplotlib axes to plot on. If None, uses the current axes.
            - Additional plot-specific parameters.

        Returns
        -------
        Axes
            The Matplotlib axes object with the plot.
        """

    def fplot_w(self, *args: Any, **kwargs: Any) -> Figure:
        style: StyleTemplate = kwargs.get("style", StyleTemplate())
        fig, ax = BasePlot.create_fig(
            figsize=kwargs.get("figsize", FIG_SIZE),
            style=style,
        )
        self.aplot(*args, {**kwargs, "ax": ax, "style": style})

        return fig

    @classmethod
    def create_fig(
        cls, figsize: Tuple[float, float], style: StyleTemplate
    ) -> Tuple[Figure, Axes]:
        fig = Figure(
            figsize=figsize,
            facecolor=style.background_color,
            edgecolor=style.background_color,
        )
        ax = fig.add_subplot(111)
        ax.set_facecolor(style.background_color)
        return fig, ax
