"""Abstract base class for all plot types."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, cast

import pandas as pd
import matplotlib.pyplot as plt
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

    def fplot(self, *args: Any, **kwargs: Any) -> Figure:
        """Plot on a new figure using the axis-level implementation.

        Parameters
        ----------
        *args : Any
            Plot-specific positional arguments forwarded to ``aplot``.
        **kwargs : Any
            Plot-specific keyword arguments forwarded to ``aplot``.
            ``figsize`` is consumed to create the figure.

        Returns
        -------
        Figure
            The Matplotlib figure containing the rendered plot.
        """
        style: Optional[StyleTemplate] = kwargs.get("style")
        if style is None:
            style = StyleTemplate()
        plot_kwargs = {k: v for k, v in kwargs.items() if k != "figsize"}
        fig, ax = BasePlot.create_fig(
            figsize=kwargs.get("figsize", FIG_SIZE),
            style=style,
        )
        self.aplot(*args, **{**plot_kwargs, "ax": ax, "style": style})

        return fig

    @classmethod
    def create_fig(
        cls, figsize: Tuple[float, float], style: StyleTemplate
    ) -> Tuple[Figure, Axes]:
        """Create a figure and axis configured from the provided style.

        Parameters
        ----------
        figsize : tuple[float, float]
            Figure size in inches.
        style : StyleTemplate
            Style template used for figure and axes backgrounds.

        Returns
        -------
        tuple[Figure, Axes]
            Created Matplotlib figure and a single subplot axes.
        """
        fig_raw, ax_raw = plt.subplots(figsize=figsize)
        fig = cast(Figure, fig_raw)
        ax = cast(Axes, ax_raw)
        fig.set_facecolor(style.background_color)
        fig.set_edgecolor(style.background_color)
        ax.set_facecolor(style.background_color)
        return fig, ax
