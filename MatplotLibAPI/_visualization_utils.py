"""Shared utilities for matplotlib-based plotting helpers."""

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _get_axis(ax: Optional[Axes] = None) -> Axes:
    """Return a Matplotlib axes, defaulting to the current one."""
    return ax if ax is not None else plt.gca()


def _wrap_aplot(
    plot_func,
    pd_df,
    figsize: Tuple[float, float],
    ax_args: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Figure:
    """Create a new figure and delegate plotting to an axis-level function."""
    ax_args = ax_args or {}
    fig, ax = plt.subplots(figsize=figsize, **ax_args)
    plot_func(pd_df=pd_df, ax=ax, **kwargs)
    return fig
