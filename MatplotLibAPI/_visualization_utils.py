"""Shared utilities for matplotlib-based plotting helpers."""

from typing import Any, Callable, Dict, Optional, Tuple, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing_extensions import Protocol


class _AplotFunc(Protocol):
    def __call__(self, *, pd_df: Any, ax: Axes, **kwargs: Any) -> Axes: ...


def _get_axis(ax: Optional[Axes] = None) -> Axes:
    """Return a Matplotlib axes, defaulting to the current one."""
    return ax if ax is not None else plt.gca()


def _wrap_aplot(
    plot_func: _AplotFunc,
    pd_df: Any,
    figsize: Tuple[float, float],
    ax_args: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Figure:
    """Create a new figure and delegate plotting to an axis-level function."""
    ax_args = ax_args or {}
    fig, ax = plt.subplots(figsize=figsize, **ax_args)
    plot_func(pd_df=pd_df, ax=ax, **kwargs)
    return cast(Figure, fig)
