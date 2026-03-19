"""Shared utilities for matplotlib-based plotting helpers."""

from typing import Any, Callable, Dict, Optional, Tuple, cast

import matplotlib.pyplot as plt
from numpy import ndarray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing_extensions import Protocol

from .style_template import StyleTemplate


class _AplotFunc(Protocol):
    def __call__(self, *, pd_df: Any, ax: Axes, **kwargs: Any) -> Axes: ...


def _get_axis(ax: Optional[Axes] = None) -> Axes:
    """Return a Matplotlib axes, defaulting to the current one."""
    return ax if ax is not None else plt.gca()


def _merge_kwargs(
    defaults: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Return a merged kwargs dictionary with caller overrides taking precedence.

    Parameters
    ----------
    defaults : dict[str, Any]
        Default keyword arguments.
    overrides : dict[str, Any], optional
        Caller-provided keyword arguments that should override defaults.

    Returns
    -------
    dict[str, Any]
        Merged keyword arguments.
    """
    merged = defaults.copy()
    if overrides:
        merged.update(overrides)
    return merged


def create_fig(
    figsize: Tuple[float, float], style: StyleTemplate
) -> Tuple[Figure, Axes]:
    fig = Figure(
        figsize=figsize,
        facecolor=style.background_color,
        edgecolor=style.background_color,
    )
    ax = fig.add_subplot(111)
    ax.set_facecolor(style.background_color)
    return fig, ax


def _wrap_aplot(
    plot_func: _AplotFunc,
    pd_df: Any,
    figsize: Tuple[float, float],
    style: StyleTemplate,
    ax_args: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Figure:
    """Create a new figure and delegate plotting to an axis-level function.

    Parameters
    ----------
    plot_func : _AplotFunc
        Axis-level plotting callable.
    pd_df : Any
        Data passed to the plotting function.
    figsize : tuple[float, float]
        Size of the created figure.
    ax_args : dict, optional
        Additional keyword arguments forwarded to ``plt.subplots``.
    **kwargs : Any
        Additional arguments forwarded to ``plot_func``.

    Returns
    -------
    Figure
        Figure containing the rendered plot.
    """
    ax_args = ax_args or {}
    fig, axes_obj = plt.subplots(figsize=figsize, **ax_args)
    fig_obj: Figure = cast(Figure, fig)
    fig_obj.patch.set_facecolor(style.background_color)
    ax: Axes
    if isinstance(axes_obj, Axes):
        ax = axes_obj
    else:
        ax = cast(Axes, axes_obj.flat[0] if isinstance(axes_obj, ndarray) else axes_obj)
    plot_func(pd_df=pd_df, ax=ax, **kwargs)

    return fig_obj
