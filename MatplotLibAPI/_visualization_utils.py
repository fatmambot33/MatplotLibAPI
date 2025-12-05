"""Shared utilities for matplotlib-based plotting helpers."""

from typing import Any, Callable, Dict, Optional, Tuple, cast

import matplotlib.pyplot as plt
from numpy import ndarray
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
    save_path: Optional[str] = None,
    savefig_kwargs: Optional[Dict[str, Any]] = None,
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
    save_path : str, optional
        File path where the figure should be saved. The default is ``None``
        and no file is written.
    savefig_kwargs : dict, optional
        Extra keyword arguments forwarded to ``Figure.savefig`` when
        ``save_path`` is provided. Defaults to ``None``.
    **kwargs : Any
        Additional arguments forwarded to ``plot_func``.

    Returns
    -------
    Figure
        Figure containing the rendered plot. If ``save_path`` is supplied the
        figure is saved before being returned.
    """
    ax_args = ax_args or {}
    fig, axes_obj = plt.subplots(figsize=figsize, **ax_args)
    ax: Axes
    if isinstance(axes_obj, Axes):
        ax = axes_obj
    else:
        ax = cast(Axes, axes_obj.flat[0] if isinstance(axes_obj, ndarray) else axes_obj)
    plot_func(pd_df=pd_df, ax=ax, **kwargs)
    fig_obj: Figure = cast(Figure, fig)
    if save_path:
        fig_obj.savefig(save_path, **(savefig_kwargs or {}))
    return fig_obj
