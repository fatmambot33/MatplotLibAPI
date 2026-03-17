"""Weight scaling helpers for network plots."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from .constants import _WEIGHT_PERCENTILES


def _softmax(x: Iterable[float]) -> np.ndarray:
    """Compute softmax values for array-like input."""
    x_arr = np.array(x)
    shifted = x_arr - np.max(x_arr)
    exp_shifted = np.exp(shifted)
    return exp_shifted / exp_shifted.sum()


def _scale_weights(
    weights: Iterable[float],
    scale_min: float = 0,
    scale_max: float = 1,
    deciles: Optional[np.ndarray] = None,
) -> List[float]:
    """Scale weights into deciles within the given range."""
    weights_arr = np.array(list(weights), dtype=float)
    if weights_arr.size == 0:
        return []

    soft = _softmax(weights_arr)
    deciles_arr = (
        np.percentile(weights_arr, _WEIGHT_PERCENTILES) if deciles is None else deciles
    )

    scaled = np.zeros_like(weights_arr, dtype=float)
    edges = np.concatenate(([-np.inf], deciles_arr, [np.inf]))
    bins = np.digitize(weights_arr, edges) - 1

    for idx in range(10):
        mask = bins == idx
        if not np.any(mask):
            continue
        bin_soft = soft[mask]
        if bin_soft.max() - bin_soft.min() < 1e-12:
            scaled[mask] = scale_min + (scale_max - scale_min) * (idx / 9)
        else:
            normalized = (bin_soft - bin_soft.min()) / (bin_soft.max() - bin_soft.min())
            scaled[mask] = scale_min + (scale_max - scale_min) * (
                idx / 9 + normalized / 9
            )

    return scaled.tolist()


__all__ = ["_softmax", "_scale_weights"]
