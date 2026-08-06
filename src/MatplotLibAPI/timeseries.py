"""Canonical time-series plotting module."""

from .timeserie import fplot_timeserie as fplot_timeseries

# Compatibility alias retained until the explicit 5.0 removal gate.
fplot_timeserie = fplot_timeseries

__all__ = ["fplot_timeseries", "fplot_timeserie"]
