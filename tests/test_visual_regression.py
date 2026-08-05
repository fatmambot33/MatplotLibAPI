"""Deterministic pixel-level regression checks for representative charts."""

from collections.abc import Callable

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

matplotlib.use("Agg")

from MatplotLibAPI import (  # noqa: E402
    fplot_area,
    fplot_bar,
    fplot_heatmap,
    fplot_pie_donut,
    fplot_table,
    fplot_timeserie,
)


def _pixels(figure: Figure) -> np.ndarray:
    """Return a copied RGBA buffer after a deterministic canvas draw."""
    figure.set_dpi(100)
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba()).copy()


def _assert_deterministic(factory: Callable[[], Figure]) -> None:
    """Assert two independent renders are exactly pixel-identical."""
    first = factory()
    second = factory()
    first_pixels = _pixels(first)
    second_pixels = _pixels(second)
    assert first_pixels.shape == second_pixels.shape
    assert first_pixels.size > 0
    np.testing.assert_array_equal(first_pixels, second_pixels)


def test_representative_matplotlib_charts_are_deterministic() -> None:
    """Detect unintended visual nondeterminism across core chart families."""
    frame = pd.DataFrame(
        {
            "category": ["A", "A", "B", "B"],
            "group": ["North", "South", "North", "South"],
            "value": [12, 9, 15, 11],
        }
    )
    dates = ["2026-01-01", "2026-02-01", "2026-01-01", "2026-02-01"]
    timeline = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "group": ["A", "A", "B", "B"],
            "value": [10, 14, 8, 13],
        }
    )
    shares = frame.groupby("category", as_index=False)["value"].sum()

    factories = [
        lambda: fplot_bar(
            frame,
            category="category",
            value="value",
            group="group",
            stacked=True,
        ),
        lambda: fplot_area(
            frame,
            x="category",
            y="value",
            label="group",
            stacked=True,
        ),
        lambda: fplot_heatmap(
            frame,
            index="category",
            columns="group",
            values="value",
        ),
        lambda: fplot_pie_donut(
            shares,
            category="category",
            value="value",
            donut=True,
        ),
        lambda: fplot_table(
            pd_df=frame,
            cols=["category", "group", "value"],
        ),
        lambda: fplot_timeserie(
            pd_df=timeline,
            label="group",
            x="date",
            y="value",
        ),
    ]

    for factory in factories:
        _assert_deterministic(factory)
