"""Tests for accessible and semantic presentation presets."""

import matplotlib.pyplot as plt
import pandas as pd

from MatplotLibAPI.executor import execute_plot
from MatplotLibAPI.plugins import PluginRegistry
from MatplotLibAPI.specs import PlotSpec


def _registry() -> PluginRegistry:
    """Return a minimal Matplotlib registry for presentation tests."""

    def plot(pd_df: pd.DataFrame, x: str, y: str):
        figure, axis = plt.subplots()
        axis.plot(pd_df[x], pd_df[y])
        return figure

    registry = PluginRegistry()
    registry.context.register_plot("line", plot)
    return registry


def test_presentation_round_trip_and_execution_metadata() -> None:
    """Presentation preferences should serialize and reach the executor."""
    spec = PlotSpec.from_dict(
        {
            "chart": "line",
            "encoding": {"x": "x", "y": "y"},
            "presentation": {
                "accessibility": "colorblind",
                "number_format": "currency",
                "currency": "EUR",
                "alt_text": "Revenue over time.",
            },
        }
    )

    restored = PlotSpec.from_json(spec.to_json())
    result = execute_plot(
        restored,
        pd.DataFrame({"x": [1, 2], "y": [10, 20]}),
        registry=_registry(),
    )

    assert restored.presentation.accessibility == "colorblind"
    assert result.metadata["alt_text"] == "Revenue over time."
    assert result.metadata["presentation"]["currency"] == "EUR"
