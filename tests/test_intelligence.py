"""Tests for bounded local profiling and explainable recommendations."""

import pandas as pd

from MatplotLibAPI.intelligence import (
    apply_repair_suggestions,
    profile_dataframe,
    recommend_plots,
    suggest_plot_spec_repairs,
)
from MatplotLibAPI.plugins import PluginRegistry
from MatplotLibAPI.specs import PlotSpec


def _registry() -> PluginRegistry:
    """Return a minimal registry for deterministic repair tests."""

    def plot(pd_df: pd.DataFrame, category: str, value: str) -> pd.DataFrame:
        return pd_df[[category, value]]

    registry = PluginRegistry()
    registry.context.register_plot("bar", plot, aliases=("bars",))
    return registry


def test_profile_is_bounded_and_semantic() -> None:
    """Profiles should use a deterministic bounded sample and infer roles."""
    frame = pd.DataFrame(
        {
            "category": ["A", "B", "A", "B"],
            "value": [1, 2, 3, 4],
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
            ),
        }
    )

    profile = profile_dataframe(frame, max_rows=2)
    value = profile.to_dict()

    assert value["sampled_rows"] == 2
    assert value["truncated"] is True
    assert value["numeric_columns"] == ["value"]
    assert value["datetime_columns"] == ["date"]


def test_recommendations_are_ranked_and_explainable() -> None:
    """Time-series data should produce a canonical explained recommendation."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "value": [1, 2],
        }
    )

    recommendations = recommend_plots(frame)

    assert recommendations[0].chart == "timeseries"
    assert recommendations[0].score > recommendations[-1].score
    assert recommendations[0].reasons


def test_repairs_are_opt_in_and_apply_explicitly() -> None:
    """Repair suggestions should not mutate a spec until explicitly applied."""
    frame = pd.DataFrame({"category": ["A"], "value": [1]})
    spec = PlotSpec.from_dict(
        {
            "chart": "bars",
            "encoding": {"category": "categroy", "value": "value"},
        }
    )

    suggestions = suggest_plot_spec_repairs(spec, frame, registry=_registry())
    repaired = apply_repair_suggestions(spec, suggestions)

    assert spec.chart == "bars"
    assert repaired.chart == "bar"
    assert repaired.encoding["category"] == "category"
