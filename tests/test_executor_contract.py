"""Tests for the validated plotting executor and safety policy."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from MatplotLibAPI import PlotSpec, PlotValidationError
from MatplotLibAPI.executor import RenderPolicy, execute_plot
from MatplotLibAPI.plugins import PluginRegistry


def _bar(pd_df: pd.DataFrame, category: str, value: str, title: str = ""):
    figure, axis = plt.subplots()
    axis.bar(pd_df[category], pd_df[value])
    axis.set_title(title)
    return figure


def _registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.context.register_plot("bar", _bar, description="Test bar chart.")
    return registry


def test_executor_returns_png_and_metadata() -> None:
    """A validated request should render deterministic artifact metadata."""
    frame = pd.DataFrame({"product": ["A", "B"], "revenue": [1, 2]})
    spec = PlotSpec.from_dict(
        {
            "chart": "bar",
            "encoding": {"category": "product", "value": "revenue"},
            "output": {"format": "png"},
        }
    )

    result = execute_plot(spec, frame, registry=_registry())

    assert result.payload is not None
    assert result.payload.startswith(b"\x89PNG")
    assert result.metadata["rows"] == 2


def test_executor_rejects_missing_columns() -> None:
    """Column references should fail before the plotting function runs."""
    frame = pd.DataFrame({"product": ["A"], "revenue": [1]})
    spec = PlotSpec.from_dict(
        {
            "chart": "bar",
            "encoding": {"category": "missing", "value": "revenue"},
        }
    )

    with pytest.raises(PlotValidationError) as error:
        execute_plot(spec, frame, registry=_registry())

    assert error.value.issues[0].code == "data.missing_column"


def test_policy_keeps_outputs_inside_workspace(tmp_path: Path) -> None:
    """Agent output paths must remain inside the explicit workspace."""
    frame = pd.DataFrame({"product": ["A"], "revenue": [1]})
    spec = PlotSpec.from_dict(
        {
            "chart": "bar",
            "encoding": {"category": "product", "value": "revenue"},
            "output": {"format": "png", "path": "charts/result.png"},
        }
    )

    result = execute_plot(
        spec,
        frame,
        registry=_registry(),
        policy=RenderPolicy(workspace=tmp_path),
    )

    assert Path(str(result.output_path)).is_file()
    assert Path(str(result.output_path)).is_relative_to(tmp_path)


def test_policy_rejects_workspace_escape(tmp_path: Path) -> None:
    """Relative path traversal must not escape the agent workspace."""
    policy = RenderPolicy(workspace=tmp_path)

    with pytest.raises(PlotValidationError) as error:
        policy.resolve_output_path("../outside.png")

    assert error.value.issues[0].code == "policy.output_outside_workspace"
