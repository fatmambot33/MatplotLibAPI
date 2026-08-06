"""Deterministic agent-readiness evaluations that require no model or credentials."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from .conformance import validate_registry_conformance
from .executor import inspect_dataframe, recommend_plot
from .intelligence import suggest_plot_spec_repairs
from .migration import migrate_plot_spec_for_v5, v5_compatibility_status
from .plugins import PluginRegistry
from .specs import PlotSpec, PlotValidationError


@dataclass(frozen=True)
class EvaluationResult:
    """Outcome of one deterministic evaluation case."""

    name: str
    passed: bool
    expected: Any
    actual: Any

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable evaluation result."""
        return {
            "name": self.name,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
        }


def _recommendation_cases() -> Sequence[Mapping[str, Any]]:
    return (
        {
            "name": "category-and-value",
            "frame": pd.DataFrame({"category": ["A", "B"], "value": [1, 2]}),
            "expected": "bar",
        },
        {
            "name": "single-numeric",
            "frame": pd.DataFrame({"value": [1, 2, 3]}),
            "expected": "histogram_kde",
        },
        {
            "name": "multiple-numeric",
            "frame": pd.DataFrame({"a": [1, 2], "b": [2, 4]}),
            "expected": "correlation_matrix",
        },
        {
            "name": "datetime-and-value",
            "frame": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                    "value": [1, 2],
                }
            ),
            "expected": "timeseries",
        },
        {
            "name": "non-numeric",
            "frame": pd.DataFrame({"name": ["A", "B"]}),
            "expected": "table",
        },
    )


def _repair_registry() -> PluginRegistry:
    """Build a tiny deterministic registry for repair evaluation."""

    def plot(pd_df: pd.DataFrame, category: str, value: str) -> pd.DataFrame:
        """Return the selected columns for repair evaluation."""
        return pd.DataFrame(pd_df.loc[:, [category, value]])

    registry = PluginRegistry()
    registry.context.register_plot("bar", plot)
    return registry


def run_agent_evaluations() -> Dict[str, Any]:
    """Run deterministic intelligence, schema, and compatibility evaluations."""
    started = perf_counter()
    results: List[EvaluationResult] = []
    for case in _recommendation_cases():
        actual = recommend_plot(case["frame"])["chart"]
        results.append(
            EvaluationResult(
                name=str(case["name"]),
                passed=actual == case["expected"],
                expected=case["expected"],
                actual=actual,
            )
        )

    try:
        PlotSpec.from_dict({"chart": "", "encoding": {}})
    except PlotValidationError as exc:
        code = exc.issues[0].code
        results.append(
            EvaluationResult(
                name="invalid-spec-rejection",
                passed=code == "spec.missing_chart",
                expected="spec.missing_chart",
                actual=code,
            )
        )
    else:
        results.append(
            EvaluationResult(
                name="invalid-spec-rejection",
                passed=False,
                expected="PlotValidationError",
                actual="accepted",
            )
        )

    profile = inspect_dataframe(pd.DataFrame({"x": [1, None], "y": [2, 3]}))
    missing = profile["missing"]["x"]
    results.append(
        EvaluationResult(
            name="profile-missing-values",
            passed=missing == 1,
            expected=1,
            actual=missing,
        )
    )

    repair_spec = PlotSpec.from_dict(
        {
            "chart": "bar",
            "encoding": {"category": "categroy", "value": "value"},
        }
    )
    repairs = suggest_plot_spec_repairs(
        repair_spec,
        pd.DataFrame({"category": ["A"], "value": [1]}),
        registry=_repair_registry(),
    )
    repair_code = repairs[0].code if repairs else None
    results.append(
        EvaluationResult(
            name="repair-missing-column",
            passed=repair_code == "repair.missing_column",
            expected="repair.missing_column",
            actual=repair_code,
        )
    )

    migrated = migrate_plot_spec_for_v5(PlotSpec.from_dict({"chart": "timeserie"}))
    results.append(
        EvaluationResult(
            name="v5-canonical-timeseries",
            passed=migrated.chart == "timeseries",
            expected="timeseries",
            actual=migrated.chart,
        )
    )

    gate = v5_compatibility_status(as_of=date(2026, 8, 6))
    results.append(
        EvaluationResult(
            name="v5-removal-gate",
            passed=gate["breaking_removals_allowed"] is False,
            expected=False,
            actual=gate["breaking_removals_allowed"],
        )
    )

    conformance_registry = _repair_registry()
    conformance = validate_registry_conformance(conformance_registry)
    results.append(
        EvaluationResult(
            name="plugin-conformance",
            passed=conformance.passed,
            expected=True,
            actual=conformance.passed,
        )
    )

    passed = sum(result.passed for result in results)
    return {
        "passed": passed,
        "failed": len(results) - passed,
        "total": len(results),
        "duration_seconds": perf_counter() - started,
        "results": [result.to_dict() for result in results],
        "credentials_required": False,
        "llm_required": False,
    }


__all__ = ["EvaluationResult", "run_agent_evaluations"]
