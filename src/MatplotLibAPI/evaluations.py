"""Deterministic agent-readiness evaluations that require no model or credentials."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from .executor import inspect_dataframe, recommend_plot
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
            "name": "non-numeric",
            "frame": pd.DataFrame({"name": ["A", "B"]}),
            "expected": "table",
        },
    )


def run_agent_evaluations() -> Dict[str, Any]:
    """Run deterministic schema and chart-selection evaluations."""
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
