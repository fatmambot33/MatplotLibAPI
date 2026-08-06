"""Explicit compatibility diagnostics and preparation for MatplotLibAPI 5.0."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Mapping, Optional, Tuple

from .specs import PlotSpec

V5_REMOVAL_NOT_BEFORE = date(2027, 2, 6)
V5_CANONICAL_CHARTS: Mapping[str, str] = {
    "timeserie": "timeseries",
    "histogram": "histogram_kde",
    "pie": "pie_donut",
}


@dataclass(frozen=True)
class MigrationNotice:
    """One explicit compatibility or migration notice."""

    code: str
    message: str
    path: Tuple[str, ...]
    replacement: Optional[Any] = None
    breaking_in: str = "5.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable migration notice."""
        result: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "path": list(self.path),
            "breaking_in": self.breaking_in,
        }
        if self.replacement is not None:
            result["replacement"] = self.replacement
        return result


def audit_plot_spec_for_v5(spec: PlotSpec) -> Tuple[MigrationNotice, ...]:
    """Return deterministic notices for a specification affected by 5.0."""
    replacement = V5_CANONICAL_CHARTS.get(spec.chart)
    if replacement is None:
        return ()
    return (
        MigrationNotice(
            code="migration.legacy_chart_alias",
            message=(
                f"Chart {spec.chart!r} is a compatibility alias; use "
                f"{replacement!r} before 5.0."
            ),
            path=("chart",),
            replacement=replacement,
        ),
    )


def migrate_plot_spec_for_v5(spec: PlotSpec) -> PlotSpec:
    """Return a new specification using 5.0 canonical chart names."""
    replacement = V5_CANONICAL_CHARTS.get(spec.chart)
    if replacement is None:
        return spec
    value = spec.to_dict()
    value["chart"] = replacement
    metadata = dict(value.get("metadata", {}))
    metadata["migrated_from_chart"] = spec.chart
    value["metadata"] = metadata
    return PlotSpec.from_dict(value)


def v5_compatibility_status(*, as_of: Optional[date] = None) -> Dict[str, Any]:
    """Return the explicit gate preventing premature breaking removals."""
    resolved_date = as_of or date.today()
    ready = resolved_date >= V5_REMOVAL_NOT_BEFORE
    return {
        "target": "5.0.0",
        "as_of": resolved_date.isoformat(),
        "removal_not_before": V5_REMOVAL_NOT_BEFORE.isoformat(),
        "breaking_removals_allowed": ready,
        "legacy_aliases": dict(V5_CANONICAL_CHARTS),
        "next_action": (
            "Remove compatibility aliases and plugin API v1 support."
            if ready
            else "Keep aliases enabled; use migration diagnostics and canonical names."
        ),
    }


__all__ = [
    "V5_CANONICAL_CHARTS",
    "V5_REMOVAL_NOT_BEFORE",
    "MigrationNotice",
    "audit_plot_spec_for_v5",
    "migrate_plot_spec_for_v5",
    "v5_compatibility_status",
]
