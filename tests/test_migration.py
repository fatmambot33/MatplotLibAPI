"""Tests for the explicit MatplotLibAPI 5.0 migration gate."""

from datetime import date

from MatplotLibAPI.migration import (
    audit_plot_spec_for_v5,
    migrate_plot_spec_for_v5,
    v5_compatibility_status,
)
from MatplotLibAPI.specs import PlotSpec


def test_timeserie_migrates_to_canonical_timeseries() -> None:
    """The legacy spelling should migrate without mutating the source spec."""
    spec = PlotSpec.from_dict({"chart": "timeserie"})

    notices = audit_plot_spec_for_v5(spec)
    migrated = migrate_plot_spec_for_v5(spec)

    assert notices[0].replacement == "timeseries"
    assert spec.chart == "timeserie"
    assert migrated.chart == "timeseries"


def test_breaking_removal_gate_is_date_bound() -> None:
    """Legacy removals must remain blocked during the documented window."""
    before = v5_compatibility_status(as_of=date(2026, 8, 6))
    after = v5_compatibility_status(as_of=date(2027, 2, 6))

    assert before["breaking_removals_allowed"] is False
    assert after["breaking_removals_allowed"] is True
