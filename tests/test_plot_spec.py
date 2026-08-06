"""Contract tests for canonical schema-driven plot requests."""

import json

import pytest

from MatplotLibAPI import (
    PLOT_SPEC_SCHEMA_VERSION,
    PlotSpec,
    PlotValidationError,
)


def test_plot_spec_round_trip_is_deterministic() -> None:
    """A specification should survive a deterministic JSON round trip."""
    spec = PlotSpec.from_dict(
        {
            "chart": "bar",
            "encoding": {"category": "product", "value": "revenue"},
            "options": {"stacked": True},
            "output": {"format": "png", "dpi": 200},
        }
    )

    restored = PlotSpec.from_json(spec.to_json())

    assert restored.to_dict() == spec.to_dict()
    assert json.loads(spec.to_json())["schema_version"] == PLOT_SPEC_SCHEMA_VERSION


def test_legacy_params_migrate_to_options() -> None:
    """The compact legacy params shape should migrate without ambiguity."""
    spec = PlotSpec.from_dict(
        {"schema_version": "1", "chart": "bar", "params": {"stacked": True}}
    )

    assert spec.options == {"stacked": True}
    assert spec.schema_version == PLOT_SPEC_SCHEMA_VERSION


def test_duplicate_parameters_are_rejected() -> None:
    """Encoding and option dictionaries must not override each other."""
    with pytest.raises(PlotValidationError) as error:
        PlotSpec.from_dict(
            {
                "chart": "bar",
                "encoding": {"value": "revenue"},
                "options": {"value": "other"},
            }
        )

    assert error.value.issues[0].code == "spec.duplicate_parameter"


def test_plot_spec_schema_is_strict() -> None:
    """The exported schema should be versioned and reject unknown root keys."""
    schema = PlotSpec.json_schema()

    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == "1.0"
    assert schema["properties"]["output"]["properties"]["format"]["enum"]
