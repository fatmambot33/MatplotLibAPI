"""Serializable plotting contracts shared by every MatplotLibAPI interface."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

PLOT_SPEC_SCHEMA_VERSION = "1.0"
_ALLOWED_OUTPUT_FORMATS = ("figure", "png", "svg", "json")


@dataclass(frozen=True)
class ValidationIssue:
    """Machine-readable validation error or warning."""

    code: str
    message: str
    path: Tuple[str, ...] = ()
    severity: str = "error"
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        result: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "path": list(self.path),
            "severity": self.severity,
        }
        if self.details:
            result["details"] = dict(self.details)
        return result


class PlotValidationError(ValueError):
    """Raised when a plot specification or execution input is invalid."""

    def __init__(self, issues: Iterable[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        message = "; ".join(issue.message for issue in self.issues)
        super().__init__(message or "Plot validation failed")

    def to_dict(self) -> Dict[str, Any]:
        """Return a machine-readable error payload."""
        return {
            "error": "plot_validation_failed",
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class DataSource:
    """Optional local data source embedded in a plot specification."""

    csv_path: Optional[str] = None
    table: Optional[Tuple[Mapping[str, Any], ...]] = None

    def __post_init__(self) -> None:
        if self.csv_path is not None and self.table is not None:
            raise PlotValidationError(
                [
                    ValidationIssue(
                        code="data.multiple_sources",
                        message="Provide either csv_path or table, not both.",
                        path=("data",),
                    )
                ]
            )
        if self.csv_path is None and self.table is None:
            raise PlotValidationError(
                [
                    ValidationIssue(
                        code="data.missing_source",
                        message="Provide csv_path or table.",
                        path=("data",),
                    )
                ]
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DataSource":
        """Build a data source from a mapping."""
        table = value.get("table")
        rows: Optional[Tuple[Mapping[str, Any], ...]] = None
        if table is not None:
            if not isinstance(table, Sequence) or isinstance(table, (str, bytes)):
                raise PlotValidationError(
                    [
                        ValidationIssue(
                            code="data.invalid_table",
                            message="data.table must be a list of row objects.",
                            path=("data", "table"),
                        )
                    ]
                )
            rows = tuple(dict(row) for row in table)
        csv_path = value.get("csv_path")
        return cls(csv_path=str(csv_path) if csv_path is not None else None, table=rows)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        if self.csv_path is not None:
            return {"csv_path": self.csv_path}
        return {"table": [dict(row) for row in self.table or ()]}


@dataclass(frozen=True)
class OutputSpec:
    """Requested rendering output."""

    format: str = "figure"
    path: Optional[str] = None
    dpi: int = 150
    transparent: bool = False

    def __post_init__(self) -> None:
        issues: List[ValidationIssue] = []
        if self.format not in _ALLOWED_OUTPUT_FORMATS:
            issues.append(
                ValidationIssue(
                    code="output.unsupported_format",
                    message=f"Unsupported output format: {self.format!r}.",
                    path=("output", "format"),
                    details={"supported": list(_ALLOWED_OUTPUT_FORMATS)},
                )
            )
        if self.dpi < 36 or self.dpi > 1200:
            issues.append(
                ValidationIssue(
                    code="output.invalid_dpi",
                    message="output.dpi must be between 36 and 1200.",
                    path=("output", "dpi"),
                )
            )
        if issues:
            raise PlotValidationError(issues)

    @classmethod
    def from_dict(cls, value: Optional[Mapping[str, Any]]) -> "OutputSpec":
        """Build an output specification from a mapping."""
        if value is None:
            return cls()
        return cls(
            format=str(value.get("format", "figure")),
            path=str(value["path"]) if value.get("path") is not None else None,
            dpi=int(value.get("dpi", 150)),
            transparent=bool(value.get("transparent", False)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        result: Dict[str, Any] = {
            "format": self.format,
            "dpi": self.dpi,
            "transparent": self.transparent,
        }
        if self.path is not None:
            result["path"] = self.path
        return result


@dataclass(frozen=True)
class PlotSpec:
    """Canonical, serializable request for one chart."""

    chart: str
    encoding: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)
    title: Optional[str] = None
    data: Optional[DataSource] = None
    output: OutputSpec = field(default_factory=OutputSpec)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = PLOT_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        issues: List[ValidationIssue] = []
        if not isinstance(self.chart, str) or not self.chart.strip():
            issues.append(
                ValidationIssue(
                    code="spec.missing_chart",
                    message="chart must be a non-empty string.",
                    path=("chart",),
                )
            )
        if self.schema_version != PLOT_SPEC_SCHEMA_VERSION:
            issues.append(
                ValidationIssue(
                    code="spec.unsupported_version",
                    message=f"Unsupported schema version: {self.schema_version!r}.",
                    path=("schema_version",),
                    details={"supported": PLOT_SPEC_SCHEMA_VERSION},
                )
            )
        for name, value in (("encoding", self.encoding), ("options", self.options)):
            if not isinstance(value, Mapping):
                issues.append(
                    ValidationIssue(
                        code=f"spec.invalid_{name}",
                        message=f"{name} must be an object.",
                        path=(name,),
                    )
                )
        overlap = sorted(set(self.encoding).intersection(self.options))
        if overlap:
            issues.append(
                ValidationIssue(
                    code="spec.duplicate_parameter",
                    message="A parameter cannot appear in both encoding and options.",
                    path=("encoding",),
                    details={"parameters": overlap},
                )
            )
        if issues:
            raise PlotValidationError(issues)
        object.__setattr__(self, "chart", self.chart.strip())
        object.__setattr__(self, "encoding", dict(self.encoding))
        object.__setattr__(self, "options", dict(self.options))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlotSpec":
        """Build a plot specification, migrating supported legacy shapes."""
        migrated = migrate_plot_spec(value)
        data_value = migrated.get("data")
        output_value = migrated.get("output")
        return cls(
            chart=str(migrated.get("chart", "")),
            encoding=dict(migrated.get("encoding", {})),
            options=dict(migrated.get("options", {})),
            title=(
                str(migrated["title"])
                if migrated.get("title") is not None
                else None
            ),
            data=(
                DataSource.from_dict(data_value)
                if isinstance(data_value, Mapping)
                else None
            ),
            output=OutputSpec.from_dict(
                output_value if isinstance(output_value, Mapping) else None
            ),
            metadata=dict(migrated.get("metadata", {})),
            schema_version=str(
                migrated.get("schema_version", PLOT_SPEC_SCHEMA_VERSION)
            ),
        )

    @classmethod
    def from_json(cls, value: str) -> "PlotSpec":
        """Build a plot specification from JSON text."""
        parsed = json.loads(value)
        if not isinstance(parsed, Mapping):
            raise PlotValidationError(
                [
                    ValidationIssue(
                        code="spec.invalid_document",
                        message="A plot specification must be a JSON object.",
                    )
                ]
            )
        return cls.from_dict(parsed)

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "PlotSpec":
        """Read a plot specification from a JSON file."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def parameters(self) -> Dict[str, Any]:
        """Return the merged keyword arguments passed to the plot function."""
        result = dict(self.encoding)
        result.update(self.options)
        if self.title is not None and "title" not in result:
            result["title"] = self.title
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Return a deterministic JSON-serializable representation."""
        result: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "chart": self.chart,
            "encoding": dict(self.encoding),
            "options": dict(self.options),
            "output": self.output.to_dict(),
        }
        if self.title is not None:
            result["title"] = self.title
        if self.data is not None:
            result["data"] = self.data.to_dict()
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """Serialize the specification deterministically."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def json_schema(cls) -> Dict[str, Any]:
        """Return the canonical JSON Schema for plot requests."""
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://matplotlibapi.dev/schemas/plot-spec-1.0.json",
            "title": "MatplotLibAPI PlotSpec",
            "type": "object",
            "additionalProperties": False,
            "required": ["chart"],
            "properties": {
                "schema_version": {
                    "type": "string",
                    "const": PLOT_SPEC_SCHEMA_VERSION,
                    "default": PLOT_SPEC_SCHEMA_VERSION,
                },
                "chart": {"type": "string", "minLength": 1},
                "encoding": {"type": "object", "default": {}},
                "options": {"type": "object", "default": {}},
                "title": {"type": ["string", "null"]},
                "metadata": {"type": "object", "default": {}},
                "data": {
                    "type": "object",
                    "oneOf": [
                        {
                            "required": ["csv_path"],
                            "properties": {"csv_path": {"type": "string"}},
                            "additionalProperties": False,
                        },
                        {
                            "required": ["table"],
                            "properties": {
                                "table": {
                                    "type": "array",
                                    "items": {"type": "object"},
                                }
                            },
                            "additionalProperties": False,
                        },
                    ],
                },
                "output": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": list(_ALLOWED_OUTPUT_FORMATS),
                            "default": "figure",
                        },
                        "path": {"type": ["string", "null"]},
                        "dpi": {
                            "type": "integer",
                            "minimum": 36,
                            "maximum": 1200,
                            "default": 150,
                        },
                        "transparent": {"type": "boolean", "default": False},
                    },
                },
            },
        }


def migrate_plot_spec(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Migrate supported legacy specification shapes to schema version 1.0."""
    migrated = dict(value)
    version = str(migrated.get("schema_version", "1.0"))
    if version in {"1", "1.0"}:
        migrated["schema_version"] = PLOT_SPEC_SCHEMA_VERSION
    else:
        raise PlotValidationError(
            [
                ValidationIssue(
                    code="spec.unsupported_version",
                    message=f"Unsupported schema version: {version!r}.",
                    path=("schema_version",),
                )
            ]
        )

    parameters = migrated.pop("params", None)
    if parameters is not None:
        if "encoding" in migrated or "options" in migrated:
            raise PlotValidationError(
                [
                    ValidationIssue(
                        code="spec.ambiguous_legacy_params",
                        message="Do not combine legacy params with encoding or options.",
                        path=("params",),
                    )
                ]
            )
        if not isinstance(parameters, Mapping):
            raise PlotValidationError(
                [
                    ValidationIssue(
                        code="spec.invalid_params",
                        message="params must be an object.",
                        path=("params",),
                    )
                ]
            )
        migrated["options"] = dict(parameters)
    return migrated


@dataclass
class RenderResult:
    """Result returned by the canonical plotting executor."""

    chart: str
    backend: str
    output_format: str
    figure: Any = field(default=None, repr=False)
    payload: Optional[bytes] = field(default=None, repr=False)
    output_path: Optional[str] = None
    warnings: Tuple[ValidationIssue, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return serializable execution metadata without the figure object."""
        result: Dict[str, Any] = {
            "chart": self.chart,
            "backend": self.backend,
            "output_format": self.output_format,
            "warnings": [warning.to_dict() for warning in self.warnings],
            "metadata": dict(self.metadata),
        }
        if self.output_path is not None:
            result["output_path"] = self.output_path
        if self.payload is not None:
            result["output_bytes"] = len(self.payload)
        return result


__all__ = [
    "PLOT_SPEC_SCHEMA_VERSION",
    "DataSource",
    "OutputSpec",
    "PlotSpec",
    "PlotValidationError",
    "RenderResult",
    "ValidationIssue",
    "migrate_plot_spec",
]
