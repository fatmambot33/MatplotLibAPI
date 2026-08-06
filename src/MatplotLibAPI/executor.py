"""Validated local plotting executor shared by Python, CLI, and agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from .plugins import PluginError, PluginRegistry, PlotDescriptor, create_registry
from .specs import (
    DataSource,
    PlotSpec,
    PlotValidationError,
    RenderResult,
    ValidationIssue,
)


@dataclass(frozen=True)
class RenderPolicy:
    """Local filesystem and resource limits for deterministic agent execution."""

    workspace: Path = field(default_factory=Path.cwd)
    max_rows: int = 100_000
    max_columns: int = 200
    max_cells: int = 2_000_000
    max_input_bytes: int = 50 * 1024 * 1024
    max_output_bytes: int = 20 * 1024 * 1024
    allow_absolute_paths: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", self.workspace.expanduser().resolve())

    def resolve_input_path(self, value: str) -> Path:
        """Resolve and validate a readable path inside the configured workspace."""
        return self._resolve_path(value, purpose="input", must_exist=True)

    def resolve_output_path(self, value: str) -> Path:
        """Resolve and validate a writable path inside the configured workspace."""
        return self._resolve_path(value, purpose="output", must_exist=False)

    def _resolve_path(self, value: str, *, purpose: str, must_exist: bool) -> Path:
        candidate = Path(value).expanduser()
        if candidate.is_absolute() and not self.allow_absolute_paths:
            raise PlotValidationError(
                [
                    ValidationIssue(
                        code=f"policy.absolute_{purpose}_path",
                        message=f"Absolute {purpose} paths are disabled.",
                        path=("data" if purpose == "input" else "output", "path"),
                    )
                ]
            )
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.workspace / candidate).resolve()
        )
        try:
            resolved.relative_to(self.workspace)
        except ValueError as exc:
            raise PlotValidationError(
                [
                    ValidationIssue(
                        code=f"policy.{purpose}_outside_workspace",
                        message=f"The {purpose} path must stay inside the workspace.",
                        details={"workspace": str(self.workspace)},
                    )
                ]
            ) from exc
        if must_exist and not resolved.is_file():
            raise PlotValidationError(
                [
                    ValidationIssue(
                        code="data.file_not_found",
                        message=f"Data file not found: {resolved}.",
                        path=("data", "csv_path"),
                    )
                ]
            )
        return resolved


def _validate_frame(frame: pd.DataFrame, policy: RenderPolicy) -> None:
    issues = []
    rows, columns = frame.shape
    if rows > policy.max_rows:
        issues.append(
            ValidationIssue(
                code="policy.max_rows_exceeded",
                message=f"Input contains {rows} rows; maximum is {policy.max_rows}.",
                details={"actual": rows, "maximum": policy.max_rows},
            )
        )
    if columns > policy.max_columns:
        issues.append(
            ValidationIssue(
                code="policy.max_columns_exceeded",
                message=(
                    f"Input contains {columns} columns; maximum is "
                    f"{policy.max_columns}."
                ),
                details={"actual": columns, "maximum": policy.max_columns},
            )
        )
    cells = rows * columns
    if cells > policy.max_cells:
        issues.append(
            ValidationIssue(
                code="policy.max_cells_exceeded",
                message=f"Input contains {cells} cells; maximum is {policy.max_cells}.",
                details={"actual": cells, "maximum": policy.max_cells},
            )
        )
    if issues:
        raise PlotValidationError(issues)


def load_dataframe(
    source: DataSource,
    *,
    policy: Optional[RenderPolicy] = None,
) -> pd.DataFrame:
    """Load and validate a DataFrame from an embedded local source."""
    resolved_policy = policy or RenderPolicy()
    if source.table is not None:
        frame = pd.DataFrame([dict(row) for row in source.table])
    else:
        path = resolved_policy.resolve_input_path(str(source.csv_path))
        size = path.stat().st_size
        if size > resolved_policy.max_input_bytes:
            raise PlotValidationError(
                [
                    ValidationIssue(
                        code="policy.max_input_bytes_exceeded",
                        message=(
                            f"Input is {size} bytes; maximum is "
                            f"{resolved_policy.max_input_bytes}."
                        ),
                    )
                ]
            )
        if path.suffix.lower() != ".csv":
            raise PlotValidationError(
                [
                    ValidationIssue(
                        code="data.unsupported_file_type",
                        message="Only CSV input is supported by the portable executor.",
                        path=("data", "csv_path"),
                    )
                ]
            )
        frame = pd.read_csv(path)
    _validate_frame(frame, resolved_policy)
    return frame


def _resolve_frame(
    spec: PlotSpec,
    data: Optional[Union[pd.DataFrame, Sequence[Mapping[str, Any]]]],
    policy: RenderPolicy,
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy(deep=False)
        _validate_frame(frame, policy)
        return frame
    if data is not None:
        frame = pd.DataFrame([dict(row) for row in data])
        _validate_frame(frame, policy)
        return frame
    if spec.data is None:
        raise PlotValidationError(
            [
                ValidationIssue(
                    code="data.missing_source",
                    message="Provide a DataFrame, table rows, or spec.data.",
                    path=("data",),
                )
            ]
        )
    return load_dataframe(spec.data, policy=policy)


def validate_plot_request(
    spec: PlotSpec,
    frame: pd.DataFrame,
    descriptor: PlotDescriptor,
) -> Tuple[ValidationIssue, ...]:
    """Validate parameters and referenced DataFrame columns."""
    issues = []
    parameters = spec.parameters()
    schema = descriptor.parameter_schema
    properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
    required = schema.get("required", ()) if isinstance(schema, Mapping) else ()
    for name in required:
        if name not in parameters:
            issues.append(
                ValidationIssue(
                    code="parameters.missing_required",
                    message=f"Missing required parameter: {name}.",
                    path=("encoding", name),
                )
            )
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(parameters).difference(properties))
        for name in unknown:
            issues.append(
                ValidationIssue(
                    code="parameters.unknown",
                    message=f"Unknown parameter for {descriptor.name}: {name}.",
                    path=("options", name),
                )
            )
    for name in descriptor.column_parameters:
        value = parameters.get(name)
        if isinstance(value, str) and value not in frame.columns:
            issues.append(
                ValidationIssue(
                    code="data.missing_column",
                    message=f"Column {value!r} referenced by {name!r} does not exist.",
                    path=("encoding", name),
                    details={"available_columns": [str(column) for column in frame.columns]},
                )
            )
        elif isinstance(value, (list, tuple)):
            missing = [
                item
                for item in value
                if isinstance(item, str) and item not in frame.columns
            ]
            if missing:
                issues.append(
                    ValidationIssue(
                        code="data.missing_columns",
                        message=f"Referenced columns do not exist: {missing}.",
                        path=("encoding", name),
                    )
                )
    if issues:
        raise PlotValidationError(issues)
    return ()


def _serialize_figure(
    figure: Any,
    *,
    output_format: str,
    dpi: int,
    transparent: bool,
) -> bytes:
    if output_format == "json":
        if not hasattr(figure, "to_json"):
            raise PlotValidationError(
                [
                    ValidationIssue(
                        code="output.json_not_supported",
                        message="This plotting backend does not support JSON output.",
                    )
                ]
            )
        return str(figure.to_json()).encode("utf-8")
    if hasattr(figure, "savefig"):
        buffer = BytesIO()
        figure.savefig(
            buffer,
            format=output_format,
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
        )
        return buffer.getvalue()
    if hasattr(figure, "to_image"):
        return bytes(figure.to_image(format=output_format))
    raise PlotValidationError(
        [
            ValidationIssue(
                code="output.unsupported_figure",
                message="The plotting function returned an unsupported figure object.",
            )
        ]
    )


def execute_plot(
    spec: Union[PlotSpec, Mapping[str, Any]],
    data: Optional[Union[pd.DataFrame, Sequence[Mapping[str, Any]]]] = None,
    *,
    registry: Optional[PluginRegistry] = None,
    policy: Optional[RenderPolicy] = None,
) -> RenderResult:
    """Validate and execute one canonical plot request."""
    resolved_spec = spec if isinstance(spec, PlotSpec) else PlotSpec.from_dict(spec)
    resolved_registry = registry or create_registry()
    resolved_policy = policy or RenderPolicy()
    frame = _resolve_frame(resolved_spec, data, resolved_policy)
    try:
        descriptor = resolved_registry.context.get_descriptor(resolved_spec.chart)
    except PluginError as exc:
        raise PlotValidationError(
            [
                ValidationIssue(
                    code="plot.unknown_chart",
                    message=f"Unknown chart: {resolved_spec.chart!r}.",
                    path=("chart",),
                    details={"supported": resolved_registry.context.list_plots()},
                )
            ]
        ) from exc

    validate_plot_request(resolved_spec, frame, descriptor)
    parameters = resolved_spec.parameters()
    parameters[descriptor.data_parameter] = frame
    figure = descriptor.function(**parameters)
    output_format = resolved_spec.output.format
    payload: Optional[bytes] = None
    output_path: Optional[str] = None
    if output_format != "figure":
        if output_format not in descriptor.output_formats:
            raise PlotValidationError(
                [
                    ValidationIssue(
                        code="output.format_not_supported_by_plot",
                        message=(
                            f"{descriptor.name} does not support {output_format!r} output."
                        ),
                        details={"supported": list(descriptor.output_formats)},
                    )
                ]
            )
        payload = _serialize_figure(
            figure,
            output_format=output_format,
            dpi=resolved_spec.output.dpi,
            transparent=resolved_spec.output.transparent,
        )
        if len(payload) > resolved_policy.max_output_bytes:
            raise PlotValidationError(
                [
                    ValidationIssue(
                        code="policy.max_output_bytes_exceeded",
                        message=(
                            f"Output is {len(payload)} bytes; maximum is "
                            f"{resolved_policy.max_output_bytes}."
                        ),
                    )
                ]
            )
        if resolved_spec.output.path is not None:
            path = resolved_policy.resolve_output_path(resolved_spec.output.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            output_path = str(path)

    return RenderResult(
        chart=descriptor.name,
        backend=descriptor.backend,
        output_format=output_format,
        figure=figure,
        payload=payload,
        output_path=output_path,
        metadata={
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "schema_version": resolved_spec.schema_version,
        },
    )


def inspect_dataframe(frame: pd.DataFrame) -> Dict[str, Any]:
    """Return deterministic local profiling metadata for chart selection."""
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "missing": {
            str(column): int(value) for column, value in frame.isna().sum().items()
        },
        "numeric_columns": [
            str(column)
            for column in frame.select_dtypes(include="number").columns
        ],
        "datetime_columns": [
            str(column)
            for column in frame.columns
            if pd.api.types.is_datetime64_any_dtype(frame[column])
        ],
    }


def recommend_plot(frame: pd.DataFrame) -> Dict[str, Any]:
    """Recommend a chart deterministically without an LLM or credentials."""
    profile = inspect_dataframe(frame)
    numeric = profile["numeric_columns"]
    datetime_columns = profile["datetime_columns"]
    categorical = [
        column
        for column in profile["column_names"]
        if column not in numeric and column not in datetime_columns
    ]
    if datetime_columns and numeric:
        chart = "timeserie"
        encoding = {"x": datetime_columns[0], "y": numeric[0]}
        reason = "A datetime and numeric measure support a time-series view."
    elif len(numeric) >= 2:
        chart = "correlation_matrix"
        encoding = {}
        reason = "Multiple numeric columns support correlation analysis."
    elif numeric and categorical:
        chart = "bar"
        encoding = {"category": categorical[0], "value": numeric[0]}
        reason = "A categorical dimension and numeric measure support comparison."
    elif numeric:
        chart = "histogram_kde"
        encoding = {"column": numeric[0]}
        reason = "A single numeric measure is best inspected as a distribution."
    else:
        chart = "table"
        encoding = {}
        reason = "No numeric analytical measure was detected."
    return {"chart": chart, "encoding": encoding, "reason": reason, "profile": profile}


def openai_tool_definitions(
    *, registry: Optional[PluginRegistry] = None
) -> Sequence[Mapping[str, Any]]:
    """Generate OpenAI-compatible plotting tools from the registry."""
    resolved_registry = registry or create_registry()
    return resolved_registry.context.openai_tools()


__all__ = [
    "RenderPolicy",
    "execute_plot",
    "inspect_dataframe",
    "load_dataframe",
    "openai_tool_definitions",
    "recommend_plot",
    "validate_plot_request",
]
