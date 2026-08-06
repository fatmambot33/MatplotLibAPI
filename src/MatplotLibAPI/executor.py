"""Validated local plotting executor shared by Python, CLI, and agents."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from .intelligence import profile_dataframe, recommend_plots
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
        """Normalize the configured workspace path."""
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
                    details={
                        "available_columns": [str(column) for column in frame.columns]
                    },
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


def _presentation_context(spec: PlotSpec) -> Any:
    """Return a temporary Matplotlib context for accessibility presets."""
    preset = spec.presentation.accessibility
    if preset == "default":
        return nullcontext()
    try:
        import matplotlib as mpl
        from cycler import cycler
    except ImportError:  # pragma: no cover - matplotlib is a core dependency
        return nullcontext()
    palettes = {
        "high-contrast": ("#000000", "#0072B2", "#D55E00", "#009E73"),
        "colorblind": (
            "#0072B2",
            "#E69F00",
            "#009E73",
            "#D55E00",
            "#CC79A7",
            "#56B4E9",
        ),
    }
    palette = palettes.get(preset)
    if palette is None:
        return nullcontext()
    return mpl.rc_context(
        {
            "axes.prop_cycle": cycler(color=palette),
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "lines.linewidth": 2.0,
        }
    )


def _number_formatter(kind: str, currency: str) -> Any:
    """Return a deterministic scalar formatter callable."""
    if kind == "integer":
        return lambda value, _position: f"{value:,.0f}"
    if kind == "percent":
        return lambda value, _position: f"{value * 100:,.1f}%"
    if kind == "currency":
        return lambda value, _position: f"{currency} {value:,.2f}"
    if kind == "compact":

        def compact(value: float, _position: int) -> str:
            """Format a value with a compact magnitude suffix."""
            absolute = abs(value)
            if absolute >= 1_000_000_000:
                return f"{value / 1_000_000_000:.1f}B"
            if absolute >= 1_000_000:
                return f"{value / 1_000_000:.1f}M"
            if absolute >= 1_000:
                return f"{value / 1_000:.1f}K"
            return f"{value:g}"

        return compact
    return lambda value, _position: f"{value:,.2f}"


def _apply_presentation(figure: Any, spec: PlotSpec) -> Tuple[ValidationIssue, ...]:
    """Apply semantic formatting and text alternatives after rendering."""
    warnings = []
    presentation = spec.presentation
    if hasattr(figure, "axes"):
        try:
            from matplotlib.ticker import FuncFormatter

            formatter = None
            if presentation.number_format != "auto":
                formatter = FuncFormatter(
                    _number_formatter(
                        presentation.number_format,
                        presentation.currency,
                    )
                )
            for axis in figure.axes:
                axis.grid(presentation.show_grid, alpha=0.25)
                if formatter is not None:
                    axis.yaxis.set_major_formatter(formatter)
            if presentation.alt_text:
                figure.set_label(presentation.alt_text)
        except (AttributeError, TypeError, ValueError) as exc:
            warnings.append(
                ValidationIssue(
                    code="presentation.partial_application",
                    message=f"Presentation preferences were only partly applied: {exc}",
                    severity="warning",
                )
            )
    elif hasattr(figure, "update_layout"):
        metadata = {"alt_text": presentation.alt_text} if presentation.alt_text else {}
        figure.update_layout(meta=metadata)
        if presentation.number_format == "percent":
            figure.update_yaxes(tickformat=".1%")
        elif presentation.number_format == "currency":
            figure.update_yaxes(
                tickprefix=f"{presentation.currency} ", tickformat=",.2f"
            )
        elif presentation.number_format == "integer":
            figure.update_yaxes(tickformat=",.0f")
        elif presentation.number_format == "number":
            figure.update_yaxes(tickformat=",.2f")
        elif presentation.number_format == "compact":
            figure.update_yaxes(tickformat="~s")
    return tuple(warnings)


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
    with _presentation_context(resolved_spec):
        figure = descriptor.function(**parameters)
    presentation_warnings = _apply_presentation(figure, resolved_spec)
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
        warnings=presentation_warnings,
        metadata={
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "schema_version": resolved_spec.schema_version,
            "presentation": resolved_spec.presentation.to_dict(),
            "alt_text": resolved_spec.presentation.alt_text,
        },
    )


def inspect_dataframe(
    frame: pd.DataFrame,
    *,
    max_rows: int = 5_000,
    max_sample_values: int = 5,
) -> Dict[str, Any]:
    """Return a bounded deterministic local dataframe profile."""
    return profile_dataframe(
        frame,
        max_rows=max_rows,
        max_sample_values=max_sample_values,
    ).to_dict()


def recommend_plot(frame: pd.DataFrame) -> Dict[str, Any]:
    """Recommend charts deterministically with scores and explanations."""
    profile = profile_dataframe(frame)
    recommendations = recommend_plots(profile)
    top = recommendations[0].to_dict()
    top["profile"] = profile.to_dict()
    top["recommendations"] = [item.to_dict() for item in recommendations]
    return top


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
