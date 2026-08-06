"""Deterministic local data intelligence for plotting agents and people."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import pandas as pd

from .plugins import PluginError, PluginRegistry, create_registry
from .specs import PlotSpec


@dataclass(frozen=True)
class ColumnProfile:
    """Bounded deterministic summary for one dataframe column."""

    name: str
    dtype: str
    semantic_type: str
    missing_count: int
    missing_fraction: float
    unique_count: int
    sample_values: Tuple[Any, ...] = ()
    minimum: Optional[Any] = None
    maximum: Optional[Any] = None
    mean: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        result: Dict[str, Any] = {
            "name": self.name,
            "dtype": self.dtype,
            "semantic_type": self.semantic_type,
            "missing_count": self.missing_count,
            "missing_fraction": self.missing_fraction,
            "unique_count": self.unique_count,
            "sample_values": list(self.sample_values),
        }
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        if self.mean is not None:
            result["mean"] = self.mean
        return result


@dataclass(frozen=True)
class DataProfile:
    """Stable bounded dataframe profile used by recommendation logic."""

    rows: int
    columns: int
    sampled_rows: int
    truncated: bool
    column_profiles: Tuple[ColumnProfile, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        columns = [profile.to_dict() for profile in self.column_profiles]
        return {
            "rows": self.rows,
            "columns": self.columns,
            "sampled_rows": self.sampled_rows,
            "truncated": self.truncated,
            "column_profiles": columns,
            "column_names": [profile.name for profile in self.column_profiles],
            "dtypes": {profile.name: profile.dtype for profile in self.column_profiles},
            "missing": {
                profile.name: profile.missing_count for profile in self.column_profiles
            },
            "numeric_columns": [
                profile.name
                for profile in self.column_profiles
                if profile.semantic_type == "numeric"
            ],
            "datetime_columns": [
                profile.name
                for profile in self.column_profiles
                if profile.semantic_type == "datetime"
            ],
            "categorical_columns": [
                profile.name
                for profile in self.column_profiles
                if profile.semantic_type in {"categorical", "boolean"}
            ],
            "text_columns": [
                profile.name
                for profile in self.column_profiles
                if profile.semantic_type == "text"
            ],
        }


@dataclass(frozen=True)
class PlotRecommendation:
    """Explainable deterministic chart recommendation."""

    chart: str
    encoding: Mapping[str, Any]
    score: float
    reasons: Tuple[str, ...]
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable recommendation."""
        return {
            "chart": self.chart,
            "encoding": dict(self.encoding),
            "score": self.score,
            "reasons": list(self.reasons),
            "reason": " ".join(self.reasons),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class RepairSuggestion:
    """One opt-in, explainable plot specification repair."""

    code: str
    message: str
    path: Tuple[str, ...]
    current: Any = None
    proposed: Any = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable repair suggestion."""
        return {
            "code": self.code,
            "message": self.message,
            "path": list(self.path),
            "current": self.current,
            "proposed": self.proposed,
            "confidence": self.confidence,
        }


def _json_scalar(value: Any) -> Any:
    """Normalize pandas and NumPy scalar values for JSON output."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _semantic_type(series: pd.Series, unique_count: int) -> str:
    """Infer a conservative semantic role from one sampled series."""
    if series.dropna().empty:
        return "unknown"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    non_null = max(1, int(series.notna().sum()))
    if unique_count <= 50 or unique_count / non_null <= 0.2:
        return "categorical"
    return "text"


def profile_dataframe(
    frame: pd.DataFrame,
    *,
    max_rows: int = 5_000,
    max_sample_values: int = 5,
) -> DataProfile:
    """Profile a dataframe deterministically using a bounded head sample.

    Parameters
    ----------
    frame:
        Dataframe to profile.
    max_rows:
        Maximum number of rows inspected. Sampling is deterministic and uses
        the first rows so repeated calls produce the same result.
    max_sample_values:
        Maximum representative non-null values retained per column.

    Returns
    -------
    DataProfile
        Stable local profile with semantic roles and bounded statistics.
    """
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if max_sample_values < 0:
        raise ValueError("max_sample_values cannot be negative")
    sample = frame.head(max_rows)
    profiles: List[ColumnProfile] = []
    for index, column in enumerate(frame.columns):
        name = str(column)
        series = cast(pd.Series, sample.iloc[:, index])
        missing_count = int(cast(Any, series.isna().sum()))
        unique_count = int(cast(Any, series.nunique(dropna=True)))
        semantic_type = _semantic_type(series, unique_count)
        unique_values: List[Any] = []
        if max_sample_values:
            for value in series.dropna().drop_duplicates().head(max_sample_values):
                unique_values.append(_json_scalar(value))
        minimum: Optional[Any] = None
        maximum: Optional[Any] = None
        mean: Optional[float] = None
        non_null = series.dropna()
        if semantic_type in {"numeric", "datetime"} and not non_null.empty:
            minimum = _json_scalar(non_null.min())
            maximum = _json_scalar(non_null.max())
        if semantic_type == "numeric" and not non_null.empty:
            mean = float(cast(Any, non_null.mean()))
        sampled_rows = int(sample.shape[0])
        profiles.append(
            ColumnProfile(
                name=name,
                dtype=str(series.dtype),
                semantic_type=semantic_type,
                missing_count=missing_count,
                missing_fraction=(
                    missing_count / sampled_rows if sampled_rows else 0.0
                ),
                unique_count=unique_count,
                sample_values=tuple(unique_values),
                minimum=minimum,
                maximum=maximum,
                mean=mean,
            )
        )
    return DataProfile(
        rows=int(frame.shape[0]),
        columns=int(frame.shape[1]),
        sampled_rows=int(sample.shape[0]),
        truncated=int(frame.shape[0]) > int(sample.shape[0]),
        column_profiles=tuple(profiles),
    )


def recommend_plots(
    value: Union[pd.DataFrame, DataProfile],
    *,
    limit: int = 3,
) -> Tuple[PlotRecommendation, ...]:
    """Return ranked deterministic recommendations with explicit reasons."""
    profile = value if isinstance(value, DataProfile) else profile_dataframe(value)
    serialized = profile.to_dict()
    numeric = list(serialized["numeric_columns"])
    datetimes = list(serialized["datetime_columns"])
    categoricals = list(serialized["categorical_columns"])
    text = list(serialized["text_columns"])
    recommendations: List[PlotRecommendation] = []

    if datetimes and numeric:
        recommendations.append(
            PlotRecommendation(
                chart="timeseries",
                encoding={"x": datetimes[0], "y": numeric[0]},
                score=0.98,
                reasons=(
                    "A datetime field provides an ordered temporal axis.",
                    "A numeric field provides a measurable trend.",
                ),
            )
        )
    if len(numeric) >= 2:
        recommendations.append(
            PlotRecommendation(
                chart="correlation_matrix",
                encoding={},
                score=0.91,
                reasons=(
                    "Multiple numeric fields support pairwise relationship analysis.",
                ),
            )
        )
    if categoricals and numeric:
        recommendations.append(
            PlotRecommendation(
                chart="bar",
                encoding={"category": categoricals[0], "value": numeric[0]},
                score=0.89,
                reasons=(
                    "A categorical dimension supports grouped comparison.",
                    "A numeric measure supports magnitude comparison.",
                ),
            )
        )
        category_profile = next(
            item for item in profile.column_profiles if item.name == categoricals[0]
        )
        if 1 < category_profile.unique_count <= 8:
            recommendations.append(
                PlotRecommendation(
                    chart="pie_donut",
                    encoding={"category": categoricals[0], "value": numeric[0]},
                    score=0.62,
                    reasons=(
                        "The category count is small enough for a part-to-whole view.",
                    ),
                    warnings=("Prefer a bar chart when precise comparison matters.",),
                )
            )
    if numeric:
        recommendations.append(
            PlotRecommendation(
                chart="histogram_kde",
                encoding={"column": numeric[0]},
                score=0.82,
                reasons=("A numeric measure supports distribution analysis.",),
            )
        )
    if text and not numeric:
        recommendations.append(
            PlotRecommendation(
                chart="wordcloud",
                encoding={"text_column": text[0]},
                score=0.55,
                reasons=("A free-text field can be summarized by term prominence.",),
                warnings=("Use a table when exact values are more important.",),
            )
        )
    recommendations.append(
        PlotRecommendation(
            chart="table",
            encoding={},
            score=0.30,
            reasons=(
                "A table is the deterministic fallback for mixed or sparse data.",
            ),
        )
    )
    recommendations.sort(key=lambda item: (-item.score, item.chart))
    return tuple(recommendations[: max(1, limit)])


def _candidate_column(
    available: Sequence[str],
    requested: str,
) -> Optional[str]:
    """Return one deterministic close column match when confidence is useful."""
    matches = get_close_matches(requested, list(available), n=1, cutoff=0.72)
    return matches[0] if matches else None


def suggest_plot_spec_repairs(
    spec: PlotSpec,
    frame: pd.DataFrame,
    *,
    registry: Optional[PluginRegistry] = None,
) -> Tuple[RepairSuggestion, ...]:
    """Suggest safe opt-in repairs without mutating the specification."""
    resolved_registry = registry or create_registry()
    suggestions: List[RepairSuggestion] = []
    canonical = resolved_registry.context.resolve_name(spec.chart)
    if canonical != spec.chart:
        suggestions.append(
            RepairSuggestion(
                code="repair.canonical_chart",
                message=f"Use canonical chart name {canonical!r}.",
                path=("chart",),
                current=spec.chart,
                proposed=canonical,
                confidence=1.0,
            )
        )
    try:
        descriptor = resolved_registry.context.get_descriptor(spec.chart)
    except PluginError:
        matches = get_close_matches(
            spec.chart,
            resolved_registry.context.list_plots(),
            n=1,
            cutoff=0.65,
        )
        if matches:
            suggestions.append(
                RepairSuggestion(
                    code="repair.unknown_chart",
                    message=f"Replace unknown chart with {matches[0]!r}.",
                    path=("chart",),
                    current=spec.chart,
                    proposed=matches[0],
                    confidence=0.75,
                )
            )
        return tuple(suggestions)

    parameters = spec.parameters()
    available = [str(column) for column in frame.columns]
    for name in descriptor.column_parameters:
        value = parameters.get(name)
        if isinstance(value, str) and value not in available:
            proposed = _candidate_column(available, value)
            if proposed is not None:
                path_root = "encoding" if name in spec.encoding else "options"
                suggestions.append(
                    RepairSuggestion(
                        code="repair.missing_column",
                        message=f"Replace missing column {value!r} with {proposed!r}.",
                        path=(path_root, name),
                        current=value,
                        proposed=proposed,
                        confidence=0.80,
                    )
                )
    schema = descriptor.parameter_schema
    properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
    if schema.get("additionalProperties") is False:
        for name in sorted(set(parameters).difference(properties)):
            if name == "title":
                continue
            path_root = "encoding" if name in spec.encoding else "options"
            suggestions.append(
                RepairSuggestion(
                    code="repair.remove_unknown_parameter",
                    message=f"Remove unsupported parameter {name!r}.",
                    path=(path_root, name),
                    current=parameters[name],
                    proposed=None,
                    confidence=1.0,
                )
            )
    return tuple(suggestions)


def apply_repair_suggestions(
    spec: PlotSpec,
    suggestions: Iterable[RepairSuggestion],
    *,
    minimum_confidence: float = 0.0,
) -> PlotSpec:
    """Apply explicitly supplied repairs and return a new PlotSpec."""
    value = spec.to_dict()
    for suggestion in suggestions:
        if suggestion.confidence < minimum_confidence:
            continue
        if suggestion.path == ("chart",):
            value["chart"] = suggestion.proposed
            continue
        if len(suggestion.path) != 2:
            continue
        section, name = suggestion.path
        mapping = dict(value.get(section, {}))
        if suggestion.proposed is None:
            mapping.pop(name, None)
        else:
            mapping[name] = suggestion.proposed
        value[section] = mapping
    return PlotSpec.from_dict(value)


__all__ = [
    "ColumnProfile",
    "DataProfile",
    "PlotRecommendation",
    "RepairSuggestion",
    "apply_repair_suggestions",
    "profile_dataframe",
    "recommend_plots",
    "suggest_plot_spec_repairs",
]
