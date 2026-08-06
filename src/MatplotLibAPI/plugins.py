"""Typed plugin discovery and schema-rich plot registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    cast,
    runtime_checkable,
)

PLUGIN_API_VERSION = "2"
LEGACY_PLUGIN_API_VERSION = "1"
ENTRY_POINT_GROUP = "matplotlibapi.plugins"
PlotCallable = Callable[..., Any]


class PluginError(RuntimeError):
    """Base error raised by the plugin surface."""


class DuplicatePluginError(PluginError):
    """Raised when a plugin, plot, or alias is registered twice."""


def _json_type(annotation: Any, default: Any) -> Dict[str, Any]:
    """Return a conservative JSON Schema fragment for a Python parameter."""
    if default is not inspect.Parameter.empty and default is None:
        return {}
    if annotation in {str} or isinstance(default, str):
        return {"type": "string"}
    if annotation in {bool} or isinstance(default, bool):
        return {"type": "boolean"}
    if annotation in {int} or (
        isinstance(default, int) and not isinstance(default, bool)
    ):
        return {"type": "integer"}
    if annotation in {float} or isinstance(default, float):
        return {"type": "number"}
    if annotation in {list, List, Sequence, tuple, Tuple} or isinstance(
        default, (list, tuple)
    ):
        return {"type": "array"}
    if annotation in {dict, Dict, Mapping} or isinstance(default, Mapping):
        return {"type": "object"}
    return {}


@dataclass(frozen=True)
class PlotDescriptor:
    """Discoverable contract for one plotting callable."""

    name: str
    function: PlotCallable = field(repr=False, compare=False)
    description: str = ""
    backend: str = "matplotlib"
    data_parameter: str = "pd_df"
    parameter_schema: Mapping[str, Any] = field(default_factory=dict)
    capabilities: Tuple[str, ...] = ("render",)
    output_formats: Tuple[str, ...] = ("figure", "png", "svg")
    examples: Tuple[Mapping[str, Any], ...] = ()
    column_parameters: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Return serializable discovery metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "backend": self.backend,
            "data_parameter": self.data_parameter,
            "parameter_schema": dict(self.parameter_schema),
            "capabilities": list(self.capabilities),
            "output_formats": list(self.output_formats),
            "examples": [dict(example) for example in self.examples],
            "column_parameters": list(self.column_parameters),
            "aliases": list(self.aliases),
        }

    def to_openai_tool(self) -> Dict[str, Any]:
        """Return an OpenAI-compatible function tool definition."""
        return {
            "type": "function",
            "function": {
                "name": f"plot_{self.name}",
                "description": self.description or f"Render a {self.name} chart.",
                "parameters": dict(self.parameter_schema),
            },
        }


def infer_plot_descriptor(
    name: str,
    function: PlotCallable,
    *,
    description: str = "",
    backend: str = "matplotlib",
    data_parameter: Optional[str] = None,
    capabilities: Sequence[str] = ("render",),
    output_formats: Sequence[str] = ("figure", "png", "svg"),
    examples: Sequence[Mapping[str, Any]] = (),
    aliases: Sequence[str] = (),
) -> PlotDescriptor:
    """Infer a descriptor from a plotting function signature."""
    signature = inspect.signature(function)
    candidate_names = ("pd_df", "data", "df", "frame")
    resolved_data_parameter = data_parameter
    if resolved_data_parameter is None:
        resolved_data_parameter = next(
            (
                candidate
                for candidate in candidate_names
                if candidate in signature.parameters
            ),
            next(iter(signature.parameters), "pd_df"),
        )

    properties: Dict[str, Any] = {}
    required: List[str] = []
    column_parameters: List[str] = []
    column_candidates = {
        "x",
        "y",
        "z",
        "value",
        "values",
        "category",
        "group",
        "label",
        "column",
        "columns",
        "index",
        "source",
        "target",
        "text_column",
        "weight_column",
        "path",
        "labels",
        "parents",
    }

    for parameter in signature.parameters.values():
        if parameter.name in {resolved_data_parameter, "self", "cls"}:
            continue
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        schema = _json_type(parameter.annotation, parameter.default)
        if parameter.default is not inspect.Parameter.empty:
            if parameter.default is not None and isinstance(
                parameter.default, (str, int, float, bool, list, tuple, dict)
            ):
                schema["default"] = parameter.default
        else:
            required.append(parameter.name)
        properties[parameter.name] = schema
        if parameter.name in column_candidates:
            column_parameters.append(parameter.name)

    parameter_schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
    }
    if required:
        parameter_schema["required"] = required

    return PlotDescriptor(
        name=name,
        function=function,
        description=description,
        backend=backend,
        data_parameter=resolved_data_parameter,
        parameter_schema=parameter_schema,
        capabilities=tuple(capabilities),
        output_formats=tuple(output_formats),
        examples=tuple(dict(example) for example in examples),
        column_parameters=tuple(column_parameters),
        aliases=tuple(aliases),
    )


@runtime_checkable
class Plugin(Protocol):
    """Minimal contract implemented by MatplotLibAPI plugins."""

    name: str
    api_version: str

    def setup(self, context: "PluginContext") -> None:
        """Register plotting functions in ``context``."""


@dataclass
class PluginContext:
    """Registration and discovery context passed to plugins."""

    _plots: Dict[str, PlotCallable] = field(default_factory=dict)
    _descriptors: Dict[str, PlotDescriptor] = field(default_factory=dict)
    _aliases: Dict[str, str] = field(default_factory=dict)

    def register_plot(
        self,
        name: str,
        function: PlotCallable,
        *,
        descriptor: Optional[PlotDescriptor] = None,
        description: str = "",
        backend: str = "matplotlib",
        data_parameter: Optional[str] = None,
        capabilities: Sequence[str] = ("render",),
        output_formats: Sequence[str] = ("figure", "png", "svg"),
        examples: Sequence[Mapping[str, Any]] = (),
        aliases: Sequence[str] = (),
    ) -> None:
        """Register one named plotting callable and its discoverable contract."""
        if not name or name in self._plots or name in self._aliases:
            raise DuplicatePluginError(f"Plot already registered: {name!r}")
        resolved = descriptor or infer_plot_descriptor(
            name,
            function,
            description=description,
            backend=backend,
            data_parameter=data_parameter,
            capabilities=capabilities,
            output_formats=output_formats,
            examples=examples,
            aliases=aliases,
        )
        if resolved.name != name or resolved.function is not function:
            raise PluginError("Descriptor name and function must match registration")
        for alias in resolved.aliases:
            if not alias or alias in self._plots or alias in self._aliases:
                raise DuplicatePluginError(f"Plot alias already registered: {alias!r}")
        self._plots[name] = function
        self._descriptors[name] = resolved
        for alias in resolved.aliases:
            self._aliases[alias] = name

    def resolve_name(self, name: str) -> str:
        """Resolve a canonical plot name from a name or alias."""
        return self._aliases.get(name, name)

    def get_plot(self, name: str) -> PlotCallable:
        """Return a registered plot callable by name or alias."""
        canonical = self.resolve_name(name)
        try:
            return self._plots[canonical]
        except KeyError as exc:
            raise PluginError(f"Unknown plot: {name!r}") from exc

    def get_descriptor(self, name: str) -> PlotDescriptor:
        """Return a schema-rich descriptor by name or alias."""
        canonical = self.resolve_name(name)
        try:
            return self._descriptors[canonical]
        except KeyError as exc:
            raise PluginError(f"Unknown plot: {name!r}") from exc

    def describe_plot(self, name: str) -> Dict[str, Any]:
        """Return serializable metadata for one plot."""
        return self.get_descriptor(name).to_dict()

    def list_plots(self) -> List[str]:
        """Return registered canonical plot names in deterministic order."""
        return sorted(self._plots)

    def list_aliases(self) -> Dict[str, str]:
        """Return compatibility aliases in deterministic order."""
        return {name: self._aliases[name] for name in sorted(self._aliases)}

    def list_descriptors(self) -> List[Dict[str, Any]]:
        """Return deterministic serializable plot descriptors."""
        return [self._descriptors[name].to_dict() for name in self.list_plots()]

    def openai_tools(self) -> List[Dict[str, Any]]:
        """Generate tool definitions from the canonical descriptors."""
        return [self._descriptors[name].to_openai_tool() for name in self.list_plots()]


class PluginRegistry:
    """Load, validate, and expose MatplotLibAPI plugins."""

    def __init__(self) -> None:
        """Initialize an empty deterministic plugin registry."""
        self.context = PluginContext()
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_api_versions: Dict[str, str] = {}

    def register(self, plugin: Plugin) -> None:
        """Register one plugin atomically."""
        if not isinstance(plugin, Plugin):
            raise PluginError("Plugin does not implement the required contract")
        if plugin.api_version not in {PLUGIN_API_VERSION, LEGACY_PLUGIN_API_VERSION}:
            raise PluginError(
                f"Unsupported plugin API {plugin.api_version!r}; "
                f"expected {PLUGIN_API_VERSION!r}"
            )
        if plugin.name in self._plugins:
            raise DuplicatePluginError(f"Plugin already registered: {plugin.name!r}")

        previous_plots = dict(self.context._plots)
        previous_descriptors = dict(self.context._descriptors)
        previous_aliases = dict(self.context._aliases)
        try:
            plugin.setup(self.context)
        except Exception:
            self.context._plots = previous_plots
            self.context._descriptors = previous_descriptors
            self.context._aliases = previous_aliases
            raise
        self._plugins[plugin.name] = plugin
        self._plugin_api_versions[plugin.name] = plugin.api_version

    def compatibility_report(self) -> Dict[str, Any]:
        """Return legacy plugin and alias compatibility diagnostics."""
        legacy_plugins = sorted(
            name
            for name, version in self._plugin_api_versions.items()
            if version == LEGACY_PLUGIN_API_VERSION
        )
        return {
            "canonical_plugin_api": PLUGIN_API_VERSION,
            "legacy_plugin_api": LEGACY_PLUGIN_API_VERSION,
            "legacy_plugins": legacy_plugins,
            "aliases": self.context.list_aliases(),
            "ready_for_plugin_api_v1_removal": not legacy_plugins,
        }

    def discover(self) -> None:
        """Load plugins installed through Python entry points."""
        discovered = metadata.entry_points()
        if hasattr(discovered, "select"):
            entries: Iterable[metadata.EntryPoint] = discovered.select(
                group=ENTRY_POINT_GROUP
            )
        else:  # pragma: no cover - Python 3.9 compatibility
            legacy = cast(
                Mapping[str, Iterable[metadata.EntryPoint]],
                discovered,
            )
            entries = legacy.get(ENTRY_POINT_GROUP, ())
        for entry in sorted(entries, key=lambda item: item.name):
            plugin_factory = entry.load()
            plugin = plugin_factory()
            if plugin.name in self._plugins:
                continue
            self.register(plugin)

    def list_plugins(self) -> List[str]:
        """Return registered plugin names in deterministic order."""
        return sorted(self._plugins)


class CorePlotsPlugin:
    """Built-in plugin exposing the stable package plotting API."""

    name = "core"
    api_version = PLUGIN_API_VERSION

    def setup(self, context: PluginContext) -> None:
        """Register stable package-root plotting functions."""
        from . import (
            fplot_area,
            fplot_bar,
            fplot_box_violin,
            fplot_correlation_matrix,
            fplot_heatmap,
            fplot_histogram_kde,
            fplot_pie_donut,
            fplot_sankey,
            fplot_sunburst,
            fplot_table,
            fplot_timeseries,
            fplot_treemap,
            fplot_waffle,
            fplot_wordcloud,
        )

        plots = {
            "area": (fplot_area, "Render an area chart.", ()),
            "bar": (fplot_bar, "Render a bar or stacked-bar chart.", ()),
            "box_violin": (
                fplot_box_violin,
                "Render a box or violin distribution chart.",
                (),
            ),
            "correlation_matrix": (
                fplot_correlation_matrix,
                "Render a numeric correlation matrix.",
                (),
            ),
            "heatmap": (fplot_heatmap, "Render a pivoted heatmap.", ()),
            "histogram_kde": (
                fplot_histogram_kde,
                "Render a histogram with optional density estimation.",
                ("histogram",),
            ),
            "pie_donut": (
                fplot_pie_donut,
                "Render a pie or donut chart.",
                ("pie",),
            ),
            "sankey": (fplot_sankey, "Render a Sankey flow chart.", ()),
            "sunburst": (fplot_sunburst, "Render a sunburst chart.", ()),
            "table": (fplot_table, "Render a formatted table.", ()),
            "timeseries": (
                fplot_timeseries,
                "Render a time-series chart.",
                ("timeserie",),
            ),
            "treemap": (fplot_treemap, "Render a treemap.", ()),
            "waffle": (fplot_waffle, "Render a waffle chart.", ()),
            "wordcloud": (fplot_wordcloud, "Render a word cloud.", ()),
        }
        plotly_names = {"sankey", "sunburst", "treemap"}
        for name, (function, description, aliases) in plots.items():
            context.register_plot(
                name,
                function,
                description=description,
                backend="plotly" if name in plotly_names else "matplotlib",
                output_formats=(
                    ("figure", "png", "json")
                    if name in plotly_names
                    else ("figure", "png", "svg")
                ),
                aliases=aliases,
            )


def create_registry(*, discover: bool = True) -> PluginRegistry:
    """Create a registry with core plots and optional third-party plugins."""
    registry = PluginRegistry()
    registry.register(CorePlotsPlugin())
    if discover:
        registry.discover()
    return registry


__all__ = [
    "ENTRY_POINT_GROUP",
    "LEGACY_PLUGIN_API_VERSION",
    "PLUGIN_API_VERSION",
    "CorePlotsPlugin",
    "DuplicatePluginError",
    "PlotDescriptor",
    "Plugin",
    "PluginContext",
    "PluginError",
    "PluginRegistry",
    "create_registry",
    "infer_plot_descriptor",
]
