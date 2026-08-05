"""Typed plugin discovery and registration for MatplotLibAPI."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Callable, Dict, Iterable, List, Protocol, runtime_checkable

PLUGIN_API_VERSION = "1"
ENTRY_POINT_GROUP = "matplotlibapi.plugins"
PlotCallable = Callable[..., Any]


class PluginError(RuntimeError):
    """Base error raised by the plugin surface."""


class DuplicatePluginError(PluginError):
    """Raised when a plugin or plot name is registered twice."""


@runtime_checkable
class Plugin(Protocol):
    """Minimal contract implemented by MatplotLibAPI plugins."""

    name: str
    api_version: str

    def setup(self, context: "PluginContext") -> None:
        """Register plotting functions in ``context``."""


@dataclass
class PluginContext:
    """Registration context passed to plugins."""

    _plots: Dict[str, PlotCallable] = field(default_factory=dict)

    def register_plot(self, name: str, function: PlotCallable) -> None:
        """Register one named plotting callable.

        Parameters
        ----------
        name:
            Stable public plot name.
        function:
            Callable implementing the plot.
        """
        if not name or name in self._plots:
            raise DuplicatePluginError(f"Plot already registered: {name!r}")
        self._plots[name] = function

    def get_plot(self, name: str) -> PlotCallable:
        """Return a registered plot callable by name."""
        try:
            return self._plots[name]
        except KeyError as exc:
            raise PluginError(f"Unknown plot: {name!r}") from exc

    def list_plots(self) -> List[str]:
        """Return registered plot names in deterministic order."""
        return sorted(self._plots)


class PluginRegistry:
    """Load, validate, and expose MatplotLibAPI plugins."""

    def __init__(self) -> None:
        self.context = PluginContext()
        self._plugins: Dict[str, Plugin] = {}

    def register(self, plugin: Plugin) -> None:
        """Register one plugin atomically."""
        if not isinstance(plugin, Plugin):
            raise PluginError("Plugin does not implement the required contract")
        if plugin.api_version != PLUGIN_API_VERSION:
            raise PluginError(
                f"Unsupported plugin API {plugin.api_version!r}; "
                f"expected {PLUGIN_API_VERSION!r}"
            )
        if plugin.name in self._plugins:
            raise DuplicatePluginError(f"Plugin already registered: {plugin.name!r}")

        previous = dict(self.context._plots)
        try:
            plugin.setup(self.context)
        except Exception:
            self.context._plots = previous
            raise
        self._plugins[plugin.name] = plugin

    def discover(self) -> None:
        """Load plugins installed through Python entry points."""
        discovered = metadata.entry_points()
        if hasattr(discovered, "select"):
            entries: Iterable[metadata.EntryPoint] = discovered.select(
                group=ENTRY_POINT_GROUP
            )
        else:  # pragma: no cover - Python 3.9 compatibility
            entries = discovered.get(ENTRY_POINT_GROUP, [])
        for entry in sorted(entries, key=lambda item: item.name):
            plugin_factory = entry.load()
            self.register(plugin_factory())

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
            fplot_timeserie,
            fplot_treemap,
            fplot_waffle,
            fplot_wordcloud,
        )

        plots = {
            "area": fplot_area,
            "bar": fplot_bar,
            "box_violin": fplot_box_violin,
            "correlation_matrix": fplot_correlation_matrix,
            "heatmap": fplot_heatmap,
            "histogram_kde": fplot_histogram_kde,
            "pie_donut": fplot_pie_donut,
            "sankey": fplot_sankey,
            "sunburst": fplot_sunburst,
            "table": fplot_table,
            "timeserie": fplot_timeserie,
            "treemap": fplot_treemap,
            "waffle": fplot_waffle,
            "wordcloud": fplot_wordcloud,
        }
        for name, function in plots.items():
            context.register_plot(name, function)


def create_registry(*, discover: bool = True) -> PluginRegistry:
    """Create a registry with core plots and optional third-party plugins."""
    registry = PluginRegistry()
    registry.register(CorePlotsPlugin())
    if discover:
        registry.discover()
    return registry


__all__ = [
    "ENTRY_POINT_GROUP",
    "PLUGIN_API_VERSION",
    "CorePlotsPlugin",
    "DuplicatePluginError",
    "Plugin",
    "PluginContext",
    "PluginError",
    "PluginRegistry",
    "create_registry",
]
