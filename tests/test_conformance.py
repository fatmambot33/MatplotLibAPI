"""Tests for the plugin ecosystem conformance surface."""

from pathlib import Path

from MatplotLibAPI.conformance import (
    plugin_template_files,
    validate_registry_conformance,
    write_plugin_scaffold,
)
from MatplotLibAPI.plugins import PluginRegistry


def test_registry_conformance_is_machine_readable() -> None:
    """A valid descriptor registry should produce a passing result."""

    def plot(pd_df, x: str, y: str):
        return pd_df, x, y

    registry = PluginRegistry()
    registry.context.register_plot("line", plot, aliases=("lines",))

    result = validate_registry_conformance(registry)

    assert result.passed is True
    assert result.to_dict()["plot_count"] == 1


def test_plugin_template_is_complete_and_writable(tmp_path: Path) -> None:
    """The official template should create an installable project shape."""
    files = plugin_template_files("Example Plot")
    created = write_plugin_scaffold("Example Plot", tmp_path / "plugin")

    assert "pyproject.toml" in files
    assert "tests/test_conformance.py" in created
    assert (tmp_path / "plugin" / "pyproject.toml").is_file()
