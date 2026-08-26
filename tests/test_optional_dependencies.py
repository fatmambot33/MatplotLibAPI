"""Contracts for minimal and optional runtime dependency boundaries."""

from pathlib import Path

try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # Python 3.9-3.10
    import tomli as tomllib  # type: ignore


def _names(values: list[str]) -> set[str]:
    """Return normalized package names from requirement strings."""
    return {
        value.split("[")[0].split("==")[0].split(">=")[0].split("~=")[0].strip().lower()
        for value in values
    }


def _project() -> dict:
    """Return project metadata from pyproject.toml."""
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_plotly_export_dependencies_are_optional() -> None:
    """Keep static Plotly export tooling out of the minimal installation."""
    project = _project()
    runtime = _names(project["dependencies"])
    extras = project["optional-dependencies"]
    plotly_export = _names(extras["plotly-export"])

    assert "kaleido" not in runtime
    assert "nbformat" not in runtime
    assert {"kaleido", "nbformat"} <= plotly_export


def test_all_extra_expands_optional_runtime_features() -> None:
    """Keep the convenience extra explicit and non-recursive."""
    extras = _project()["optional-dependencies"]
    all_names = _names(extras["all"])

    assert {"kaleido", "nbformat"} <= all_names
    assert "matplotlibapi" not in all_names
    assert "mcp" not in all_names


def test_distribution_has_no_owned_mcp_surface() -> None:
    """Keep MCP out of this project's dependencies, extras, and executables."""
    project = _project()
    runtime = _names(project["dependencies"])
    extras = project["optional-dependencies"]
    scripts = project["scripts"]

    assert "mcp" not in runtime
    assert "mcp" not in extras
    assert "matplotlibapi-mcp" not in scripts
    assert not Path("src/MatplotLibAPI/mcp_server.py").exists()
    assert not Path("src/MatplotLibAPI/mcp").exists()
