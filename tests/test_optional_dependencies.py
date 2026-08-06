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


def test_plotly_export_dependencies_are_optional() -> None:
    """Keep static Plotly export tooling out of the minimal installation."""
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    runtime = _names(project["dependencies"])
    extras = project["optional-dependencies"]
    plotly_export = _names(extras["plotly-export"])

    assert "kaleido" not in runtime
    assert "nbformat" not in runtime
    assert {"kaleido", "nbformat"} <= plotly_export


def test_all_extra_expands_optional_runtime_features() -> None:
    """Keep the convenience extra explicit and non-recursive."""
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    extras = project["optional-dependencies"]
    all_names = _names(extras["all"])

    assert {"mcp", "kaleido", "nbformat"} <= all_names
    assert "matplotlibapi" not in all_names
