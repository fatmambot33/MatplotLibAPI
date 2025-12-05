"""Ensure pyproject.toml lists all runtime dependencies."""

from pathlib import Path

try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # Python < 3.11 fallback
    import tomli as tomllib  # type: ignore


RUNTIME_DEPENDENCIES = {
    "matplotlib",
    "networkx",
    "numpy",
    "pandas",
    "plotly",
    "seaborn",
}


def test_pyproject_contains_required_dependencies() -> None:
    """Verify required runtime libraries are declared in pyproject.toml."""

    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = {
        dep.split("[")[0]
        .split("==")[0]
        .split(">=")[0]
        .split("~=")[0]
        .strip()
        for dep in pyproject["project"]["dependencies"]
    }

    missing_dependencies = RUNTIME_DEPENDENCIES - dependencies
    assert not missing_dependencies, (
        "Missing dependencies in pyproject.toml: "
        f"{sorted(missing_dependencies)}"
    )
