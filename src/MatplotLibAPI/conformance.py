"""Plugin scaffolding and deterministic conformance validation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .plugins import PLUGIN_API_VERSION, Plugin, PluginRegistry


@dataclass(frozen=True)
class ConformanceIssue:
    """One machine-readable plugin conformance finding."""

    code: str
    message: str
    path: Tuple[str, ...] = ()
    severity: str = "error"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable finding."""
        return {
            "code": self.code,
            "message": self.message,
            "path": list(self.path),
            "severity": self.severity,
        }


@dataclass(frozen=True)
class ConformanceResult:
    """Structured plugin conformance result."""

    plugin: str
    passed: bool
    plot_count: int
    issues: Tuple[ConformanceIssue, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable conformance result."""
        return {
            "plugin": self.plugin,
            "passed": self.passed,
            "plot_count": self.plot_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def validate_registry_conformance(
    registry: PluginRegistry,
    *,
    plugin_name: str = "registry",
) -> ConformanceResult:
    """Validate descriptors, aliases, examples, and generated tools."""
    issues: List[ConformanceIssue] = []
    context = registry.context
    descriptors = context.list_descriptors()
    names = context.list_plots()
    aliases = context.list_aliases()
    if names != sorted(names):
        issues.append(
            ConformanceIssue(
                code="conformance.nondeterministic_names",
                message="Canonical plot names must be sorted deterministically.",
            )
        )
    for descriptor in descriptors:
        name = str(descriptor.get("name", ""))
        schema = descriptor.get("parameter_schema")
        path = ("plots", name)
        if not name:
            issues.append(
                ConformanceIssue(
                    code="conformance.missing_name",
                    message="Every descriptor requires a canonical name.",
                    path=path,
                )
            )
        if not isinstance(schema, Mapping) or schema.get("type") != "object":
            issues.append(
                ConformanceIssue(
                    code="conformance.invalid_schema",
                    message="parameter_schema must be a JSON object schema.",
                    path=path + ("parameter_schema",),
                )
            )
        elif schema.get("additionalProperties") is not False:
            issues.append(
                ConformanceIssue(
                    code="conformance.open_schema",
                    message="parameter_schema must reject unknown parameters.",
                    path=path + ("parameter_schema", "additionalProperties"),
                )
            )
        formats = descriptor.get("output_formats", [])
        if "figure" not in formats:
            issues.append(
                ConformanceIssue(
                    code="conformance.figure_output_required",
                    message="Every plot must support in-memory figure output.",
                    path=path + ("output_formats",),
                )
            )
        examples = descriptor.get("examples", [])
        if not all(isinstance(example, Mapping) for example in examples):
            issues.append(
                ConformanceIssue(
                    code="conformance.invalid_example",
                    message="Descriptor examples must be objects.",
                    path=path + ("examples",),
                )
            )
    for alias, canonical in sorted(aliases.items()):
        if alias in names:
            issues.append(
                ConformanceIssue(
                    code="conformance.alias_collision",
                    message=f"Alias {alias!r} collides with a canonical name.",
                    path=("aliases", alias),
                )
            )
        if canonical not in names:
            issues.append(
                ConformanceIssue(
                    code="conformance.alias_target_missing",
                    message=f"Alias {alias!r} targets unknown plot {canonical!r}.",
                    path=("aliases", alias),
                )
            )
    tools = context.openai_tools()
    tool_names = [str(tool["function"]["name"]) for tool in tools]
    if len(tool_names) != len(set(tool_names)):
        issues.append(
            ConformanceIssue(
                code="conformance.duplicate_tool_name",
                message="Generated OpenAI tool names must be unique.",
            )
        )
    return ConformanceResult(
        plugin=plugin_name,
        passed=not any(issue.severity == "error" for issue in issues),
        plot_count=len(descriptors),
        issues=tuple(issues),
    )


def validate_plugin_conformance(plugin: Plugin) -> ConformanceResult:
    """Register one plugin in isolation and validate its public contract."""
    registry = PluginRegistry()
    registry.register(plugin)
    return validate_registry_conformance(registry, plugin_name=plugin.name)


def _module_name(value: str) -> str:
    """Return a safe Python module name for a plugin project."""
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    if not normalized:
        raise ValueError("Plugin name must contain letters or digits")
    if normalized[0].isdigit():
        normalized = f"plugin_{normalized}"
    return normalized


def plugin_template_files(name: str) -> Mapping[str, str]:
    """Return a complete minimal plugin project as path-to-content mapping."""
    module = _module_name(name)
    class_name = "".join(part.capitalize() for part in module.split("_")) + "Plugin"
    pyproject = f"""[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n\n[project]\nname = "matplotlibapi-plugin-{module.replace('_', '-')}"\nversion = "0.1.0"\nrequires-python = ">=3.9"\ndependencies = ["MatplotLibAPI>=4.4,<5"]\n\n[project.entry-points."matplotlibapi.plugins"]\n{module} = "{module}.plugin:{class_name}"\n\n[tool.hatch.build.targets.wheel]\npackages = ["src/{module}"]\n"""
    plugin = f'''"""Example MatplotLibAPI plugin generated by the official scaffold."""\n\nfrom __future__ import annotations\n\nfrom typing import Any\n\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\nfrom MatplotLibAPI.plugins import PLUGIN_API_VERSION, PluginContext\n\n\ndef fplot_example(pd_df: pd.DataFrame, x: str, y: str, **kwargs: Any):\n    """Render a minimal example line chart."""\n    figure, axis = plt.subplots()\n    axis.plot(pd_df[x], pd_df[y], **kwargs)\n    return figure\n\n\nclass {class_name}:\n    """First-party-compatible example plugin."""\n\n    name = "{module}"\n    api_version = PLUGIN_API_VERSION\n\n    def setup(self, context: PluginContext) -> None:\n        """Register the example plot and its schema-rich descriptor."""\n        context.register_plot(\n            "{module}_example",\n            fplot_example,\n            description="Render the scaffolded example line chart.",\n            examples=({{"x": "date", "y": "value"}},),\n        )\n'''
    init = f'''"""{name} MatplotLibAPI plugin."""\n\nfrom .plugin import {class_name}, fplot_example\n\n__all__ = ["{class_name}", "fplot_example"]\n'''
    test = f'''"""Conformance test for the scaffolded plugin."""\n\nfrom MatplotLibAPI.conformance import validate_plugin_conformance\nfrom {module}.plugin import {class_name}\n\n\ndef test_plugin_conformance() -> None:\n    """The generated plugin should satisfy the public plugin contract."""\n    result = validate_plugin_conformance({class_name}())\n    assert result.passed, result.to_dict()\n'''
    return {
        "pyproject.toml": pyproject,
        f"src/{module}/__init__.py": init,
        f"src/{module}/plugin.py": plugin,
        "tests/test_conformance.py": test,
        "README.md": (
            f"# {name}\n\nGenerated with `matplotlibapi plugins scaffold`.\n\n"
            f"Plugin API: {PLUGIN_API_VERSION}.\n"
        ),
    }


def write_plugin_scaffold(
    name: str,
    destination: Path,
    *,
    overwrite: bool = False,
) -> Sequence[str]:
    """Write the official plugin template and return created relative paths."""
    files = plugin_template_files(name)
    destination = destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Destination is not empty: {destination}. Use overwrite=True explicitly."
        )
    created: List[str] = []
    for relative_path, content in files.items():
        path = destination / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {path}")
        path.write_text(content, encoding="utf-8")
        created.append(relative_path)
    return tuple(sorted(created))


__all__ = [
    "ConformanceIssue",
    "ConformanceResult",
    "plugin_template_files",
    "validate_plugin_conformance",
    "validate_registry_conformance",
    "write_plugin_scaffold",
]
