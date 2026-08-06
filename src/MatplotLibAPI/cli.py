"""Command-line interface for local schema-driven plotting."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from .conformance import validate_registry_conformance, write_plugin_scaffold
from .evaluations import run_agent_evaluations
from .executor import (
    RenderPolicy,
    execute_plot,
    inspect_dataframe,
    openai_tool_definitions,
    recommend_plot,
)
from .intelligence import suggest_plot_spec_repairs
from .migration import (
    audit_plot_spec_for_v5,
    migrate_plot_spec_for_v5,
    v5_compatibility_status,
)
from .plugins import create_registry
from .specs import PlotSpec, PlotValidationError, ValidationIssue


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _load_csv(path: str, policy: RenderPolicy) -> pd.DataFrame:
    resolved = policy.resolve_input_path(path)
    size = resolved.stat().st_size
    if size > policy.max_input_bytes:
        raise PlotValidationError(
            [
                ValidationIssue(
                    code="policy.max_input_bytes_exceeded",
                    message=(
                        f"Input is {size} bytes; maximum is "
                        f"{policy.max_input_bytes}."
                    ),
                )
            ]
        )
    frame = pd.read_csv(resolved)
    rows, columns = frame.shape
    if rows > policy.max_rows or columns > policy.max_columns:
        raise PlotValidationError(
            [
                ValidationIssue(
                    code="policy.profile_limit_exceeded",
                    message="Input exceeds the configured profiling limits.",
                    details={"rows": rows, "columns": columns},
                )
            ]
        )
    return frame


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="matplotlibapi",
        description="Schema-driven local plotting for people and agents.",
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace used to constrain local input and output paths.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plots = subparsers.add_parser("plots", help="Discover plotting capabilities.")
    plots_sub = plots.add_subparsers(dest="plots_command", required=True)
    plots_sub.add_parser("list", help="List canonical plots.")
    describe = plots_sub.add_parser("describe", help="Describe one plot.")
    describe.add_argument("chart")

    plugins = subparsers.add_parser("plugins", help="Inspect and build plugins.")
    plugins_sub = plugins.add_subparsers(dest="plugins_command", required=True)
    plugins_sub.add_parser("list", help="List installed plugins.")
    plugins_sub.add_parser("conform", help="Validate registry conformance.")
    scaffold = plugins_sub.add_parser(
        "scaffold", help="Create an official plugin project template."
    )
    scaffold.add_argument("name")
    scaffold.add_argument("destination")
    scaffold.add_argument("--force", action="store_true")

    schema = subparsers.add_parser("schema", help="Print canonical schemas.")
    schema.add_argument(
        "kind", choices=("plot-spec", "openai-tools"), default="plot-spec"
    )

    validate = subparsers.add_parser("validate", help="Validate a plot spec.")
    validate.add_argument("spec")

    render = subparsers.add_parser("render", help="Render a plot spec.")
    render.add_argument("spec")
    render.add_argument("--data", help="CSV path overriding spec.data.")
    render.add_argument("--output", help="Output path overriding spec.output.path.")

    inspect_parser = subparsers.add_parser("inspect", help="Profile a CSV file.")
    inspect_parser.add_argument("data")
    inspect_parser.add_argument("--max-rows", type=int, default=5_000)

    recommend = subparsers.add_parser(
        "recommend", help="Recommend charts deterministically."
    )
    recommend.add_argument("data")

    repair = subparsers.add_parser(
        "repair", help="Suggest safe opt-in repairs for a plot spec."
    )
    repair.add_argument("spec")
    repair.add_argument("--data", required=True)

    migrate = subparsers.add_parser(
        "migrate", help="Audit and migrate a plot spec for 5.0 canonical names."
    )
    migrate.add_argument("spec")
    migrate.add_argument("--write", help="Optional path for the migrated JSON spec.")

    presets = subparsers.add_parser(
        "presets", help="List accessible and semantic presentation presets."
    )
    presets.add_argument("kind", choices=("list",), default="list")

    subparsers.add_parser(
        "compatibility", help="Show plugin and 5.0 compatibility gates."
    )
    subparsers.add_parser("doctor", help="Check the local installation contract.")
    subparsers.add_parser("test", help="Run a lightweight installation self-test.")
    subparsers.add_parser("eval", help="Run deterministic agent evaluations.")
    benchmark = subparsers.add_parser("benchmark", help="Benchmark local discovery.")
    benchmark.add_argument("--iterations", type=int, default=100)
    subparsers.add_parser("docs", help="Show documentation entry points.")
    subparsers.add_parser("examples", help="Show executable example entry points.")
    subparsers.add_parser("upgrade", help="Show the safe package upgrade command.")
    subparsers.add_parser("uninstall", help="Show the package uninstall command.")
    return parser


def _doctor() -> Dict[str, Any]:
    core_dependencies = ("pandas", "matplotlib")
    optional_dependencies = (
        "networkx",
        "plotly",
        "seaborn",
        "sklearn",
        "wordcloud",
        "kaleido",
        "nbformat",
    )
    dependencies: Dict[str, Dict[str, Any]] = {}
    for name in core_dependencies + optional_dependencies:
        try:
            module = importlib.import_module(name)
            dependencies[name] = {
                "available": True,
                "version": getattr(module, "__version__", "unknown"),
                "required": name in core_dependencies,
            }
        except ImportError:
            dependencies[name] = {
                "available": False,
                "required": name in core_dependencies,
            }
    registry = create_registry()
    core_ok = all(dependencies[name]["available"] for name in core_dependencies)
    return {
        "ok": core_ok,
        "python": platform.python_version(),
        "dependencies": dependencies,
        "plugins": registry.list_plugins(),
        "plots": registry.context.list_plots(),
        "compatibility": registry.compatibility_report(),
        "credentials_required": False,
    }


def _presentation_presets() -> Dict[str, Any]:
    return {
        "accessibility": ["default", "high-contrast", "colorblind"],
        "number_formats": [
            "auto",
            "number",
            "integer",
            "percent",
            "currency",
            "compact",
        ],
        "example": {
            "presentation": {
                "accessibility": "colorblind",
                "number_format": "currency",
                "currency": "EUR",
                "alt_text": "Monthly revenue by market.",
                "show_grid": True,
            }
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command-line interface and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    policy = RenderPolicy(workspace=Path(args.workspace))
    try:
        registry = create_registry()
        if args.command == "plots":
            if args.plots_command == "list":
                _print_json(registry.context.list_descriptors())
            else:
                _print_json(registry.context.describe_plot(args.chart))
        elif args.command == "plugins":
            if args.plugins_command == "list":
                _print_json(
                    {
                        "plugins": registry.list_plugins(),
                        "compatibility": registry.compatibility_report(),
                    }
                )
            elif args.plugins_command == "conform":
                result = validate_registry_conformance(registry)
                _print_json(result.to_dict())
                return 0 if result.passed else 1
            else:
                destination = policy.resolve_output_path(args.destination)
                created = write_plugin_scaffold(
                    args.name,
                    destination,
                    overwrite=args.force,
                )
                _print_json(
                    {
                        "created": list(created),
                        "destination": str(destination),
                        "plugin_api": "2",
                    }
                )
        elif args.command == "schema":
            if args.kind == "plot-spec":
                _print_json(PlotSpec.json_schema())
            else:
                _print_json(openai_tool_definitions(registry=registry))
        elif args.command == "validate":
            spec = PlotSpec.from_path(args.spec)
            _print_json({"valid": True, "spec": spec.to_dict()})
        elif args.command == "render":
            spec = PlotSpec.from_path(args.spec)
            data = _load_csv(args.data, policy) if args.data else None
            if args.output:
                value = spec.to_dict()
                value["output"] = dict(value["output"])
                value["output"]["path"] = args.output
                if value["output"].get("format") == "figure":
                    value["output"]["format"] = (
                        Path(args.output).suffix.lstrip(".") or "png"
                    )
                spec = PlotSpec.from_dict(value)
            result = execute_plot(spec, data=data, registry=registry, policy=policy)
            _print_json(result.to_dict())
        elif args.command == "inspect":
            frame = _load_csv(args.data, policy)
            _print_json(inspect_dataframe(frame, max_rows=max(1, args.max_rows)))
        elif args.command == "recommend":
            _print_json(recommend_plot(_load_csv(args.data, policy)))
        elif args.command == "repair":
            spec = PlotSpec.from_path(args.spec)
            frame = _load_csv(args.data, policy)
            suggestions = suggest_plot_spec_repairs(
                spec,
                frame,
                registry=registry,
            )
            _print_json(
                {
                    "spec": spec.to_dict(),
                    "suggestions": [item.to_dict() for item in suggestions],
                    "applied": False,
                }
            )
        elif args.command == "migrate":
            spec = PlotSpec.from_path(args.spec)
            notices = audit_plot_spec_for_v5(spec)
            migrated = migrate_plot_spec_for_v5(spec)
            if args.write:
                output = policy.resolve_output_path(args.write)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(migrated.to_json() + "\n", encoding="utf-8")
            _print_json(
                {
                    "notices": [notice.to_dict() for notice in notices],
                    "migrated": migrated.to_dict(),
                    "output_path": str(output) if args.write else None,
                }
            )
        elif args.command == "presets":
            _print_json(_presentation_presets())
        elif args.command == "compatibility":
            _print_json(
                {
                    "plugins": registry.compatibility_report(),
                    "v5": v5_compatibility_status(),
                }
            )
        elif args.command == "doctor":
            result = _doctor()
            _print_json(result)
            return 0 if result["ok"] else 1
        elif args.command == "test":
            descriptors = registry.context.list_descriptors()
            conformance = validate_registry_conformance(registry)
            result = {
                "ok": bool(descriptors) and conformance.passed,
                "descriptor_count": len(descriptors),
                "schema_version": PlotSpec.json_schema()["properties"][
                    "schema_version"
                ]["const"],
                "conformance": conformance.to_dict(),
            }
            _print_json(result)
            return 0 if result["ok"] else 1
        elif args.command == "eval":
            result = run_agent_evaluations()
            _print_json(result)
            return 0 if result["failed"] == 0 else 1
        elif args.command == "benchmark":
            iterations = max(1, args.iterations)
            started = perf_counter()
            for _ in range(iterations):
                registry.context.list_descriptors()
                registry.context.openai_tools()
            duration = perf_counter() - started
            _print_json(
                {
                    "iterations": iterations,
                    "duration_seconds": duration,
                    "operations_per_second": (iterations * 2) / duration,
                }
            )
        elif args.command == "docs":
            _print_json(
                {
                    "readme": "README.md",
                    "plot_spec": "docs/PLOT_SPEC.md",
                    "intelligence": "docs/DATA_INTELLIGENCE.md",
                    "plugins": "docs/PLUGIN_ECOSYSTEM.md",
                    "migration": "docs/MIGRATING_TO_5.md",
                    "api_reference": "docs/API_REFERENCE.md",
                }
            )
        elif args.command == "examples":
            _print_json(
                {
                    "gallery": "python -m examples.gallery",
                    "sample_data": "python scripts/generate_sample_data.py",
                }
            )
        elif args.command == "upgrade":
            _print_json(
                {"command": f"{sys.executable} -m pip install --upgrade MatplotLibAPI"}
            )
        elif args.command == "uninstall":
            _print_json({"command": f"{sys.executable} -m pip uninstall MatplotLibAPI"})
        return 0
    except PlotValidationError as exc:
        _print_json(exc.to_dict())
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _print_json({"error": "command_failed", "message": str(exc)})
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
