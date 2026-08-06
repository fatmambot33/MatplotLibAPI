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

from .evaluations import run_agent_evaluations
from .executor import (
    RenderPolicy,
    execute_plot,
    inspect_dataframe,
    openai_tool_definitions,
    recommend_plot,
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
    return pd.read_csv(resolved)


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

    plugins = subparsers.add_parser("plugins", help="Inspect installed plugins.")
    plugins.add_subparsers(dest="plugins_command", required=True).add_parser(
        "list", help="List installed plugins."
    )

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

    recommend = subparsers.add_parser(
        "recommend", help="Recommend a chart deterministically."
    )
    recommend.add_argument("data")

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
    dependencies = {}
    for name in ("pandas", "matplotlib", "networkx", "plotly"):
        try:
            module = importlib.import_module(name)
            dependencies[name] = {
                "available": True,
                "version": getattr(module, "__version__", "unknown"),
            }
        except ImportError:
            dependencies[name] = {"available": False}
    registry = create_registry()
    return {
        "ok": all(value["available"] for value in dependencies.values()),
        "python": platform.python_version(),
        "dependencies": dependencies,
        "plugins": registry.list_plugins(),
        "plots": registry.context.list_plots(),
        "credentials_required": False,
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
            _print_json({"plugins": registry.list_plugins()})
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
                    value["output"]["format"] = Path(args.output).suffix.lstrip(".") or "png"
                spec = PlotSpec.from_dict(value)
            result = execute_plot(spec, data=data, registry=registry, policy=policy)
            _print_json(result.to_dict())
        elif args.command == "inspect":
            _print_json(inspect_dataframe(_load_csv(args.data, policy)))
        elif args.command == "recommend":
            _print_json(recommend_plot(_load_csv(args.data, policy)))
        elif args.command == "doctor":
            result = _doctor()
            _print_json(result)
            return 0 if result["ok"] else 1
        elif args.command == "test":
            descriptors = registry.context.list_descriptors()
            result = {
                "ok": bool(descriptors),
                "descriptor_count": len(descriptors),
                "schema_version": PlotSpec.json_schema()["properties"]["schema_version"]["const"],
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
