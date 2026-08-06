"""Run a lightweight, deterministic schema-discovery benchmark."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

from MatplotLibAPI import PlotSpec, create_registry


def main() -> int:
    """Run the benchmark and print a JSON result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--budget-seconds", type=float, default=2.0)
    args = parser.parse_args()

    registry = create_registry()
    started = perf_counter()
    for _ in range(max(1, args.iterations)):
        registry.context.list_descriptors()
        registry.context.openai_tools()
        PlotSpec.json_schema()
    duration = perf_counter() - started
    result = {
        "iterations": args.iterations,
        "duration_seconds": duration,
        "budget_seconds": args.budget_seconds,
        "passed": duration <= args.budget_seconds,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
