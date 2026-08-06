# Agent evaluations and benchmarks

MatplotLibAPI evaluates agent-facing behavior without requiring an LLM,
credentials, network access, or a hosted service.

## Evaluation baseline

Run:

```bash
matplotlibapi eval
```

The baseline verifies:

- categorical plus numeric data recommends a bar chart
- one numeric column recommends a histogram
- multiple numeric columns recommend a correlation matrix
- non-numeric data recommends a table
- invalid specifications are rejected with stable codes
- local profiling reports missing values correctly

The command emits machine-readable JSON and exits non-zero if any case fails.

## Discovery benchmark

Run:

```bash
matplotlibapi benchmark --iterations 1000
python scripts/benchmark_agent_plotting.py --iterations 1000
```

The benchmark repeatedly generates registry descriptors, OpenAI tool schemas,
and the canonical plot schema. It is intentionally local and deterministic.
The script has an explicit duration budget and exits non-zero when exceeded.

Rendering benchmarks should use fixed input fixtures, the non-interactive
Matplotlib backend, stable dimensions, and explicit output formats. Performance
budgets must be generous enough to avoid platform-dependent failures while still
catching large regressions.
