# Agent evaluations and benchmarks

MatplotLibAPI evaluates agent-facing behavior without requiring an LLM,
credentials, network access, or a hosted service.

## Evaluation baseline

Run:

```bash
matplotlibapi eval
```

The machine-readable baseline verifies:

- categorical and numeric data recommends a bar chart
- numeric distributions and correlations are ranked deterministically
- datetime and numeric data uses the canonical `timeseries` chart
- recommendation results include scores and explicit reasons
- bounded profiles report semantic roles, missingness, and truncation
- invalid specifications are rejected with stable codes
- safe repair suggestions are non-mutating and explicitly applicable
- the core registry passes plugin conformance
- legacy chart names produce 5.0 migration notices
- the breaking-removal gate remains closed before 2027-02-06

The command exits non-zero if any case fails.

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
