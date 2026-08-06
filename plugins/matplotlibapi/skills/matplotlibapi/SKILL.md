---
name: matplotlibapi
description: Profile local data, explain chart recommendations, repair and validate PlotSpecs, and render accessible charts through MatplotLibAPI's schema-driven contract.
---

# MatplotLibAPI

## Credential policy

MatplotLibAPI requires no credentials. Never request API keys, tokens, hosted
authentication, secret-store access, or remote data uploads. Process only local
files and table data the user has explicitly supplied.

An optional `.env` may contain only non-secret local runtime settings:

```dotenv
MPLBACKEND=Agg
```

## Setup

```bash
python -m pip install --upgrade "git+https://github.com/fatmambot33/MatplotLibAPI.git"
```

Verify the installation:

```bash
matplotlibapi doctor
matplotlibapi test
matplotlibapi plugins conform
```

## Canonical workflow

1. Profile data locally with `matplotlibapi inspect` or `profile_dataframe`.
2. Review ranked, explainable chart options with `matplotlibapi recommend` or
   `recommend_plots`.
3. Discover canonical plot descriptors with `matplotlibapi plots list` or
   `create_registry().context.list_descriptors()`.
4. Build a `PlotSpec` using semantic column roles in `encoding`, visual settings
   in `options`, and accessibility or number-format choices in `presentation`.
5. Use `matplotlibapi repair` or `suggest_plot_spec_repairs` for opt-in,
   structured corrections. Never silently mutate a user specification.
6. Validate before execution.
7. Render through `execute_plot`, the CLI, or MCP. Do not bypass the canonical
   executor unless the registry cannot express the requested result.
8. Keep input and output paths inside an explicit workspace.
9. Report the output path, profile truncation status, recommendation reasons,
   alt text, and any structured warnings.

## Discovery and migration

```python
from MatplotLibAPI import (
    PlotSpec,
    create_registry,
    openai_tool_definitions,
    v5_compatibility_status,
)

registry = create_registry()
print(registry.context.list_descriptors())
print(PlotSpec.json_schema())
print(openai_tool_definitions(registry=registry))
print(v5_compatibility_status())
```

Use `timeseries` as the canonical chart name. Treat `timeserie`, `histogram`, and
`pie` as compatibility aliases and use migration diagnostics before 5.0. The
breaking-removal gate must remain closed before its documented date.

Use Matplotlib for static output and Plotly only when interactivity materially
improves the result. Prefer deterministic, reproducible transformations and
preserve source data.
