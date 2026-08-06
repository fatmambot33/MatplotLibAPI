---
name: matplotlibapi
description: Discover, validate, and render local charts through MatplotLibAPI's schema-driven plotting contract.
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
```

## Canonical workflow

1. Inspect the data locally with `matplotlibapi inspect` or
   `inspect_dataframe`.
2. Discover plot descriptors with `matplotlibapi plots list` or
   `create_registry().context.list_descriptors()`.
3. Choose the simplest chart that answers the user's question. Use
   `recommend_plot` only as a deterministic starting point.
4. Build a `PlotSpec` using semantic column roles in `encoding` and visual
   settings in `options`.
5. Validate the specification before execution.
6. Render through `execute_plot`, the CLI, or MCP. Do not bypass the canonical
   executor with custom plotting code unless the supported registry cannot
   express the requested result.
7. Keep input and output paths inside an explicit workspace.
8. Report the resulting path and any structured warnings.

## Discovery

```python
from MatplotLibAPI import PlotSpec, create_registry, openai_tool_definitions

registry = create_registry()
print(registry.context.list_descriptors())
print(PlotSpec.json_schema())
print(openai_tool_definitions(registry=registry))
```

Use Matplotlib for static output and Plotly only when interactivity materially
improves the result. Prefer deterministic, reproducible transformations and
preserve source data.
