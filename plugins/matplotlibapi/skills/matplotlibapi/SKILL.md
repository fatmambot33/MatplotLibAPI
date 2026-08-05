---
name: matplotlibapi
description: Install and use MatplotLibAPI from Git for local, credential-free data inspection and visualization.
---

# MatplotLibAPI

## Credential policy

MatplotLibAPI requires no credentials. Never request API keys, tokens, hosted authentication, or secret-store access for this plugin. All plotting and data processing must run locally against files and data the user has explicitly provided.

An optional local `.env` may be used only for non-secret runtime configuration:

```bash
cp plugins/matplotlibapi/.env.example .env
```

```dotenv
MPLBACKEND=Agg
```

Ensure `.env` is ignored by Git. Do not place credentials in it for this plugin.

## Setup

Install the repository version locally:

```bash
python -m pip install --upgrade "git+https://github.com/fatmambot33/MatplotLibAPI.git"
```

Use the package's plugin registry as the primary discovery surface:

```python
from MatplotLibAPI import create_registry

registry = create_registry()
print(registry.context.list_plots())
```

## Workflow

1. Confirm no external credentials or hosted services are needed.
2. Inspect columns, dtypes, missing values, and row counts locally.
3. Choose the simplest chart that answers the user's question.
4. Prefer registered `fplot_*` helpers over custom plotting code.
5. Keep transformations explicit and preserve source data.
6. Render headlessly when needed with `MPLBACKEND=Agg`.
7. Save the figure to a clear local output path and report that path.

Use Matplotlib for static outputs and registered Plotly helpers only when interactive output materially improves the result.
