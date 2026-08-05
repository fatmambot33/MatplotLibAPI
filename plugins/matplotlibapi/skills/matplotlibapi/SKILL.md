---
name: matplotlibapi
description: Install and use MatplotLibAPI from Git to inspect tabular data, select appropriate visualizations, render charts, and save reproducible figures.
---

# MatplotLibAPI

## Setup

Ensure the current Python environment has the repository version installed:

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

1. Inspect columns, dtypes, missing values, and row counts.
2. Choose the simplest chart that answers the user's question.
3. Prefer registered `fplot_*` helpers over custom plotting code.
4. Keep transformations explicit and preserve the source data.
5. Render headlessly when needed with `MPLBACKEND=Agg`.
6. Save the figure to a clear output path and report that path.

Use Matplotlib figures for static outputs and the registered Plotly helpers when interactive output materially improves the result.
