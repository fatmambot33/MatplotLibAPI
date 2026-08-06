# MatplotLibAPI

MatplotLibAPI is a typed, local-first plotting engine for pandas users, Python
applications, plugins, CLIs, and AI agents. It provides a small stable plotting
API plus one schema-driven execution contract that can be discovered and used
without credentials or hosted services.

## Installation

```bash
pip install MatplotLibAPI
```

Install optional MCP support with:

```bash
pip install "MatplotLibAPI[mcp]"
```

## Python quick start

```python
import pandas as pd
from MatplotLibAPI import fplot_bar

sales = pd.DataFrame(
    {
        "product": ["A", "A", "B", "B"],
        "region": ["North", "South", "North", "South"],
        "revenue": [12, 9, 15, 11],
    }
)

figure = fplot_bar(
    sales,
    category="product",
    value="revenue",
    group="region",
    stacked=True,
)
```

## Canonical plot specification

`PlotSpec` is the portable contract shared by Python, plugins, the CLI, MCP,
Codex, and generated OpenAI tool definitions.

```python
import pandas as pd
from MatplotLibAPI import PlotSpec, execute_plot

sales = pd.DataFrame(
    {
        "product": ["A", "B"],
        "revenue": [12, 15],
    }
)

spec = PlotSpec.from_dict(
    {
        "chart": "bar",
        "encoding": {
            "category": "product",
            "value": "revenue",
        },
        "options": {"stacked": False},
        "output": {"format": "png", "path": "charts/revenue.png"},
    }
)

result = execute_plot(spec, sales)
print(result.to_dict())
```

The executor validates chart names, parameters, referenced columns, local paths,
input dimensions, and output size before returning a `RenderResult`.

## Discovery and plugins

```python
from MatplotLibAPI import create_registry, openai_tool_definitions

registry = create_registry()
print(registry.context.list_plots())
print(registry.context.describe_plot("bar"))
print(openai_tool_definitions(registry=registry))
```

Each registered plot has one `PlotDescriptor` containing its callable,
parameter schema, backend, capabilities, aliases, examples, and supported output
formats. Legacy plugin API version 1 remains accepted; new plugins use version 2.

## Command line

```bash
matplotlibapi plots list
matplotlibapi plots describe bar
matplotlibapi schema plot-spec
matplotlibapi schema openai-tools
matplotlibapi inspect data.csv
matplotlibapi recommend data.csv
matplotlibapi validate plot.json
matplotlibapi render plot.json --data data.csv --output chart.png
matplotlibapi doctor
matplotlibapi test
matplotlibapi eval
matplotlibapi benchmark
```

All file operations are constrained to `--workspace` by default. Absolute paths
and workspace traversal are rejected unless an embedding application explicitly
uses a more permissive `RenderPolicy`.

## MCP

Start the optional stdio server with:

```bash
matplotlibapi-mcp
```

The MCP generic renderer and dedicated tools use the same canonical executor and
registry metadata. `describe_plot_modules` also returns plot descriptors and
OpenAI-compatible tool schemas.

## Stable package-root API

### Contracts and execution

- `PLOT_SPEC_SCHEMA_VERSION`
- `DataSource`
- `OutputSpec`
- `PlotSpec`
- `PlotValidationError`
- `ValidationIssue`
- `RenderPolicy`
- `RenderResult`
- `execute_plot`
- `validate_plot_request`
- `inspect_dataframe`
- `recommend_plot`
- `migrate_plot_spec`
- `openai_tool_definitions`

### Plugin surface

- `PLUGIN_API_VERSION`
- `Plugin`
- `PluginContext`
- `PluginRegistry`
- `PlotDescriptor`
- `CorePlotsPlugin`
- `create_registry`
- `infer_plot_descriptor`

### Plotting helpers

- `fplot_area`
- `fplot_bar`
- `fplot_box_violin`
- `fplot_correlation_matrix`
- `fplot_heatmap`
- `fplot_histogram_kde`
- `fplot_pie_donut`
- `fplot_sankey`
- `fplot_sunburst`
- `fplot_table`
- `fplot_timeserie`
- `fplot_timeseries`
- `fplot_treemap`
- `fplot_waffle`
- `fplot_wordcloud`

### Other types

- `CorrelationMethod`
- `DataFrameAccessor`

Specialized object APIs remain available from their modules:

```python
from MatplotLibAPI.bubble import Bubble
from MatplotLibAPI.network import NetworkGraph
from MatplotLibAPI.Pivot import plot_pivoted_bars
```

## Deterministic evaluations

The agent evaluation baseline requires no LLM, network access, API keys, or
credentials:

```bash
matplotlibapi eval
python scripts/benchmark_agent_plotting.py
```

It covers chart recommendations, invalid specification rejection, profiling,
schema discovery, and local performance budgets.

## Development

```bash
black --check src tests scripts
pydocstyle src scripts
pyright
pytest --cov=MatplotLibAPI --cov-report=term-missing
python -m build
python -m twine check dist/*
```

See `docs/PLOT_SPEC.md`, `docs/API_REFERENCE.md`, `docs/AGENT_EVALS.md`, and
`CONTRIBUTING.md` for the complete contracts.
