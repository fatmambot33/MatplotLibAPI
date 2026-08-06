# MatplotLibAPI

MatplotLibAPI is a typed, local-first plotting engine for pandas users, Python
applications, plugins, CLIs, and AI agents. It provides a small stable plotting
API plus one schema-driven execution contract that can be discovered and used
without credentials or hosted services.

## Installation

```bash
pip install MatplotLibAPI
```

Install optional MCP and Plotly static-export support with:

```bash
pip install "MatplotLibAPI[mcp]"
pip install "MatplotLibAPI[plotly-export]"
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
        "presentation": {
            "accessibility": "colorblind",
            "number_format": "currency",
            "currency": "EUR",
            "alt_text": "Revenue by product.",
        },
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
formats. Plugin API version 2 is canonical; version 1 remains accepted during the
documented 4.x compatibility window.

Create and validate a third-party plugin with:

```bash
matplotlibapi plugins scaffold example-plugin ./example-plugin
matplotlibapi plugins conform
```

## Command line

```bash
matplotlibapi plots list
matplotlibapi plots describe bar
matplotlibapi schema plot-spec
matplotlibapi schema openai-tools
matplotlibapi inspect data.csv
matplotlibapi recommend data.csv
matplotlibapi repair plot.json --data data.csv
matplotlibapi presets list
matplotlibapi migrate plot.json
matplotlibapi compatibility
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
registry metadata. MCP also exposes bounded profiling, ranked recommendations,
repair suggestions, and compatibility status. `describe_plot_modules` returns
plot descriptors and OpenAI-compatible tool schemas.

## Data-aware intelligence

```python
import pandas as pd
from MatplotLibAPI import PlotSpec, profile_dataframe, recommend_plots

frame = pd.DataFrame(
    {
        "date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
        "revenue": [100, 125],
    }
)
profile = profile_dataframe(frame)
recommendations = recommend_plots(profile)
spec = PlotSpec.from_dict(
    {
        "chart": recommendations[0].chart,
        "encoding": recommendations[0].encoding,
    }
)
```

Profiles are bounded and deterministic. Recommendations include scores, reasons,
and warnings. Repair suggestions are opt-in and never mutate the source spec.

## Canonical time-series API and 5.0 preparation

`timeseries` and `fplot_timeseries` are canonical. The historic `timeserie`
spelling remains available until the executable compatibility gate permits a
5.0 removal after 2027-02-06. Use `matplotlibapi migrate` before that boundary.

## Stable package-root API

### Contracts and execution

- `PLOT_SPEC_SCHEMA_VERSION`
- `DataSource`
- `OutputSpec`
- `PresentationSpec`
- `PlotSpec`
- `PlotValidationError`
- `ValidationIssue`
- `RenderPolicy`
- `RenderResult`
- `execute_plot`
- `validate_plot_request`
- `inspect_dataframe`
- `recommend_plot`
- `profile_dataframe`
- `recommend_plots`
- `suggest_plot_spec_repairs`
- `apply_repair_suggestions`
- `migrate_plot_spec`
- `migrate_plot_spec_for_v5`
- `v5_compatibility_status`
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
- `validate_plugin_conformance`
- `validate_registry_conformance`
- `write_plugin_scaffold`

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
- `fplot_timeseries` (canonical)
- `fplot_timeserie` (compatibility alias)
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

It covers explained chart recommendations, invalid specification rejection,
bounded profiling, opt-in repair suggestions, plugin conformance, 5.0 migration
gates, schema discovery, and local performance budgets.

## Development

```bash
black --check src tests scripts
pydocstyle src scripts
pyright
pytest --cov=MatplotLibAPI --cov-report=term-missing
python -m build
python -m twine check dist/*
```

See `docs/PLOT_SPEC.md`, `docs/DATA_INTELLIGENCE.md`,
`docs/PLUGIN_ECOSYSTEM.md`, `docs/MIGRATING_TO_5.md`,
`docs/API_REFERENCE.md`, and `CONTRIBUTING.md` for the complete contracts.
