# MatplotLibAPI

MatplotLibAPI provides a small, high-level plotting API for pandas DataFrames. It covers common analytical charts, tables, network visualizations, hierarchical charts, and an optional MCP server for agent-driven rendering.

## Installation

```bash
pip install MatplotLibAPI
```

For MCP support:

```bash
pip install "MatplotLibAPI[mcp]"
```

## Quick start

Supported convenience functions are imported from the package root:

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

fig = fplot_bar(
    sales,
    category="product",
    value="revenue",
    group="region",
    stacked=True,
)
fig.show()
```

## Supported public API

The names below are the stable package-root import surface defined by `MatplotLibAPI.__all__`.

| Symbol | Purpose |
| --- | --- |
| `CorrelationMethod` | Supported correlation methods. |
| `DataFrameAccessor` | pandas DataFrame plotting accessor. |
| `fplot_area` | Area chart. |
| `fplot_bar` | Bar or stacked-bar chart. |
| `fplot_box_violin` | Box or violin chart. |
| `fplot_correlation_matrix` | Correlation matrix. |
| `fplot_heatmap` | Pivoted heatmap. |
| `fplot_histogram_kde` | Histogram with optional KDE. |
| `fplot_pie_donut` | Pie or donut chart. |
| `fplot_sankey` | Sankey diagram. |
| `fplot_sunburst` | Sunburst chart. |
| `fplot_table` | Matplotlib table. |
| `fplot_timeserie` | Time-series chart. |
| `fplot_treemap` | Treemap. |
| `fplot_waffle` | Waffle chart. |
| `fplot_wordcloud` | Word cloud. |

Use module-level imports only for specialized classes or legacy helpers that are not part of the package-root contract:

```python
from MatplotLibAPI.bubble import Bubble
from MatplotLibAPI.network import NetworkGraph
from MatplotLibAPI.Pivot import plot_pivoted_bars
```

## Examples

### Area

```python
import pandas as pd
from MatplotLibAPI import fplot_area

frame = pd.DataFrame(
    {
        "quarter": ["Q1", "Q2", "Q1", "Q2"],
        "segment": ["Free", "Free", "Pro", "Pro"],
        "subscriptions": [80, 95, 30, 45],
    }
)
fig = fplot_area(frame, x="quarter", y="subscriptions", label="segment", stacked=True)
```

### Histogram and KDE

```python
import pandas as pd
from MatplotLibAPI import fplot_histogram_kde

frame = pd.DataFrame({"waiting_time_minutes": [3, 5, 5, 8, 10, 13]})
fig = fplot_histogram_kde(
    frame,
    column="waiting_time_minutes",
    bins=6,
    kde=True,
)
```

### Box or violin

```python
import pandas as pd
from MatplotLibAPI import fplot_box_violin

frame = pd.DataFrame(
    {
        "department": ["Sales", "Sales", "Product", "Product"],
        "score": [7.2, 8.1, 8.8, 9.0],
    }
)
fig = fplot_box_violin(
    frame,
    column="score",
    category="department",
    use_violin=True,
)
```

### Heatmap and correlation matrix

```python
import pandas as pd
from MatplotLibAPI import fplot_correlation_matrix, fplot_heatmap

engagement = pd.DataFrame(
    {
        "month": ["Jan", "Jan", "Feb", "Feb"],
        "channel": ["Email", "Social", "Email", "Social"],
        "engagements": [120, 200, 150, 230],
    }
)
metrics = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 5]})

heatmap = fplot_heatmap(
    engagement,
    index="month",
    columns="channel",
    values="engagements",
)
correlation = fplot_correlation_matrix(metrics)
```

### Pie, waffle, and Sankey

```python
import pandas as pd
from MatplotLibAPI import fplot_pie_donut, fplot_sankey, fplot_waffle

shares = pd.DataFrame(
    {"device": ["Desktop", "Mobile"], "sessions": [40, 60]}
)
flows = pd.DataFrame(
    {"source": ["Visit", "Visit"], "target": ["Buy", "Leave"], "value": [35, 65]}
)

pie = fplot_pie_donut(shares, category="device", value="sessions", donut=True)
waffle = fplot_waffle(shares, category="device", value="sessions")
sankey = fplot_sankey(flows, source="source", target="target", value="value")
```

### Table and time series

```python
import pandas as pd
from MatplotLibAPI import fplot_table, fplot_timeserie

frame = pd.DataFrame(
    {
        "date": ["2026-01-01", "2026-02-01"],
        "group": ["A", "A"],
        "value": [10, 14],
    }
)

table = fplot_table(pd_df=frame, cols=["date", "value"])
series = fplot_timeserie(pd_df=frame, label="group", x="date", y="value")
```

### Treemap, sunburst, and word cloud

```python
import pandas as pd
from MatplotLibAPI import fplot_sunburst, fplot_treemap, fplot_wordcloud

hierarchy = pd.DataFrame(
    {
        "labels": ["All", "A", "B"],
        "parents": ["", "All", "All"],
        "values": [30, 10, 20],
    }
)
words = pd.DataFrame({"word": ["simple", "reliable"], "weight": [5, 3]})

sunburst = fplot_sunburst(
    hierarchy,
    labels="labels",
    parents="parents",
    values="values",
)
treemap = fplot_treemap(
    pd_df=pd.DataFrame({"path": ["A", "B"], "values": [10, 20]}),
    path="path",
    values="values",
)
cloud = fplot_wordcloud(words, text_column="word", weight_column="weight")
```

### Specialized object APIs

`Bubble`, `NetworkGraph`, and the pivot helpers remain intentionally module-level APIs:

```python
import pandas as pd
from MatplotLibAPI.bubble import Bubble
from MatplotLibAPI.network import NetworkGraph
from MatplotLibAPI.Pivot import plot_pivoted_bars

bubble_data = pd.DataFrame(
    {"country": ["A", "B"], "gdp": [10, 20], "life": [70, 80], "population": [5, 8]}
)
bubble = Bubble(
    pd_df=bubble_data,
    label="country",
    x="gdp",
    y="life",
    z="population",
).fplot_w()

edges = pd.DataFrame(
    {"source": ["A", "B"], "target": ["B", "C"], "weight": [1, 2]}
)
network = NetworkGraph.from_pandas_edgelist(
    edges,
    source="source",
    target="target",
    edge_weight_col="weight",
).fplot_w(edge_weight_col="weight")

pivot_data = pd.DataFrame(
    {"category": ["A", "B"], "date": ["2026-01", "2026-01"], "value": [1, 2]}
)
axes = plot_pivoted_bars(
    data=pivot_data,
    label="category",
    x="date",
    y="value",
)
```

## Import migration

Prefer package-root imports for every symbol in the public API table:

```python
# Before
from MatplotLibAPI.bar import fplot_bar
from MatplotLibAPI.heatmap import fplot_heatmap

# Supported public imports
from MatplotLibAPI import fplot_bar, fplot_heatmap
```

Existing module imports continue to work, but package-root imports are the documented stable contract. Specialized APIs that are not exported in `MatplotLibAPI.__all__` should continue to use their module paths.

## Sample data

Generate the repository sample CSV files with:

```bash
python scripts/generate_sample_data.py
```

## MCP integration

Start the optional stdio server with:

```bash
matplotlibapi-mcp
```

The server exposes dedicated plotting tools, a generic `plot_module` tool, and `describe_plot_modules` for capability discovery. Rendering tools accept either a CSV path or in-memory table records and return PNG bytes.

## Development

Run the repository quality checks before opening a pull request:

```bash
black --check src tests
pydocstyle src
pyright
pytest --cov=MatplotLibAPI --cov-report=term-missing
python -m build
python -m twine check dist/*
```
