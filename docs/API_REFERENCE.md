# Public API reference

This page documents the stable package-root interface for MatplotLibAPI 4.1.x.
The authoritative export list is `MatplotLibAPI.__all__`; CI verifies that every
name in that list appears here and remains importable.

## Types and integration

### `CorrelationMethod`

Type alias describing accepted correlation methods.

### `DataFrameAccessor`

The pandas DataFrame accessor registered as `DataFrame.mpl`. The accessor
forwards to the same plotting implementations used by the functional API.

## Matplotlib figure helpers

All `fplot_*` helpers below accept a pandas `DataFrame` as `pd_df` and return a
new `matplotlib.figure.Figure` unless explicitly noted otherwise.

- `fplot_area` — area chart, including grouped and stacked forms.
- `fplot_bar` — categorical, grouped, or stacked bar chart.
- `fplot_box_violin` — box or violin distribution chart.
- `fplot_correlation_matrix` — numeric correlation matrix.
- `fplot_heatmap` — pivoted heatmap.
- `fplot_histogram_kde` — histogram with optional density curve.
- `fplot_pie_donut` — pie or donut chart.
- `fplot_table` — formatted Matplotlib table.
- `fplot_timeserie` — grouped or ungrouped time-series chart.
- `fplot_waffle` — waffle chart.
- `fplot_wordcloud` — weighted word cloud.

## Plotly figure helpers

These helpers return `plotly.graph_objects.Figure` objects.

- `fplot_sankey` — source/target flow diagram.
- `fplot_sunburst` — hierarchical sunburst chart.
- `fplot_treemap` — hierarchical treemap.

## Specialized module APIs

The following APIs remain supported at their module paths but are intentionally
not package-root exports:

- `MatplotLibAPI.bubble.Bubble`
- `MatplotLibAPI.network.NetworkGraph`
- `MatplotLibAPI.Pivot.plot_pivoted_bars`

They follow the same typing, documentation, and compatibility policy as the
package-root interface, but may evolve independently in minor releases.

## Errors and validation

Plot helpers validate required columns before rendering. Missing columns or
invalid parameter combinations raise a descriptive exception rather than
silently producing an incomplete chart. Additional Matplotlib, Seaborn, or
Plotly keyword arguments are forwarded where the documented helper accepts
`**kwargs`.

## Examples

Executable examples live in `examples/gallery.py`. Run them headlessly with:

```bash
MPLBACKEND=Agg python examples/gallery.py
```
