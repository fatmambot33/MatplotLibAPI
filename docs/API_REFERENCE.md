# Public API reference

This page documents the stable package-root interface for MatplotLibAPI 4.2.x.
The authoritative export list is `MatplotLibAPI.__all__`; CI verifies that every
name in that list appears here and remains importable.

## Schema and execution contracts

- `PLOT_SPEC_SCHEMA_VERSION` — current portable plot-spec schema version.
- `DataSource` — inline table or local CSV source.
- `OutputSpec` — requested format, path, DPI, and transparency.
- `PlotSpec` — canonical serializable chart request.
- `ValidationIssue` — stable machine-readable validation item.
- `PlotValidationError` — exception containing validation issues.
- `RenderPolicy` — local workspace and resource limits.
- `RenderResult` — figure, artifact, warnings, and execution metadata.
- `execute_plot` — validated canonical rendering executor.
- `validate_plot_request` — validate parameters and column references.
- `inspect_dataframe` — deterministic local DataFrame profile.
- `recommend_plot` — deterministic chart recommendation.
- `migrate_plot_spec` — migrate supported older spec shapes.
- `openai_tool_definitions` — generate tools from registry descriptors.

## Types and pandas integration

- `CorrelationMethod` — accepted correlation methods.
- `DataFrameAccessor` — pandas accessor registered as `DataFrame.mpl`.

## Plugin surface

- `PLUGIN_API_VERSION` — current plugin contract version.
- `Plugin` — typed plugin protocol.
- `PluginContext` — callable and descriptor registration context.
- `PluginRegistry` — deterministic built-in and entry-point registry.
- `PlotDescriptor` — schema-rich plot capability contract.
- `CorePlotsPlugin` — built-in stable plotting plugin.
- `create_registry` — construct and optionally discover plugins.
- `infer_plot_descriptor` — derive a descriptor from a callable signature.

Plugin API version 2 adds descriptors and aliases. Version 1 plugins remain
accepted. Setup is atomic, and duplicate plugins, plots, or aliases are rejected.

## Matplotlib figure helpers

All helpers accept a pandas `DataFrame` as `pd_df` and return a new
`matplotlib.figure.Figure` unless explicitly noted otherwise.

- `fplot_area`
- `fplot_bar`
- `fplot_box_violin`
- `fplot_correlation_matrix`
- `fplot_heatmap`
- `fplot_histogram_kde`
- `fplot_pie_donut`
- `fplot_table`
- `fplot_timeserie`
- `fplot_timeseries` — correctly spelled compatibility alias.
- `fplot_waffle`
- `fplot_wordcloud`

## Plotly figure helpers

- `fplot_sankey`
- `fplot_sunburst`
- `fplot_treemap`

## Specialized module APIs

These remain supported at their module paths:

- `MatplotLibAPI.bubble.Bubble`
- `MatplotLibAPI.network.NetworkGraph`
- `MatplotLibAPI.Pivot.plot_pivoted_bars`

## Command-line and MCP interfaces

The `matplotlibapi` entry point exposes schema discovery, validation, rendering,
profiling, chart recommendation, diagnostics, evaluations, and benchmarks.
`matplotlibapi-mcp` exposes the same registry metadata and validated executor over
stdio when the optional MCP dependency is installed.
