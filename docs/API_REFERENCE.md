# Public API reference

This page documents the stable package-root interface for MatplotLibAPI 4.4.x.
The authoritative export list is `MatplotLibAPI.__all__`; CI verifies that every
name in that list appears here and remains importable.

## Schema and execution contracts

- `PLOT_SPEC_SCHEMA_VERSION` — current portable plot-spec schema version.
- `DataSource` — inline table or local CSV source.
- `OutputSpec` — requested format, path, DPI, and transparency.
- `PresentationSpec` — accessibility, semantic number formatting, grid, and alt-text preferences.
- `PlotSpec` — canonical serializable chart request.
- `ValidationIssue` — stable machine-readable validation item.
- `PlotValidationError` — exception containing validation issues.
- `RenderPolicy` — local workspace and resource limits.
- `RenderResult` — figure, artifact, warnings, and execution metadata.
- `execute_plot` — validated canonical rendering executor.
- `validate_plot_request` — validate parameters and column references.
- `inspect_dataframe` — bounded deterministic DataFrame profile.
- `openai_tool_definitions` — generate tools from registry descriptors.
- `migrate_plot_spec` — migrate supported older schema shapes.

## Data intelligence

- `ColumnProfile` — semantic and statistical summary for one column.
- `DataProfile` — bounded deterministic profile for a DataFrame.
- `PlotRecommendation` — ranked chart recommendation with score, reasons, and warnings.
- `RepairSuggestion` — structured opt-in PlotSpec repair.
- `profile_dataframe` — build a bounded local profile.
- `recommend_plot` — return the highest-ranked chart plus alternatives and profile evidence.
- `recommend_plots` — return ranked explained recommendations.
- `suggest_plot_spec_repairs` — identify safe deterministic repairs without mutation.
- `apply_repair_suggestions` — explicitly apply selected repairs to a new PlotSpec.

## Plugin ecosystem

- `PLUGIN_API_VERSION` — current plugin contract version.
- `Plugin` — typed plugin protocol.
- `PluginContext` — callable and descriptor registration context.
- `PluginRegistry` — deterministic built-in and entry-point registry.
- `PlotDescriptor` — schema-rich plot capability contract.
- `CorePlotsPlugin` — built-in stable plotting plugin.
- `create_registry` — construct and optionally discover plugins.
- `infer_plot_descriptor` — derive a descriptor from a callable signature.
- `ConformanceIssue` — machine-readable plugin finding.
- `ConformanceResult` — plugin or registry conformance report.
- `validate_plugin_conformance` — validate one plugin in isolation.
- `validate_registry_conformance` — validate descriptors, aliases, examples, and tools.
- `plugin_template_files` — return the official plugin scaffold as files.
- `write_plugin_scaffold` — write the official plugin project template.

Plugin API version 2 is canonical. Version 1 plugins remain accepted during the
documented compatibility window and are surfaced in compatibility diagnostics.

## 5.0 migration and compatibility

- `V5_CANONICAL_CHARTS` — legacy-to-canonical chart mapping.
- `V5_REMOVAL_NOT_BEFORE` — earliest allowed date for breaking alias removal.
- `MigrationNotice` — structured migration diagnostic.
- `audit_plot_spec_for_v5` — identify legacy chart names affected by 5.0.
- `migrate_plot_spec_for_v5` — return a canonicalized PlotSpec.
- `v5_compatibility_status` — report the explicit breaking-removal gate.

`timeseries` is canonical in descriptors, CLI, MCP, and recommendations.
`timeserie` remains a compatibility alias until the 5.0 gate permits removal.

## Types and pandas integration

- `CorrelationMethod` — accepted correlation methods.
- `DataFrameAccessor` — pandas accessor registered as `DataFrame.mpl`.

## Matplotlib figure helpers

- `fplot_area`
- `fplot_bar`
- `fplot_box_violin`
- `fplot_correlation_matrix`
- `fplot_heatmap`
- `fplot_histogram_kde`
- `fplot_pie_donut`
- `fplot_table`
- `fplot_timeseries` — canonical time-series helper.
- `fplot_timeserie` — compatibility alias retained until the 5.0 gate.
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

The `matplotlibapi` entry point exposes discovery, validation, rendering,
profiling, explained recommendations, repair suggestions, presentation presets,
plugin scaffolding and conformance, migration diagnostics, compatibility gates,
evaluations, and benchmarks. `matplotlibapi-mcp` exposes the same intelligence
and canonical registry contract over stdio when MCP support is installed.
