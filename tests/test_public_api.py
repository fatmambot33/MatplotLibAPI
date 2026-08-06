"""Regression tests for the supported package-root API."""

import MatplotLibAPI


EXPECTED_PUBLIC_API = {
    "ColumnProfile",
    "ConformanceIssue",
    "ConformanceResult",
    "CorePlotsPlugin",
    "CorrelationMethod",
    "DataFrameAccessor",
    "DataProfile",
    "DataSource",
    "MigrationNotice",
    "OutputSpec",
    "PLOT_SPEC_SCHEMA_VERSION",
    "PLUGIN_API_VERSION",
    "PlotDescriptor",
    "PlotRecommendation",
    "PlotSpec",
    "PlotValidationError",
    "Plugin",
    "PluginContext",
    "PluginRegistry",
    "PresentationSpec",
    "RenderPolicy",
    "RenderResult",
    "RepairSuggestion",
    "V5_CANONICAL_CHARTS",
    "V5_REMOVAL_NOT_BEFORE",
    "ValidationIssue",
    "apply_repair_suggestions",
    "audit_plot_spec_for_v5",
    "create_registry",
    "execute_plot",
    "fplot_area",
    "fplot_bar",
    "fplot_box_violin",
    "fplot_correlation_matrix",
    "fplot_heatmap",
    "fplot_histogram_kde",
    "fplot_pie_donut",
    "fplot_sankey",
    "fplot_sunburst",
    "fplot_table",
    "fplot_timeserie",
    "fplot_timeseries",
    "fplot_treemap",
    "fplot_waffle",
    "fplot_wordcloud",
    "infer_plot_descriptor",
    "inspect_dataframe",
    "migrate_plot_spec",
    "migrate_plot_spec_for_v5",
    "openai_tool_definitions",
    "plugin_template_files",
    "profile_dataframe",
    "recommend_plot",
    "recommend_plots",
    "suggest_plot_spec_repairs",
    "validate_plot_request",
    "validate_plugin_conformance",
    "validate_registry_conformance",
    "v5_compatibility_status",
    "write_plugin_scaffold",
}


def test_public_api_is_explicit() -> None:
    """Expose exactly the documented package-root symbols."""
    assert set(MatplotLibAPI.__all__) == EXPECTED_PUBLIC_API


def test_public_api_symbols_are_importable() -> None:
    """Resolve every supported root export."""
    for name in EXPECTED_PUBLIC_API:
        assert getattr(MatplotLibAPI, name) is not None
