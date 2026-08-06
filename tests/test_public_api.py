"""Regression tests for the supported package-root API."""

import MatplotLibAPI


EXPECTED_PUBLIC_API = {
    "CorePlotsPlugin",
    "CorrelationMethod",
    "DataFrameAccessor",
    "DataSource",
    "OutputSpec",
    "PLOT_SPEC_SCHEMA_VERSION",
    "PLUGIN_API_VERSION",
    "PlotDescriptor",
    "PlotSpec",
    "PlotValidationError",
    "Plugin",
    "PluginContext",
    "PluginRegistry",
    "RenderPolicy",
    "RenderResult",
    "ValidationIssue",
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
    "openai_tool_definitions",
    "recommend_plot",
    "validate_plot_request",
}


def test_public_api_is_explicit() -> None:
    """Expose exactly the documented package-root symbols."""
    assert set(MatplotLibAPI.__all__) == EXPECTED_PUBLIC_API


def test_public_api_symbols_are_importable() -> None:
    """Resolve every supported root export."""
    for name in EXPECTED_PUBLIC_API:
        assert getattr(MatplotLibAPI, name) is not None
