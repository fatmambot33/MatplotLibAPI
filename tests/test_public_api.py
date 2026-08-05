"""Regression tests for the supported package-root API."""

import MatplotLibAPI


EXPECTED_PUBLIC_API = {
    "CorePlotsPlugin",
    "CorrelationMethod",
    "DataFrameAccessor",
    "PLUGIN_API_VERSION",
    "Plugin",
    "PluginContext",
    "PluginRegistry",
    "create_registry",
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
    "fplot_treemap",
    "fplot_waffle",
    "fplot_wordcloud",
}


def test_public_api_is_explicit() -> None:
    """Expose exactly the documented package-root symbols."""
    assert set(MatplotLibAPI.__all__) == EXPECTED_PUBLIC_API


def test_public_api_symbols_are_importable() -> None:
    """Resolve every supported root export."""
    for name in EXPECTED_PUBLIC_API:
        assert getattr(MatplotLibAPI, name) is not None
