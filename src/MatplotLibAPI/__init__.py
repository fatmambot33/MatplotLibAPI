"""Supported public API for MatplotLibAPI.

The names exported here form the stable package-root import surface. Internal
helpers and implementation classes should be imported from their modules.
"""

from .accessor import DataFrameAccessor
from .area import fplot_area
from .bar import fplot_bar
from .box_violin import fplot_box_violin
from .conformance import (
    ConformanceIssue,
    ConformanceResult,
    plugin_template_files,
    validate_plugin_conformance,
    validate_registry_conformance,
    write_plugin_scaffold,
)
from .executor import (
    RenderPolicy,
    execute_plot,
    inspect_dataframe,
    openai_tool_definitions,
    recommend_plot,
    validate_plot_request,
)
from .heatmap import fplot_correlation_matrix, fplot_heatmap
from .intelligence import (
    ColumnProfile,
    DataProfile,
    PlotRecommendation,
    RepairSuggestion,
    apply_repair_suggestions,
    profile_dataframe,
    recommend_plots,
    suggest_plot_spec_repairs,
)
from .migration import (
    V5_CANONICAL_CHARTS,
    V5_REMOVAL_NOT_BEFORE,
    MigrationNotice,
    audit_plot_spec_for_v5,
    migrate_plot_spec_for_v5,
    v5_compatibility_status,
)
from .histogram import fplot_histogram as fplot_histogram_kde
from .pie import fplot_pie as fplot_pie_donut
from .plugins import (
    PLUGIN_API_VERSION,
    CorePlotsPlugin,
    PlotDescriptor,
    Plugin,
    PluginContext,
    PluginRegistry,
    create_registry,
    infer_plot_descriptor,
)
from .sankey import fplot_sankey
from .specs import (
    PLOT_SPEC_SCHEMA_VERSION,
    DataSource,
    OutputSpec,
    PresentationSpec,
    PlotSpec,
    PlotValidationError,
    RenderResult,
    ValidationIssue,
    migrate_plot_spec,
)
from .sunburst import fplot_sunburst
from .table import fplot_table
from .timeseries import fplot_timeserie, fplot_timeseries
from .treemap import fplot_treemap
from .types import CorrelationMethod
from .waffle import fplot_waffle
from .word_cloud import fplot_wordcloud


__all__ = [
    "PLOT_SPEC_SCHEMA_VERSION",
    "V5_CANONICAL_CHARTS",
    "V5_REMOVAL_NOT_BEFORE",
    "ColumnProfile",
    "ConformanceIssue",
    "ConformanceResult",
    "DataProfile",
    "MigrationNotice",
    "PlotRecommendation",
    "RepairSuggestion",
    "PLUGIN_API_VERSION",
    "CorePlotsPlugin",
    "CorrelationMethod",
    "DataFrameAccessor",
    "DataSource",
    "OutputSpec",
    "PresentationSpec",
    "PlotDescriptor",
    "PlotSpec",
    "PlotValidationError",
    "Plugin",
    "PluginContext",
    "PluginRegistry",
    "RenderPolicy",
    "RenderResult",
    "ValidationIssue",
    "apply_repair_suggestions",
    "audit_plot_spec_for_v5",
    "create_registry",
    "execute_plot",
    "infer_plot_descriptor",
    "inspect_dataframe",
    "migrate_plot_spec",
    "migrate_plot_spec_for_v5",
    "plugin_template_files",
    "profile_dataframe",
    "recommend_plots",
    "openai_tool_definitions",
    "recommend_plot",
    "suggest_plot_spec_repairs",
    "validate_plot_request",
    "validate_plugin_conformance",
    "validate_registry_conformance",
    "v5_compatibility_status",
    "write_plugin_scaffold",
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
]
