"""Supported public API for MatplotLibAPI.

The names exported here form the stable package-root import surface. Internal
helpers and implementation classes should be imported from their modules.
"""

from .accessor import DataFrameAccessor
from .area import fplot_area
from .bar import fplot_bar
from .box_violin import fplot_box_violin
from .heatmap import fplot_correlation_matrix, fplot_heatmap
from .histogram import fplot_histogram as fplot_histogram_kde
from .pie import fplot_pie as fplot_pie_donut
from .plugins import (
    PLUGIN_API_VERSION,
    CorePlotsPlugin,
    Plugin,
    PluginContext,
    PluginRegistry,
    create_registry,
)
from .sankey import fplot_sankey
from .sunburst import fplot_sunburst
from .table import fplot_table
from .timeserie import fplot_timeserie
from .treemap import fplot_treemap
from .types import CorrelationMethod
from .waffle import fplot_waffle
from .word_cloud import fplot_wordcloud

__all__ = [
    "PLUGIN_API_VERSION",
    "CorePlotsPlugin",
    "CorrelationMethod",
    "DataFrameAccessor",
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
]
