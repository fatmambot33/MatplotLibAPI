"""Public API and pandas accessor for MatplotLibAPI."""

from .Area import aplot_area, fplot_area
from .Bar import aplot_bar, fplot_bar
from .BoxViolin import aplot_box_violin, fplot_box_violin
from .Bubble import BUBBLE_STYLE_TEMPLATE, aplot_bubble, fplot_bubble
from .Composite import (
    plot_composite_bubble,
    plot_composite_treemap,
    plot_wordcloud_network,
)
from .Heatmap import (
    HEATMAP_STYLE_TEMPLATE,
    aplot_correlation_matrix,
    aplot_heatmap,
    fplot_correlation_matrix,
    fplot_heatmap,
)
from .Histogram import aplot_histogram_kde, fplot_histogram_kde
from .Network import (
    NETWORK_STYLE_TEMPLATE,
    aplot_network,
    aplot_network_node,
    aplot_network_components,
    fplot_network_node,
    fplot_network_components,
    fplot_network,
)
from .Pie import aplot_pie_donut, fplot_pie_donut
from .Sankey import fplot_sankey
from .StyleTemplate import (
    AREA_STYLE_TEMPLATE,
    DISTRIBUTION_STYLE_TEMPLATE,
    PIE_STYLE_TEMPLATE,
    SANKEY_STYLE_TEMPLATE,
    StyleTemplate,
)
from .Table import TABLE_STYLE_TEMPLATE, aplot_table, fplot_table
from .Timeserie import TIMESERIE_STYLE_TEMPLATE, aplot_timeserie, fplot_timeserie
from .Sunburst import fplot_sunburst
from .Treemap import TREEMAP_STYLE_TEMPLATE, fplot_treemap
from .Waffle import aplot_waffle, fplot_waffle
from .Wordcloud import (
    WORDCLOUD_STYLE_TEMPLATE,
    aplot_wordcloud,
    fplot_wordcloud,
)
from .accessor import DataFrameAccessor

__all__ = [
    "DataFrameAccessor",
    "StyleTemplate",
    "aplot_bubble",
    "aplot_network",
    "aplot_network_node",
    "aplot_network_components",
    "aplot_table",
    "aplot_timeserie",
    "aplot_wordcloud",
    "aplot_bar",
    "aplot_histogram_kde",
    "aplot_box_violin",
    "aplot_heatmap",
    "aplot_correlation_matrix",
    "aplot_area",
    "aplot_pie_donut",
    "aplot_waffle",
    "fplot_bubble",
    "fplot_network",
    "fplot_network_node",
    "fplot_network_components",
    "fplot_table",
    "fplot_timeserie",
    "fplot_wordcloud",
    "fplot_treemap",
    "fplot_sunburst",
    "fplot_bar",
    "fplot_histogram_kde",
    "fplot_box_violin",
    "fplot_heatmap",
    "fplot_correlation_matrix",
    "fplot_area",
    "fplot_pie_donut",
    "fplot_waffle",
    "fplot_sankey",
    "plot_composite_bubble",
    "plot_composite_treemap",
    "plot_wordcloud_network",
]
