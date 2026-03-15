"""Renderer registries for MCP plot-module dispatch."""

from __future__ import annotations

from typing import Any, Callable, Dict

from ..area import fplot_area
from ..bar import fplot_bar
from ..box_violin import fplot_box_violin
from ..heatmap import fplot_correlation_matrix, fplot_heatmap
from ..histogram import fplot_histogram
from ..pie import fplot_pie_donut
from ..sankey import fplot_sankey
from ..sunburst import fplot_sunburst
from ..table import fplot_table
from ..timeserie import fplot_timeserie
from ..treemap import fplot_treemap
from ..waffle import fplot_waffle
from ..word_cloud import fplot_wordcloud

Renderer = Callable[..., Any]

MATPLOTLIB_RENDERERS: Dict[str, Renderer] = {
    "bar": fplot_bar,
    "histogram": fplot_histogram,
    "box_violin": fplot_box_violin,
    "heatmap": fplot_heatmap,
    "correlation_matrix": fplot_correlation_matrix,
    "area": fplot_area,
    "pie": fplot_pie_donut,
    "waffle": fplot_waffle,
    "table": fplot_table,
    "timeserie": fplot_timeserie,
    "wordcloud": fplot_wordcloud,
}

PLOTLY_RENDERERS: Dict[str, Renderer] = {
    "sankey": fplot_sankey,
    "treemap": fplot_treemap,
    "sunburst": fplot_sunburst,
}

SUPPORTED_PLOT_MODULES = sorted(
    ["bubble", "network", *MATPLOTLIB_RENDERERS.keys(), *PLOTLY_RENDERERS.keys()]
)
