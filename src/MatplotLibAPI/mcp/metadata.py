"""Discoverability metadata used by MCP plot tools."""

from typing import Dict

SHARED_INPUT_CONTRACT: Dict[str, str] = {
    "csv_path": "Path to a CSV file with source data.",
    "table": "In-memory table as list[dict[str, Any]] records.",
    "return": "PNG bytes suitable for application/octet-stream.",
}

PLOT_MODULE_PARAMETER_HINTS: Dict[str, Dict[str, str]] = {
    "bubble": {
        "label": "Bubble label column name.",
        "x": "X-axis column name.",
        "y": "Y-axis column name.",
        "z": "Bubble size column name.",
    },
    "network": {
        "edge_source_col": "Source-node column name (default: source).",
        "edge_target_col": "Target-node column name (default: target).",
        "edge_weight_col": "Edge-weight column name (default: weight).",
    },
    "bar": {
        "category": "Category column name.",
        "value": "Numeric value column name.",
    },
    "histogram": {"value": "Numeric value column name."},
    "box_violin": {"y": "Numeric value column name."},
    "heatmap": {
        "x": "X-axis category column name.",
        "y": "Y-axis category column name.",
        "value": "Cell value column name.",
    },
    "correlation_matrix": {
        "features": "List of numeric columns used in correlation matrix.",
    },
    "area": {
        "x": "X-axis column name.",
        "y": "List of stacked y-axis columns.",
    },
    "pie": {
        "label": "Category label column name.",
        "value": "Numeric value column name.",
    },
    "waffle": {
        "label": "Category label column name.",
        "value": "Numeric value column name.",
    },
    "sankey": {
        "source": "Source-node column name.",
        "target": "Target-node column name.",
        "value": "Flow/weight column name.",
    },
    "table": {},
    "timeserie": {
        "x": "Datetime or ordered x-axis column name.",
        "y": "List of y-axis series columns.",
    },
    "wordcloud": {
        "text_column": "Text/token column name.",
        "weight_column": "Weight/frequency column name (optional).",
    },
    "treemap": {
        "path": "Hierarchy column names in order.",
        "values": "Numeric value column name.",
    },
    "sunburst": {
        "path": "Hierarchy column names in order.",
        "values": "Numeric value column name.",
    },
}

DEDICATED_PLOT_TOOLS: Dict[str, str] = {
    "plot_bubble": "bubble",
    "plot_network": "network",
    "plot_bar": "bar",
    "plot_histogram": "histogram",
    "plot_box_violin": "box_violin",
    "plot_heatmap": "heatmap",
    "plot_correlation_matrix": "correlation_matrix",
    "plot_area": "area",
    "plot_pie": "pie",
    "plot_waffle": "waffle",
    "plot_sankey": "sankey",
    "plot_table": "table",
    "plot_timeserie": "timeserie",
    "plot_wordcloud": "wordcloud",
    "plot_treemap": "treemap",
    "plot_sunburst": "sunburst",
}
