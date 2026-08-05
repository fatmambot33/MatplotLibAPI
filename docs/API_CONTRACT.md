# Public plotting API contract

This document defines the stable package-root plotting contract for MatplotLibAPI.

## Universal rules

Every plotting helper exported from `MatplotLibAPI.__all__` follows these rules:

1. Tabular input is named `pd_df` and accepts a pandas `DataFrame`.
2. Figure helpers are named `fplot_*` and return a Matplotlib `Figure`.
3. Existing keyword names remain supported within the current major version.
4. Shared concepts use the names below when they apply.

## Shared parameter vocabulary

| Concept | Canonical name | Meaning |
| --- | --- | --- |
| DataFrame | `pd_df` | Input pandas DataFrame. |
| Horizontal field | `x` | Column mapped to the horizontal axis. |
| Vertical field | `y` | Column mapped to the vertical axis. |
| Numeric measure | `value` | Column containing values to aggregate or display. |
| Category | `category` | Primary categorical grouping. |
| Secondary grouping | `group` | Series, color, or stack grouping. |
| Series label | `label` | Column identifying a plotted series. |
| Plot title | `title` | Human-readable figure title. |
| Figure size | `figsize` | Matplotlib figure size tuple. |

Chart-specific structural names such as `source`, `target`, `parents`, `labels`, `index`, and `columns` remain explicit because replacing them with generic aliases would reduce clarity.

## Package-root signature matrix

| Helper | Primary fields | Optional grouping | Return |
| --- | --- | --- | --- |
| `fplot_area` | `x`, `y` | `label` | `Figure` |
| `fplot_bar` | `category`, `value` | `group` | `Figure` |
| `fplot_box_violin` | `column` | `category` | `Figure` |
| `fplot_correlation_matrix` | numeric DataFrame columns | — | `Figure` |
| `fplot_heatmap` | `index`, `columns`, `values` | — | `Figure` |
| `fplot_histogram_kde` | `column` | — | `Figure` |
| `fplot_pie_donut` | `category`, `value` | — | `Figure` |
| `fplot_sankey` | `source`, `target`, `value` | — | `Figure` |
| `fplot_sunburst` | `labels`, `parents`, `values` | — | `Figure` |
| `fplot_table` | selected `cols` | — | `Figure` |
| `fplot_timeserie` | `x`, `y` | `label` | `Figure` |
| `fplot_treemap` | `labels`, `parents`, `values` | — | `Figure` |
| `fplot_waffle` | `category`, `value` | — | `Figure` |
| `fplot_wordcloud` | text column | optional weight column | `Figure` |

## Compatibility policy

Parameter renames are not performed silently. A future rename must first add a compatibility alias, emit a `DeprecationWarning`, document the replacement, and remain available until the next major release. Figure helpers must continue returning a `Figure`; callers that need an existing axes should use the corresponding module-level `aplot_*` API where available.
