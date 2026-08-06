# Preparing for MatplotLibAPI 5.0

The 5.0 program intentionally removes ambiguity, but breaking removals are
gated until **2027-02-06** and require explicit approval after CI and migration
evidence are green.

## Canonical chart names

| Legacy | Canonical |
| --- | --- |
| `timeserie` | `timeseries` |
| `histogram` | `histogram_kde` |
| `pie` | `pie_donut` |

Audit and migrate a specification with:

```bash
matplotlibapi migrate plot.json
matplotlibapi migrate plot.json --write migrated.json
matplotlibapi compatibility
```

Python applications can use `audit_plot_spec_for_v5`,
`migrate_plot_spec_for_v5`, and `v5_compatibility_status`.

## Plugin API

New plugins must use plugin API version 2. Version 1 remains loadable through
4.x, appears in compatibility reports, and prevents the plugin-removal readiness
flag from becoming true.

## Optional export backend

Static Plotly export dependencies are installed with:

```bash
pip install "MatplotLibAPI[plotly-export]"
```

Core in-memory plotting and JSON output do not require that extra.
