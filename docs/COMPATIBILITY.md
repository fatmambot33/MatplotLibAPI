# Compatibility and deprecation policy

MatplotLibAPI follows semantic versioning.

## 4.0.x compatibility through the 4.x line

The package-root API in `MatplotLibAPI.__all__` is stable throughout the 4.x
release line. Documented 4.0.x module imports continue to work, including
`MatplotLibAPI.bar.fplot_bar` and `MatplotLibAPI.heatmap.fplot_heatmap`.
Compatibility aliases for histogram and pie remain available as
`fplot_histogram_kde` and `fplot_pie_donut`.

## Canonical time-series naming

`timeseries` and `fplot_timeseries` are canonical from 4.4 onward. The historic
`timeserie` chart name, module, MCP tool, and `fplot_timeserie` helper remain
available during the compatibility window. `matplotlibapi migrate` and
`audit_plot_spec_for_v5` identify affected specifications.

Breaking alias and plugin API v1 removals are forbidden before **2027-02-06**.
The executable `v5_compatibility_status` gate enforces this date. A later 5.0
release still requires explicit human approval and green migration evidence.

## Deprecations

A supported API is not removed in a minor release. A future deprecation must:

1. emit `DeprecationWarning` with the replacement API;
2. identify the earliest major release where removal may occur;
3. include regression coverage for both the old and replacement calls;
4. be recorded in the changelog and migration documentation;
5. satisfy the executable compatibility gate before removal.

Undocumented implementation helpers are not compatibility promises. New code
should prefer package-root imports listed in `docs/API_REFERENCE.md`.

## Plugin compatibility

Plugin API version 2 is canonical. Version 1 remains loadable through the 4.x
line. `PluginRegistry.compatibility_report()` identifies installed legacy
plugins. The official scaffold and conformance suite generate and validate only
version 2 plugins.

## Dependency support

Runtime dependencies have tested lower bounds in `pyproject.toml`. CI tests the
normal current dependency set and a Python 3.9 minimum-dependency environment.
Plotly static export tooling is isolated in the `plotly-export` extra because it
is not required for in-memory figures or JSON output. Upper bounds are added
only for demonstrated incompatibilities.
