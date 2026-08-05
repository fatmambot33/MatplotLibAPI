# Compatibility and deprecation policy

MatplotLibAPI follows semantic versioning.

## 4.0.x compatibility in 4.1.x

The package-root API in `MatplotLibAPI.__all__` is stable for the 4.1 release
line. Documented 4.0.x module imports continue to work, including imports such
as `MatplotLibAPI.bar.fplot_bar` and `MatplotLibAPI.heatmap.fplot_heatmap`.
Compatibility aliases for histogram and pie helpers remain available from the
package root as `fplot_histogram_kde` and `fplot_pie_donut`.

## Deprecations

A supported API is not removed in a minor release. A future deprecation must:

1. emit `DeprecationWarning` with the replacement API;
2. identify the earliest major release where removal may occur;
3. include regression coverage for both the old and replacement calls;
4. be recorded in the changelog and migration documentation.

Undocumented implementation helpers are not compatibility promises. New code
should prefer the package-root imports listed in `docs/API_REFERENCE.md`.

## Dependency support

Runtime dependencies have tested lower bounds in `pyproject.toml`. CI tests the
normal current dependency set and a Python 3.9 minimum-dependency environment.
Upper bounds are added only for demonstrated incompatibilities.
