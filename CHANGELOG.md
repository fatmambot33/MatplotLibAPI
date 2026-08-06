# Changelog

All notable changes to this project are documented here.

## Unreleased

### Added

- Canonical `PlotSpec`, `DataSource`, `OutputSpec`, `RenderResult`, and
  machine-readable validation contracts.
- Strict JSON Schema export and version-1 migration support.
- Schema-rich `PlotDescriptor` discovery with capabilities, aliases, examples,
  output formats, and OpenAI-compatible tool generation.
- One validated `execute_plot` path for Python, CLI, and MCP integrations.
- Local `RenderPolicy` limits for workspace paths, rows, columns, cells, input
  bytes, and output bytes.
- `matplotlibapi` CLI for discovery, schemas, validation, rendering, profiling,
  recommendations, diagnostics, self-tests, evaluations, and benchmarks.
- Deterministic agent evaluation and discovery benchmark commands that require
  no LLM, network access, or credentials.
- Correctly spelled `fplot_timeseries` alias.

### Changed

- Advanced the plugin contract to version 2 while accepting version 1 plugins.
- Refactored MCP generic and dedicated module rendering onto the canonical
  executor and exposed registry descriptors through MCP metadata.
- Expanded the public API reference and roadmap for the 4.2 milestone.

### Security

- CLI and agent file operations are workspace constrained by default.
- Absolute paths and workspace traversal are rejected unless an embedding
  application explicitly enables them.

## 4.1.0 - 2026-08-05

### Added

- Explicit, tested package-root public plotting API.
- Stable plotting parameter and return contracts.
- Deterministic pixel-level regression coverage for representative chart families.
- Executable, headless example gallery covering every public plotting helper.
- Public API reference validated against `MatplotLibAPI.__all__`.
- Semantic-versioning, compatibility, and deprecation policy.
- PEP 561 `py.typed` marker for downstream type checkers.
- Clean-environment wheel and source-distribution smoke tests.

### Changed

- Defined justified lower bounds for all runtime dependencies.
- Added Python 3.9 minimum-dependency compatibility testing.
- Strengthened CI across Python 3.9–3.12 with formatting, docstring, typing,
  documentation, examples, tests, coverage, package, and artifact validation.
- Aligned README examples and imports with the supported public API.
- Standardized repository documentation and release expectations.

### Compatibility

- Preserved documented 4.0.x module imports and compatibility aliases.
- Added contract tests tying legacy module imports to package-root exports.
- Future removals require actionable `DeprecationWarning` messages and a major
  release boundary.

### Fixed

- Removed an incompatible redundant pytest docstring plugin.
- Preserved backward-compatible plotting keyword behavior while locking the
  supported API.
