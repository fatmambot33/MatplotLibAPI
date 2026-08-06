# Roadmap

## Status: 4.2.0 implementation complete; validation in progress

The 4.1.0 quality foundation is complete and published. The 4.2.0 milestone
turns MatplotLibAPI into one schema-driven local visualization engine for Python,
plugins, the CLI, MCP clients, Codex, and other agents.

## North star

> Make production-quality plotting equally simple, typed, predictable, and safe
> for people and agents.

## 4.2.0 workstreams

### 1. Canonical contracts — #101

- `PlotSpec`, `DataSource`, and `OutputSpec`
- deterministic JSON round trips
- strict JSON Schema export
- schema-version migration hooks
- `ValidationIssue`, `PlotValidationError`, and `RenderResult`

### 2. Registry and executor — #102

- schema-rich `PlotDescriptor`
- plugin API version 2 with version 1 compatibility
- deterministic aliases, capabilities, examples, and output formats
- OpenAI-compatible tool generation
- one validated `execute_plot` path

### 3. Safe interfaces — #103

- workspace, row, column, cell, input-byte, and output-byte policies
- `matplotlibapi` CLI
- MCP generic and dedicated tools backed by the executor
- schema-backed Codex discovery guidance
- no credentials or hosted dependencies

### 4. Quality and release — #104

- deterministic agent evaluations
- discovery and rendering benchmarks
- contract, executor, CLI, safety, compatibility, and MCP tests
- API reference, migration guidance, examples, and changelog
- installed wheel and source-distribution smoke tests

## 4.2.0 release gate

- Python 3.9–3.12 matrix passes.
- Black 25.11.0, pydocstyle, Pyright, pytest, and coverage pass.
- Documentation and examples pass headlessly.
- Plugin API version 1 compatibility remains tested.
- MCP and CLI use the canonical registry and schemas.
- Agent evaluations require no LLM or credentials.
- Package artifacts install and pass public API smoke tests.
- Release publication remains a separate governed action.

## Later milestones

### 4.3.0 — data-aware intelligence

- richer local data profiling
- deterministic chart recommendation explanations
- safe plot-spec repair suggestions
- accessibility and semantic formatting presets

### 4.4.0 — plugin ecosystem

- plugin template repository
- conformance and compatibility suite
- third-party descriptor documentation
- plugin packaging and discovery guidance

### 5.0.0 — intentional simplification

- remove aliases only after the documented deprecation window
- make `timeseries` canonical and retire `timeserie`
- finalize plugin API version 2
- separate heavyweight optional rendering backends where practical
