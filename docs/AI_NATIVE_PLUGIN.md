# AI-native plugin surface

MatplotLibAPI exposes one deterministic plotting contract across Python, the
CLI, Codex, and third-party plugins. It requires no credentials, remote
services, or hosted model calls.

## Discovery

The canonical registry publishes schema-rich `PlotDescriptor` objects and
OpenAI-compatible function tools. The same registry drives validation and
execution, preventing interface-specific chart definitions.

```python
from MatplotLibAPI import create_registry, openai_tool_definitions

registry = create_registry()
print(registry.context.list_descriptors())
print(openai_tool_definitions(registry=registry))
```

## Data intelligence

Agents should profile locally with `profile_dataframe`, review ranked results
from `recommend_plots`, and report the reasons behind a recommendation. Profiles
are bounded and deterministic. Repairs from `suggest_plot_spec_repairs` are
structured and opt-in; never silently mutate a user request.

## Plugin ecosystem

Plugin API version 2 is canonical. Use:

```bash
matplotlibapi plugins scaffold example-plugin ./example-plugin
matplotlibapi plugins conform
```

The official scaffold uses only public APIs. The conformance runner checks
plugin versions, descriptors, schemas, aliases, examples, output formats, and
generated tool names. Version 1 plugins remain loadable during the documented
4.x compatibility window and appear in compatibility diagnostics.

## Canonical naming and 5.0

Use `timeseries` as the canonical chart name. `timeserie`, `histogram`, and
`pie` remain compatibility aliases. Use `matplotlibapi migrate` and
`matplotlibapi compatibility` before a 5.0 transition. Breaking removals are
forbidden before 2027-02-06 and still require explicit human approval.
