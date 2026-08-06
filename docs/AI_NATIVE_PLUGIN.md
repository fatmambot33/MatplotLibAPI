# AI-native plugin guide

## Install

```bash
pip install MatplotLibAPI
```

For MCP:

```bash
pip install "MatplotLibAPI[mcp]"
```

For local development:

```bash
git clone https://github.com/fatmambot33/MatplotLibAPI.git
cd MatplotLibAPI
pip install -e '.[dev]'
```

## Install the Codex plugin

```bash
codex plugin marketplace add fatmambot33/MatplotLibAPI --ref main
codex plugin add matplotlibapi@fatmambot33-matplotlibapi
```

## Canonical discovery

The plugin should discover plots through the package registry rather than
hard-coding function names:

```python
from MatplotLibAPI import PlotSpec, create_registry, execute_plot

registry = create_registry()
print(registry.context.list_descriptors())
print(registry.context.openai_tools())
print(PlotSpec.json_schema())
```

Use `PlotSpec` and `execute_plot` for validated rendering. Use a restrictive
`RenderPolicy` whenever local paths are accepted. CLI and MCP interfaces derive
their discovery metadata from the same registry.

## Credential and data policy

MatplotLibAPI requires no credentials. Never request API keys, hosted
authentication, secret-store access, or remote data uploads for this plugin.
All plotting and profiling run locally against files or table records the user
has explicitly provided.
