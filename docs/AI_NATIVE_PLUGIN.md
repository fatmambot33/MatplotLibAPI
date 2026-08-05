# AI-Native Plugin Guide

## Install from PyPI

```bash
pip install MatplotLibAPI
```

## Install from Git

```bash
pip install git+https://github.com/fatmambot33/MatplotLibAPI.git
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

The plugin exposes a typed, versioned plotting contract with deterministic entry-point discovery, capability metadata, and plugin contract tests. It does not require credentials. Optional local runtime settings may be stored in `.env`.
