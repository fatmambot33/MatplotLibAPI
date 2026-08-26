# Plugin ecosystem

MatplotLibAPI plugin API version 2 is schema-first and deterministic. A plugin
registers callables through `PluginContext`; descriptors, OpenAI tools, CLI,
validation, and documentation all derive from the same registration.

## Scaffold

```bash
matplotlibapi plugins scaffold example-plugin ./example-plugin
```

The command creates a `pyproject.toml`, package, entry point, example plot,
README, and conformance test. It refuses to overwrite a non-empty destination
unless `--force` is supplied.

## Conformance

```bash
matplotlibapi plugins conform
```

`validate_plugin_conformance` and `validate_registry_conformance` verify strict
parameter schemas, canonical names, alias targets, figure output, examples, and
unique generated tool names. Results are structured and machine-readable.

## Packaging and discovery

Third-party packages declare the `matplotlibapi.plugins` entry-point group. The
entry-point value must construct a plugin with `name`, `api_version = "2"`, and
a `setup(context)` method. Discovery is sorted by entry-point name and duplicate
registration is atomic.

## Compatibility

Version 1 plugins remain accepted during the 4.x compatibility window. They are
reported by `PluginRegistry.compatibility_report()` and must migrate before the
5.0 removal gate can be satisfied.
