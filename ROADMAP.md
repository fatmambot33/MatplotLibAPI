# Roadmap

## Status: 4.1.0 complete

The first-class 4.1.0 roadmap is implemented. The release now has a stable
public API, deterministic visual checks, executable examples, validated API
documentation, dependency compatibility coverage, backward-compatibility
contracts, and clean installed-artifact smoke tests.

## Completed 4.1.0 scope

- stable and explicit package-root public API
- consistent plotting parameters and return contracts
- Python 3.9–3.12 quality matrix
- deterministic pixel-level checks for representative chart families
- executable headless gallery for every package-root plotting helper
- public API reference validated against `MatplotLibAPI.__all__`
- justified runtime dependency lower bounds
- Python 3.9 minimum-dependency compatibility job
- documented semantic-versioning and deprecation policy
- regression tests for supported 4.0.x module imports and aliases
- wheel and source-distribution validation in clean environments
- PEP 561 `py.typed` packaging support
- contribution, changelog, product, and release documentation

## Release gate

Release 4.1.0 may be published after the final GitHub Actions run passes and the
release notes are reviewed. Publication must not overwrite an existing tag or
PyPI version.

## After 4.1.0

Future work is managed as focused GitHub issues. New features must preserve the
public API contract, pass the complete quality gate, and support `PRODUCT.md`.
