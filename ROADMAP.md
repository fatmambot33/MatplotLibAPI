# Roadmap

## Status: 4.1.0 in progress

The public API, repository structure, typing, documentation baseline, and CI quality gate are complete. Release 4.1.0 remains open until the additional release-readiness work below is implemented and merged.

## Completed foundation

- stable and explicit package-root public API
- consistent plotting parameters and return contracts
- Python 3.9–3.12 support
- NumPy-style documentation and current README examples
- static typing with Pyright
- automated formatting, tests, coverage, and package validation
- contribution, changelog, product, and release documentation

## 4.1.0 release blockers

1. [#82 — Deterministic visual regression coverage](https://github.com/fatmambot33/MatplotLibAPI/issues/82)
   - protect representative Matplotlib output against unintended rendering drift
   - stabilize backend, fonts, DPI, figure size, and random inputs

2. [#83 — Executable example gallery](https://github.com/fatmambot33/MatplotLibAPI/issues/83)
   - provide one runnable example for every supported package-root plotting helper
   - execute examples headlessly in CI

3. [#84 — Generated and validated public API reference](https://github.com/fatmambot33/MatplotLibAPI/issues/84)
   - document every exported symbol and intentional module-level API
   - build documentation with warnings treated as errors

4. [#85 — Supported dependency ranges and compatibility tests](https://github.com/fatmambot33/MatplotLibAPI/issues/85)
   - define justified minimum dependency versions
   - test minimum and current compatible dependency sets

5. [#86 — Deprecation and backward-compatibility contracts](https://github.com/fatmambot33/MatplotLibAPI/issues/86)
   - preserve supported 4.0.x call patterns
   - test actionable warnings and migration paths

6. [#87 — Wheel installation and public API smoke tests](https://github.com/fatmambot33/MatplotLibAPI/issues/87)
   - test the built wheel and source distribution in clean environments
   - verify installed imports, representative plotting, metadata, contents, and the optional MCP entry point

## 4.1.0 definition of done

Release 4.1.0 may be published only when:

- issues #82–#87 are closed through reviewed changes
- all supported Python versions pass the quality gate
- minimum and current dependency environments pass
- documentation and examples build successfully
- visual regression tests pass on the designated CI environment
- wheel and source-distribution smoke tests pass from clean installations
- the changelog and migration guidance accurately describe the shipped behavior
- no release-blocking issue or pull request remains open

## After 4.1.0

Future work is managed as focused GitHub issues. New features must preserve the public API contract, pass the complete quality gate, and support `PRODUCT.md`.
