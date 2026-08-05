# Changelog

All notable changes to this project are documented here.

## Unreleased

### Planned for 4.1.0

- Deterministic visual regression coverage for representative chart families.
- An executable, CI-validated example gallery covering the supported package-root plotting API.
- A generated public API reference validated with documentation warnings treated as errors.
- Defined minimum dependency versions with minimum and current compatibility testing.
- Explicit deprecation and backward-compatibility contract coverage for supported 4.0.x usage.
- Clean-environment wheel and source-distribution smoke tests, including public imports and representative plotting.

### Added

- Explicit, tested package-root public plotting API.
- Stable plotting parameter and return contracts.
- Regression coverage for documented imports and signatures.
- Product, roadmap, contribution, and release documentation.

### Changed

- Aligned README examples with the supported public API.
- Strengthened CI across Python 3.9–3.12 with formatting, docstring, typing, tests, coverage, and package validation.
- Standardized repository documentation and release expectations.

### Fixed

- Removed an incompatible redundant pytest docstring plugin.
- Preserved backward-compatible plotting keyword behavior while locking the supported API.
