# Roadmap

## Status: 4.3.0 and 4.4.0 implemented; 5.0 readiness implemented and gated

MatplotLibAPI 4.4.0 completes every non-breaking item in the published roadmap.
The intentional 5.0 removal step is prepared by executable migration tooling and
is blocked until the documented compatibility date and explicit approval.

## North star

> Make production-quality plotting equally simple, typed, predictable, safe,
> explainable, and extensible for people and agents.

## 4.3.0 — data-aware intelligence — #107, #108

- [x] bounded deterministic dataframe profiles
- [x] semantic column roles, missingness, cardinality, ranges, and examples
- [x] ranked chart recommendations with scores, reasons, and warnings
- [x] opt-in PlotSpec repair suggestions
- [x] high-contrast and colorblind accessibility presets
- [x] semantic number, integer, percent, currency, and compact formatting
- [x] Python, CLI, MCP, Codex, and schema parity

## 4.4.0 — plugin ecosystem — #109

- [x] official installable plugin scaffold
- [x] descriptor, alias, example, output, and OpenAI-tool conformance suite
- [x] structured machine-readable conformance results
- [x] CLI scaffold and conformance commands
- [x] third-party packaging and entry-point discovery guidance
- [x] plugin API version 1 compatibility diagnostics

## 5.0.0 — intentional simplification readiness — #110

- [x] `timeseries` is canonical across descriptors, recommendations, CLI, and MCP
- [x] `timeserie`, `histogram`, and `pie` migration diagnostics
- [x] explicit non-mutating PlotSpec migration helper and CLI
- [x] executable breaking-removal date gate
- [x] plugin API version 2 readiness diagnostics
- [x] Plotly static export dependencies isolated in `plotly-export`
- [ ] remove compatibility aliases and plugin API v1 support

The final unchecked removal is intentionally blocked until **2027-02-06**. It
requires a new major-release PR, explicit human approval, green migration
evidence, and no unresolved legacy-plugin findings. Skipping this gate would
violate the repository compatibility policy.

## Quality and release gate — #111

- [x] deterministic local evaluations cover intelligence and migration behavior
- [x] contract tests cover profiles, repairs, presets, conformance, and gating
- [x] strict public API documentation remains generated and checked
- [x] Python 3.9 syntax and compatibility are retained
- [x] wheel/sdist, SBOM, attestations, and release automation remain authoritative

## Program tracking

The complete execution program is tracked by #106. Implementation branch:
`agent/full-roadmap-4-3-to-5-0`.
