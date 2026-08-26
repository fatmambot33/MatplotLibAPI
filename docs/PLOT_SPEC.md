# PlotSpec contract

`PlotSpec` is the canonical request shared by Python, CLI, Codex, plugins, and
generated OpenAI tool definitions. Schema version `1.0` remains backward
compatible in MatplotLibAPI 4.4.

```json
{
  "schema_version": "1.0",
  "chart": "timeseries",
  "encoding": {"x": "date", "y": "revenue"},
  "options": {},
  "presentation": {
    "accessibility": "colorblind",
    "number_format": "currency",
    "currency": "EUR",
    "alt_text": "Monthly revenue trend.",
    "show_grid": true
  },
  "output": {"format": "png", "dpi": 150, "transparent": false}
}
```

## Presentation

`PresentationSpec` accepts accessibility presets `default`, `high-contrast`, and
`colorblind`. Number formats are `auto`, `number`, `integer`, `percent`,
`currency`, and `compact`. These values are part of the strict JSON Schema and
survive deterministic round trips.

## Migration

The legacy compact `params` object still migrates to `options`. The 5.0
migration helpers additionally canonicalize chart aliases without changing the
schema version or mutating the source object.
