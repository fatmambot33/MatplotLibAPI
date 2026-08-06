# Data-aware intelligence

MatplotLibAPI performs profiling, recommendations, and repair suggestions
locally. No model, network, credential, or hosted service is required.

## Bounded profiles

`profile_dataframe` inspects a deterministic head sample, defaulting to 5,000
rows. It reports row and column counts, truncation, dtypes, missingness,
cardinality, representative values, numeric ranges and means, and conservative
semantic roles: numeric, datetime, boolean, categorical, text, or unknown.

```python
import pandas as pd
from MatplotLibAPI import profile_dataframe

frame = pd.DataFrame({"market": ["ES", "FR"], "revenue": [10, 20]})
profile = profile_dataframe(frame)
print(profile.to_dict())
```

## Explainable recommendations

`recommend_plots` returns ranked `PlotRecommendation` objects. Every result has
a score, explicit reasons, and optional warnings. `recommend_plot` preserves the
compact top-result API while adding alternatives and the complete profile.

## Safe repairs

`suggest_plot_spec_repairs` can canonicalize aliases, propose close column
matches, and identify unsupported parameters. It never mutates the input.
Applications must explicitly pass selected suggestions to
`apply_repair_suggestions`.

## Presentation presets

`PresentationSpec` supports high-contrast and colorblind palettes, grid control,
alt text, and number, integer, percent, currency, or compact semantic formats.
Preferences serialize inside the canonical PlotSpec and are applied by the
shared executor.
