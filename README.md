# MatplotLibAPI

MatplotLibAPI is a Python library that simplifies the process of creating various types of plots from pandas DataFrames. It provides a high-level API for generating bubble charts, network graphs, pivot tables, tables, time series plots, and treemaps.

## Installation

To install the library, you can use pip:

```bash
pip install MatplotLibAPI
```

## Usage

Here's a simple example of how to create a bubble chart using MatplotLibAPI:

```python
import pandas as pd
import matplotlib.pyplot as plt
from MatplotLibAPI.Bubble import fplot_bubble

# Create a sample DataFrame
data = {
    'country': ['A', 'B', 'C', 'D'],
    'gdp_per_capita': [45000, 42000, 52000, 48000],
    'life_expectancy': [81, 78, 83, 82],
    'population': [10, 20, 5, 30]
}
df = pd.DataFrame(data)

# Generate the bubble chart
fig = fplot_bubble(df, label='country', x='gdp_per_capita', y='life_expectancy', z='population', title='Country Statistics')

# Display the plot
plt.show()
```

## Plot Types

The library supports the following plot types:

- **Bubble (Scatter plot)**
- **Network (Graph)**
- **Pivot**
- **Table**
- **Timeserie**
- **Treemap**
