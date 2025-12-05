"""Internal type aliases used across MatplotLibAPI."""

from typing import Callable, Literal, Union

from typing_extensions import TypeAlias

import pandas as pd


# ``DataFrame.corr`` supports the three built-in correlation methods or a callable
# that returns a pandas Series. Using a local alias avoids depending on the
# private ``pandas._typing`` module, which is not considered stable across
# releases.
CorrelationMethod: TypeAlias = Union[
    Literal["pearson", "kendall", "spearman"], Callable[[pd.Series, pd.Series], float]
]
