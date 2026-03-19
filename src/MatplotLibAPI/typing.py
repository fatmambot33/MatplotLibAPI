"""Internal type aliases used across MatplotLibAPI."""

from typing import Callable, Literal, Sequence, Union

from typing_extensions import TypeAlias

# ``DataFrame.corr`` supports the three built-in correlation methods or a callable
# that operates on two array-like inputs and returns a float.
CorrelationMethod: TypeAlias = Literal["pearson", "kendall", "spearman"]
