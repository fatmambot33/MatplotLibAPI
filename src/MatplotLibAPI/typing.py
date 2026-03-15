"""Internal type aliases used across MatplotLibAPI."""

from typing import Callable, Literal, Sequence, Union

from typing_extensions import TypeAlias


# ``DataFrame.corr`` supports the three built-in correlation methods or a callable
# that operates on two array-like inputs and returns a float. Using a local alias
# avoids depending on the private ``pandas._typing`` module, which is not
# considered stable across releases.
CorrelationMethod: TypeAlias = Union[
    Literal["pearson", "kendall", "spearman"],
    Callable[[Sequence[float], Sequence[float]], float],
]
