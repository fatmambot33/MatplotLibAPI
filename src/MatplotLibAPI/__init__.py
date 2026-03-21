"""Public API and pandas accessor for MatplotLibAPI."""

from typing_extensions import TypeAlias
from typing import Literal
from .accessor import DataFrameAccessor

CorrelationMethod: TypeAlias = Literal["pearson", "kendall", "spearman"]

__all__ = ["DataFrameAccessor"]
