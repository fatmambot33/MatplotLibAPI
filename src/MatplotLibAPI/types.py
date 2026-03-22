"""Shared type aliases for MatplotLibAPI."""

from typing import Literal
from typing_extensions import TypeAlias

CorrelationMethod: TypeAlias = Literal["pearson", "kendall", "spearman"]
