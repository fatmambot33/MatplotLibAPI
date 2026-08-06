"""Shared type aliases for MatplotLibAPI."""

import sys
from typing import Literal

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:  # pragma: no cover - Python 3.9 compatibility
    from typing_extensions import TypeAlias

CorrelationMethod: TypeAlias = Literal["pearson", "kendall", "spearman"]
