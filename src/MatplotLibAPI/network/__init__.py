"""Network plotting tools for graph-based visualizations.

This package preserves the public ``MatplotLibAPI.network`` API while
organizing implementation details in submodules.
"""

from .constants import _DEFAULT, _WEIGHT_PERCENTILES
from .core import NETWORK_STYLE_TEMPLATE, NetworkGraph
from .plot import (
    aplot_network,
    aplot_network_node,
    aplot_network_components,
    fplot_network,
    fplot_network_node,
    fplot_network_components,
)
from .scaling import _scale_weights, _softmax

__all__ = [
    "aplot_network",
    "aplot_network_node",
    "aplot_network_components",
    "fplot_network",
    "fplot_network_node",
    "fplot_network_components",
    "NETWORK_STYLE_TEMPLATE",
    "NetworkGraph",
]
