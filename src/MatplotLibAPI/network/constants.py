"""Constants used by network plotting helpers."""

from __future__ import annotations

import numpy as np

_DEFAULT = {
    "MAX_EDGES": 100,
    "MAX_NODES": 30,
    "MIN_NODE_SIZE": 100,
    "MAX_NODE_SIZE": 2000,
    "MAX_EDGE_WIDTH": 10,
    "GRAPH_SCALE": 2,
    "MAX_FONT_SIZE": 20,
    "MIN_FONT_SIZE": 8,
    "SPRING_LAYOUT_K": 1.0,
    "SPRING_LAYOUT_SEED": 42,
}

_WEIGHT_PERCENTILES = np.arange(10, 100, 10)


__all__ = ["_DEFAULT", "_WEIGHT_PERCENTILES"]
