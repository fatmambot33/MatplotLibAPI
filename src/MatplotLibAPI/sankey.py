"""Sankey plotting helpers."""

from dataclasses import dataclass
from typing import Dict, List, Optional, cast

import pandas as pd
import plotly.graph_objects as go

from .style_template import SANKEY_STYLE_TEMPLATE, StyleTemplate, validate_dataframe

__all__ = ["SANKEY_STYLE_TEMPLATE", "fplot_sankey"]


@dataclass
class SankeyNode:
    label: list[str]


@dataclass
class SankeyLink:
    source: list[int]
    target: list[int]
    value: list[float]

    def __post_init__(self) -> None:
        """Validate that all lists have the same length."""
        if not (len(self.source) == len(self.target) == len(self.value)):
            raise ValueError(
                f"All lists must have the same length. "
                f"Got source={len(self.source)}, target={len(self.target)}, value={len(self.value)}"
            )


@dataclass
class SankeyData:
    node: SankeyNode
    link: SankeyLink

    @staticmethod
    def from_pandas_edgelist(
        edges_df: pd.DataFrame,
        source: str = "source",
        target: str = "target",
        edge_weight_col: str = "weight",
    ) -> "SankeyData":
        """Create SankeyData from a DataFrame."""
        validate_dataframe(edges_df, cols=[source, target, edge_weight_col])
        source_series = cast(pd.Series, edges_df[source])
        target_series = cast(pd.Series, edges_df[target])
        labels: List[str] = list(pd.unique(pd.concat([source_series, target_series])))
        label_to_index: Dict[str, int] = {name: idx for idx, name in enumerate(labels)}

        return SankeyData(
            node=SankeyNode(label=labels),
            link=SankeyLink(
                source=[label_to_index[val] for val in edges_df[source]],
                target=[label_to_index[val] for val in edges_df[target]],
                value=edges_df[edge_weight_col].tolist(),
            ),
        )


def fplot_sankey(
    pd_df: pd.DataFrame,
    source: str,
    target: str,
    value: str,
    title: Optional[str] = None,
    style: StyleTemplate = SANKEY_STYLE_TEMPLATE,
) -> go.Figure:
    """Plot a Sankey diagram showing flows between categories."""
    sankey_data = SankeyData.from_pandas_edgelist(pd_df, source, target, value)

    sankey = go.Sankey(
        node=dict(
            label=sankey_data.node.label,
            pad=15,
            thickness=20,
            color=style.font_color,
        ),
        link=dict(
            source=sankey_data.link.source,
            target=sankey_data.link.target,
            value=sankey_data.link.value,
        ),
    )

    fig = go.Figure(sankey)
    if title:
        fig.update_layout(
            title_text=title, font=dict(color=style.font_color, size=style.font_size)
        )
    return fig
