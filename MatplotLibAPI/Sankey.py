"""Sankey plotting helpers."""

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

from .StyleTemplate import SANKEY_STYLE_TEMPLATE, StyleTemplate, validate_dataframe


def fplot_sankey(
    pd_df: pd.DataFrame,
    source: str,
    target: str,
    value: str,
    title: Optional[str] = None,
    style: StyleTemplate = SANKEY_STYLE_TEMPLATE,
    save_path: Optional[str] = None,
    savefig_kwargs: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """Plot a Sankey diagram showing flows between categories."""
    validate_dataframe(pd_df, cols=[source, target, value])

    labels: List[str] = list(pd.unique(pd.concat([pd_df[source], pd_df[target]])))
    label_to_index: Dict[str, int] = {name: idx for idx, name in enumerate(labels)}

    sankey = go.Sankey(
        node=dict(label=labels, pad=15, thickness=20, color=style.font_color),
        link=dict(
            source=[label_to_index[val] for val in pd_df[source]],
            target=[label_to_index[val] for val in pd_df[target]],
            value=pd_df[value],
        ),
    )

    fig = go.Figure(sankey)
    if title:
        fig.update_layout(
            title_text=title, font=dict(color=style.font_color, size=style.font_size)
        )
    if save_path:
        if save_path.lower().endswith(('.html', '.htm')):
            fig.write_html(save_path, **(savefig_kwargs or {}))
        else:
            fig.write_image(save_path, **(savefig_kwargs or {}))
    return fig
