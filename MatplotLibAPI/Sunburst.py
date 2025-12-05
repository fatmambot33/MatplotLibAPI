"""Sunburst chart plotting utilities."""

from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go

from .StyleTemplate import (
    TREEMAP_STYLE_TEMPLATE,
    StyleTemplate,
    validate_dataframe,
)


def fplot_sunburst(
    pd_df: pd.DataFrame,
    labels: str,
    parents: str,
    values: str,
    style: StyleTemplate = TREEMAP_STYLE_TEMPLATE,
    title: Optional[str] = None,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    max_values: int = 100,
    fig: Optional[go.Figure] = None,
    save_path: Optional[str] = None,
    savefig_kwargs: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """Return a figure containing the sunburst plot.

    Parameters
    ----------
    pd_df : pd.DataFrame
        DataFrame containing the data to plot.
    labels : str
        Column representing the labels of the sectors.
    parents : str
        Column representing the parent of each sector.
    values : str
        Column containing values for each sunburst sector.
    style : StyleTemplate, optional
        Style configuration. The default is `TREEMAP_STYLE_TEMPLATE`.
    title : str, optional
        Plot title.
    sort_by : str, optional
        Column used to sort data.
    ascending : bool, optional
        Sort order for the data. The default is `False`.
    max_values : int, optional
        Maximum number of rows to plot. The default is 100.
    fig : go.Figure, optional
        Existing figure to add the sunburst chart to.

    Returns
    -------
    go.Figure
        Figure containing the sunburst plot.
    """
    cols = [labels, parents, values]
    validate_dataframe(pd_df, cols=cols, sort_by=sort_by)
    if not sort_by:
        sort_by = values
    df = pd_df.sort_values(by=sort_by, ascending=ascending)[cols].head(max_values)

    trace = go.Sunburst(
        labels=df[labels],
        parents=df[parents],
        values=df[values],
        textinfo="label+percent entry",
    )

    if not fig:
        fig = go.Figure(trace)
    else:
        fig.add_trace(trace)

    fig.update_layout(
        title=title,
        plot_bgcolor=style.background_color,
        paper_bgcolor=style.background_color,
        font=dict(family=style.font_name, size=style.font_size, color=style.font_color),
        showlegend=style.legend if style else True,
    )
    if save_path:
        if save_path.lower().endswith((".html", ".htm")):
            fig.write_html(save_path, **(savefig_kwargs or {}))
        else:
            fig.write_image(save_path, **(savefig_kwargs or {}))
    return fig
