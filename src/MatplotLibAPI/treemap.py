"""Treemap plotting utilities."""

from typing import Any, Dict, Optional, cast

import pandas as pd
import plotly.graph_objects as go
from pandas import BooleanDtype, CategoricalDtype

from .style_template import TREEMAP_STYLE_TEMPLATE, StyleTemplate, validate_dataframe

__all__ = [
    "TREEMAP_STYLE_TEMPLATE",
    "aplot_treemap",
    "fplot_treemap",
]


def aplot_treemap(
    pd_df: pd.DataFrame,
    path: str,
    values: str,
    style: Optional[StyleTemplate] = None,
    title: Optional[str] = None,
    color: Optional[str] = None,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    max_values: int = 100,
    **kwargs: Any,
) -> go.Treemap:
    """Create a treemap trace from the provided dataframe.

    Parameters
    ----------
    pd_df : pd.DataFrame
        DataFrame containing the data to plot.
    path : str
        Column representing the hierarchical path.
    values : str
        Column containing values for each treemap block.
    style : StyleTemplate, optional
        Style configuration. The default is `TREEMAP_STYLE_TEMPLATE`.
    title : str, optional
        Plot title. The default is None.
    color : str, optional
        Column used for coloring. The default is None.
    sort_by : str, optional
        Column used to sort data. The default is None.
    ascending : bool, optional
        Sort order for the data. The default is False.
    max_values : int, optional
        Maximum number of rows to plot. The default is 100.

    Returns
    -------
    go.Treemap
        Plotly treemap trace.
    """
    cols = [path, values]
    if color:
        cols.append(color)
    validate_dataframe(pd_df, cols=cols, sort_by=sort_by)

    sort_col = sort_by or values
    df = pd_df.sort_values(by=sort_col, ascending=ascending)[cols].head(max_values)
    if not style:
        style = TREEMAP_STYLE_TEMPLATE
    data: Dict[str, Any] = {
        "labels": df[path],
        "parents": [""] * len(df),
        "values": df[values],
        "textinfo": "label",
        "name": title,
        "textfont": {
            "family": style.font_name,
            "size": style.font_size,
            "color": style.font_color,
        },
    }

    if color and color in df.columns:
        color_data = cast(pd.Series, df[color])
        if isinstance(
            color_data.dtype, CategoricalDtype
        ) or pd.api.types.is_object_dtype(color_data.dtype):
            color_data = color_data.astype("category").cat.codes
        elif isinstance(color_data.dtype, BooleanDtype):
            color_data = color_data.astype(int)
        data["marker"] = dict(colorscale="Viridis", colors=color_data.tolist())

    return go.Treemap(data, root_color=style.background_color)


def fplot_treemap(
    pd_df: pd.DataFrame,
    path: str,
    values: str,
    style: StyleTemplate = TREEMAP_STYLE_TEMPLATE,
    title: Optional[str] = None,
    color: Optional[str] = None,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    max_values: int = 100,
    fig: Optional[go.Figure] = None,
) -> go.Figure:
    """Return a figure containing a treemap plot.

    Parameters
    ----------
    pd_df : pd.DataFrame
        DataFrame containing the data to plot.
    path : str
        Column representing the hierarchical path.
    values : str
        Column containing values for each treemap block.
    style : StyleTemplate, optional
        Style configuration. The default is `TREEMAP_STYLE_TEMPLATE`.
    title : str, optional
        Plot title. The default is None.
    color : str, optional
        Column used for coloring. The default is None.
    sort_by : str, optional
        Column used to sort data. The default is None.
    ascending : bool, optional
        Sort order for the data. The default is False.
    max_values : int, optional
        Maximum number of rows to plot. The default is 100.
    fig : go.Figure, optional
        Existing figure to add the treemap to. If None, create a new figure.
    save_path : str, optional
        Path to save the figure as HTML or static image. The default is None.
    savefig_kwargs : dict[str, Any], optional
        Extra keyword arguments passed to Plotly save methods.

    Returns
    -------
    go.Figure
        Figure containing the treemap plot.
    """
    treemap_trace = aplot_treemap(
        pd_df=pd_df,
        path=path,
        values=values,
        title=title,
        style=style,
        color=color,
        sort_by=sort_by,
        ascending=ascending,
        max_values=max_values,
    )

    figure = go.Figure(treemap_trace) if fig is None else fig
    if fig is not None:
        figure.add_trace(treemap_trace)

    figure.update_layout(
        title=title,
        plot_bgcolor=style.background_color,
        paper_bgcolor=style.background_color,
        font=dict(family=style.font_name, size=style.font_size, color=style.font_color),
        showlegend=style.legend,
    )
    return figure
