"""Treemap plotting utilities."""

# Hint for Visual Code Python Interactive window
# %%
from typing import Optional
import pandas as pd
from pandas import CategoricalDtype, BooleanDtype
import plotly.graph_objects as go

from MatplotLibAPI.StyleTemplate import StyleTemplate, string_formatter, percent_formatter, validate_dataframe


TREEMAP_STYLE_TEMPLATE = StyleTemplate(
    background_color='black',
    fig_border='darkgrey',
    font_color='white',
    palette='magma',
    format_funcs={"y": percent_formatter,
                  "label": string_formatter}
)


def aplot_treemap(pd_df: pd.DataFrame,
                  path: str,
                  values: str,
                  style: StyleTemplate = TREEMAP_STYLE_TEMPLATE,
                  title: Optional[str] = None,
                  color: Optional[str] = None,
                  sort_by: Optional[str] = None,
                  ascending: bool = False,
                  max_values: int = 100) -> go.Trace:
    """Create a treemap trace from the data frame.

    Args:
        pd_df (pd.DataFrame): DataFrame containing the data to plot.
        path (str): Column representing hierarchical path.
        values (str): Column containing values for each treemap block.
        style (StyleTemplate, optional): Style configuration. Defaults to ``TREEMAP_STYLE_TEMPLATE``.
        title (Optional[str], optional): Plot title. Defaults to ``None``.
        color (Optional[str], optional): Column used for coloring. Defaults to ``None``.
        sort_by (Optional[str], optional): Column used to sort data. Defaults to ``None``.
        ascending (bool, optional): Sort order for the data. Defaults to ``False``.
        max_values (int, optional): Maximum number of rows to plot. Defaults to ``100``.

    Returns:
        go.Trace: Plotly treemap trace.
    """
    cols = [path, values]
    if color:
        cols.append(color)
    validate_dataframe(pd_df, cols=cols, sort_by=sort_by)
    if not sort_by:
        sort_by = values
    df = pd_df.sort_values(by=sort_by, ascending=ascending)[
        cols].head(max_values)
    data = {"labels": df[path],
            "parents": [""] * len(df),
            "values": df[values],
            "textinfo": "label",
            "name": title,
            "textfont":
                {"family": style.font_name,
                 "size": style.font_size,
                 "color": style.font_color}
            }

    if color and color in pd_df.columns:
        color_data = pd_df[color]
        if isinstance(color_data, CategoricalDtype) or pd.api.types.is_object_dtype(color_data):
            color_data = color_data.astype('category').cat.codes
        elif isinstance(color_data, BooleanDtype):
            color_data = color_data.astype(int)
        data['marker'] = dict(colorscale="Viridis",
                              colors=color_data.to_list())

    g = go.Treemap(data,
                   root_color=style.background_color
                   )

    return g


def fplot_treemap(pd_df: pd.DataFrame,
                  path: str,
                  values: str,
                  style: StyleTemplate = TREEMAP_STYLE_TEMPLATE,
                  title: Optional[str] = None,
                  color: Optional[str] = None,
                  sort_by: Optional[str] = None,
                  ascending: bool = False,
                  max_values: int = 100,
                  fig: Optional[go.Figure] = None) -> go.Figure:
    """Return a figure containing the treemap plot.

    Args:
        pd_df (pd.DataFrame): DataFrame containing the data to plot.
        path (str): Column representing hierarchical path.
        values (str): Column containing values for each treemap block.
        style (StyleTemplate, optional): Style configuration. Defaults to ``TREEMAP_STYLE_TEMPLATE``.
        title (Optional[str], optional): Plot title. Defaults to ``None``.
        color (Optional[str], optional): Column used for coloring. Defaults to ``None``.
        sort_by (Optional[str], optional): Column used to sort data. Defaults to ``None``.
        ascending (bool, optional): Sort order for the data. Defaults to ``False``.
        max_values (int, optional): Maximum number of rows to plot. Defaults to ``100``.
        fig (Optional[go.Figure], optional): Existing figure to add the treemap to. Defaults to ``None``.

    Returns:
        go.Figure: Figure containing the treemap plot.
    """
    g = aplot_treemap(pd_df=pd_df,
                      path=path,
                      values=values,
                      title=title,
                      style=style,
                      color=color,
                      sort_by=sort_by,
                      ascending=ascending,
                      max_values=max_values)

    if not fig:
        fig = go.Figure(g)
    else:
        fig.add_trace(g)

    fig.update_layout(
        title=title,
        plot_bgcolor=style.background_color,
        paper_bgcolor=style.background_color,
        font=dict(
            family=style.font_name,
            size=style.font_size,
            color=style.font_color
        ),
        showlegend=style.legend if style else True)

    # Apply color scale
    fig.update_traces(
        marker=dict(colorscale=style.palette)
    )

    return fig
