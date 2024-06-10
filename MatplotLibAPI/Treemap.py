# Hint for Visual Code Python Interactive window
# %%
from typing import Optional
import pandas as pd
from pandas import CategoricalDtype, BooleanDtype
import plotly.graph_objects as go

from .StyleTemplate import StyleTemplate, string_formatter, percent_formatter, validate_dataframe


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
