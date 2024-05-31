from typing import Optional
import pandas as pd
import plotly.graph_objects as go
from .Style import StyleTemplate, string_formatter, _validate_panda, percent_formatter

TREEMAP_STYLE_TEMPLATE = StyleTemplate(
    background_color='black',
    fig_border='darkgrey',
    font_color='white',
    palette='magma',
    format_funcs={"y": percent_formatter,
                  "label": string_formatter}
)


def plot_treemap(pd_df: pd.DataFrame,
                 path: str,
                 values: str,
                 style: StyleTemplate = TREEMAP_STYLE_TEMPLATE,
                 title: Optional[str] = None,
                 color: Optional[str] = None,
                 sort_by: Optional[str] = None,
                 ascending: bool = False,
                 max_values: int = 100,
                 fig: Optional[go.Figure] = None) -> go.Figure:
    cols = [path, values]
    if color:
        cols.append(color)
    _validate_panda(pd_df, cols=cols, sort_by=sort_by)
    if not sort_by:
        sort_by = values
    df = pd_df.sort_values(by=sort_by, ascending=ascending)[
        cols].head(max_values)
    data = {"labels": df[path],
            "parents": [""] * len(df),
            "values": df[values],
            "textinfo": "label",
            "name": title}

    if color:
        df['color'] = df[color]

    if not fig:
        fig = go.Figure(data=data)
    else:
        fig.add_trace(go.Treemap(data))
    
    fig.update_layout(
        paper_bgcolor=style.background_color,
        plot_bgcolor=style.background_color,
        font=dict(color=style.font_color),
        margin=dict(t=50, l=25, r=25, b=25))

    # Apply color scale
    fig.update_traces(
        marker=dict(colorscale=style.palette)
    )

    return fig
