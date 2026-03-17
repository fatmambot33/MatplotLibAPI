"""Sunburst chart plotting utilities."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from .style_template import (
    TREEMAP_STYLE_TEMPLATE,
    StyleTemplate,
    validate_dataframe,
)

__all__ = ["TREEMAP_STYLE_TEMPLATE", "fplot_sunburst"]


class SunburstData:
    labels: list[str]
    parents: list[str]
    values: list[float]

    def __init__(self, labels: list[str], parents: list[str], values: list[float]):
        self.labels = labels
        self.parents = parents
        self.values = values

    @staticmethod
    def from_pandas(
        edges_df: pd.DataFrame,
        labels_col: str,
        parents_col: str,
        values_col: str,
    ) -> "SunburstData":
        """Create ``SunburstData`` from a DataFrame.

        Parameters
        ----------
        edges_df : pd.DataFrame
            DataFrame containing the data to plot.
        labels_col : str
            Column representing the labels of the sectors.
        parents_col : str
            Column representing the parent of each sector.
        values_col : str
            Column containing values for each sunburst sector.

        Returns
        -------
        SunburstData
            SunburstData object containing the labels, parents, and values for the sunburst plot.
        """
        validate_dataframe(edges_df, cols=[labels_col, parents_col, values_col])
        return SunburstData(
            labels=edges_df[labels_col].astype(str).tolist(),
            parents=edges_df[parents_col].astype(str).tolist(),
            values=edges_df[values_col].tolist(),
        )

    def fplot(
        self,
        style: StyleTemplate = TREEMAP_STYLE_TEMPLATE,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Return a figure containing the sunburst plot.

        Parameters
        ----------
        style : StyleTemplate, optional
            Style configuration. The default is `TREEMAP_STYLE_TEMPLATE`.
        title : str, optional
            Plot title.

        Returns
        -------
        go.Figure
            Figure containing the sunburst plot.
        """
        trace = go.Sunburst(
            labels=self.labels,
            parents=self.parents,
            values=self.values,
            textinfo="label+percent entry",
        )

        fig = go.Figure(trace)

        fig.update_layout(
            title=title,
            plot_bgcolor=style.background_color,
            paper_bgcolor=style.background_color,
            font=dict(
                family=style.font_name, size=style.font_size, color=style.font_color
            ),
            showlegend=style.legend if style else True,
        )
        return fig


def fplot_sunburst(
    pd_df: pd.DataFrame,
    labels: str,
    parents: str,
    values: str,
    style: StyleTemplate = TREEMAP_STYLE_TEMPLATE,
    title: Optional[str] = None,
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

    Returns
    -------
    go.Figure
        Figure containing the sunburst plot.
    """
    return SunburstData.from_pandas(
        edges_df=pd_df,
        labels_col=labels,
        parents_col=parents,
        values_col=values,
    ).fplot(
        style=style,
        title=title,
    )
