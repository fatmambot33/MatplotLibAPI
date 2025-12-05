"""Pandas accessor exposing MatplotLibAPI plotting helpers."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas.api.extensions import register_dataframe_accessor

from .Area import aplot_area, fplot_area
from .Bar import aplot_bar, fplot_bar
from .BoxViolin import aplot_box_violin, fplot_box_violin
from .Bubble import BUBBLE_STYLE_TEMPLATE, aplot_bubble, fplot_bubble
from .Composite import plot_composite_bubble, plot_composite_treemap
from .Heatmap import (
    HEATMAP_STYLE_TEMPLATE,
    aplot_correlation_matrix,
    aplot_heatmap,
    fplot_correlation_matrix,
    fplot_heatmap,
)
from .Histogram import aplot_histogram_kde, fplot_histogram_kde
from .Network import (
    NETWORK_STYLE_TEMPLATE,
    aplot_network,
    aplot_network_components,
    fplot_network,
)
from .Pie import aplot_pie_donut, fplot_pie_donut
from .Sankey import SANKEY_STYLE_TEMPLATE, fplot_sankey
from .StyleTemplate import (
    AREA_STYLE_TEMPLATE,
    DISTRIBUTION_STYLE_TEMPLATE,
    PIE_STYLE_TEMPLATE,
    TABLE_STYLE_TEMPLATE,
    TIMESERIE_STYLE_TEMPLATE,
    TREEMAP_STYLE_TEMPLATE,
    StyleTemplate,
)
from .Table import aplot_table, fplot_table
from .Timeserie import aplot_timeserie, fplot_timeserie
from .Sunburst import fplot_sunburst
from .Treemap import fplot_treemap
from .Waffle import aplot_waffle, fplot_waffle
from .Wordcloud import WORDCLOUD_STYLE_TEMPLATE, aplot_wordcloud, fplot_wordcloud


@register_dataframe_accessor("mpl")
class DataFrameAccessor:
    """Expose MatplotLibAPI plotting helpers as a pandas accessor.

    Methods
    -------
    aplot_bubble
        Plot a bubble chart using the underlying DataFrame.
    fplot_bubble
        Plot a bubble chart on a new figure.
    fplot_composite_bubble
        Plot a composite bubble chart with summary tables.
    aplot_bar
        Plot a bar or stacked bar chart on existing axes.
    fplot_bar
        Plot a bar or stacked bar chart on a new figure.
    aplot_histogram_kde
        Plot a histogram with optional KDE overlay.
    fplot_histogram_kde
        Plot a histogram with optional KDE overlay on a new figure.
    aplot_box_violin
        Plot box or violin plots to summarize distributions.
    fplot_box_violin
        Plot box or violin plots on a new figure.
    aplot_heatmap
        Plot a heatmap for dense categorical combinations.
    fplot_heatmap
        Plot a heatmap on a new figure.
    aplot_correlation_matrix
        Plot a correlation matrix heatmap.
    fplot_correlation_matrix
        Plot a correlation matrix heatmap on a new figure.
    aplot_area
        Plot an area chart, optionally stacked.
    fplot_area
        Plot an area chart, optionally stacked, on a new figure.
    aplot_pie_donut
        Plot pie or donut charts for categorical shares.
    fplot_pie_donut
        Plot pie or donut charts on a new figure.
    aplot_waffle
        Plot waffle charts for categorical proportions.
    fplot_waffle
        Plot waffle charts on a new figure.
    fplot_sankey
        Plot a Sankey diagram for flow data.
    aplot_table
        Plot a table of the DataFrame's data on a Matplotlib axes.
    fplot_table
        Plot a table of the DataFrame's data on a new figure.
    aplot_timeserie
        Plot a time series on a Matplotlib axes.
    fplot_timeserie
        Plot a time series on a new figure.
    aplot_wordcloud
        Plot a word cloud on a Matplotlib axes.
    fplot_wordcloud
        Plot a word cloud on a new figure.
    aplot_network
        Plot a network graph on a Matplotlib axes.
    aplot_network_components
        Plot connected components of a network graph on multiple axes.
    fplot_network
        Plot a network graph on a new figure.
    fplot_treemap
        Plot a treemap on a new Plotly figure.
    fplot_sunburst
        Plot a sunburst chart on a new Plotly figure.
    fplot_composite_treemap
        Plot a composite treemap on a new Plotly figure.
    """

    def __init__(self, pd_df: pd.DataFrame):
        """Store the parent DataFrame."""
        self._obj = pd_df

    def aplot_bubble(
        self,
        label: str,
        x: str,
        y: str,
        z: str,
        title: Optional[str] = None,
        style: StyleTemplate = BUBBLE_STYLE_TEMPLATE,
        max_values: int = 50,
        center_to_mean: bool = False,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        hline: bool = False,
        vline: bool = False,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Plot a bubble chart using the underlying DataFrame.

        Parameters
        ----------
        label : str
            Column to use for bubble labels.
        x : str
            Column to use for the x-axis.
        y : str
            Column to use for the y-axis.
        z : str
            Column to use for the bubble size.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is `BUBBLE_STYLE_TEMPLATE`.
        max_values : int, optional
            Maximum number of bubbles to plot. The default is 50.
        center_to_mean : bool, optional
            If True, center x-axis values around the mean. The default is `False`.
        sort_by : str, optional
            Column to sort by before plotting.
        ascending : bool, optional
            Sort order. The default is `False`.
        hline : bool, optional
            If True, draw a horizontal line at the mean of y-values. The default is `False`.
        vline : bool, optional
            If True, draw a vertical line at the mean of x-values. The default is `False`.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses the current axes.

        Returns
        -------
        Axes
            The Matplotlib axes object with the plot.
        """
        return aplot_bubble(
            pd_df=self._obj,
            label=label,
            x=x,
            y=y,
            z=z,
            title=title,
            style=style,
            max_values=max_values,
            center_to_mean=center_to_mean,
            sort_by=sort_by,
            ascending=ascending,
            hline=hline,
            vline=vline,
            ax=ax,
        )

    def fplot_bubble(
        self,
        label: str,
        x: str,
        y: str,
        z: str,
        title: Optional[str] = None,
        style: StyleTemplate = BUBBLE_STYLE_TEMPLATE,
        max_values: int = 50,
        center_to_mean: bool = False,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        hline: bool = False,
        vline: bool = False,
        figsize: Tuple[float, float] = (19.2, 10.8),
    ) -> Figure:
        """Plot a bubble chart on a new figure.

        Parameters
        ----------
        label : str
            Column to use for bubble labels.
        x : str
            Column to use for the x-axis.
        y : str
            Column to use for the y-axis.
        z : str
            Column to use for the bubble size.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is `BUBBLE_STYLE_TEMPLATE`.
        max_values : int, optional
            Maximum number of bubbles to plot. The default is 50.
        center_to_mean : bool, optional
            If True, center x-axis values around the mean. The default is `False`.
        sort_by : str, optional
            Column to sort by before plotting.
        ascending : bool, optional
            Sort order. The default is `False`.
        hline : bool, optional
            If True, draw a horizontal line at the mean of y-values. The default is `False`.
        vline : bool, optional
            If True, draw a vertical line at the mean of x-values. The default is `False`.
        figsize : tuple[float, float], optional
            Figure size. The default is (19.2, 10.8).

        Returns
        -------
        Figure
            The new Matplotlib figure with the plot.
        """
        return fplot_bubble(
            pd_df=self._obj,
            label=label,
            x=x,
            y=y,
            z=z,
            title=title,
            style=style,
            max_values=max_values,
            center_to_mean=center_to_mean,
            sort_by=sort_by,
            ascending=ascending,
            hline=hline,
            vline=vline,
            figsize=figsize,
        )

    def fplot_composite_bubble(
        self,
        label: str,
        x: str,
        y: str,
        z: str,
        title: Optional[str] = None,
        style: StyleTemplate = BUBBLE_STYLE_TEMPLATE,
        max_values: int = 100,
        center_to_mean: bool = False,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        table_rows: int = 10,
    ) -> Figure:
        """Plot a composite bubble chart with summary tables.

        This plot combines a bubble chart with tables summarizing the
        top and bottom rows of the dataset.

        Parameters
        ----------
        label : str
            Column to use for bubble labels.
        x : str
            Column to use for the x-axis.
        y : str
            Column to use for the y-axis.
        z : str
            Column to use for the bubble size.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is `BUBBLE_STYLE_TEMPLATE`.
        max_values : int, optional
            Maximum number of bubbles to plot. The default is 100.
        center_to_mean : bool, optional
            If True, center x-axis values around the mean. The default is `False`.
        sort_by : str, optional
            Column to sort by before plotting.
        ascending : bool, optional
            Sort order. The default is `False`.
        table_rows : int, optional
            Number of rows for the summary tables. The default is 10.

        Returns
        -------
        Figure
            The new Matplotlib figure with the composite plot.
        """
        return plot_composite_bubble(
            pd_df=self._obj,
            label=label,
            x=x,
            y=y,
            z=z,
            title=title,
            style=style,
            max_values=max_values,
            center_to_mean=center_to_mean,
            sort_by=sort_by,
            ascending=ascending,
            table_rows=table_rows,
        )

    def aplot_bar(
        self,
        category: str,
        value: str,
        group: Optional[str] = None,
        stacked: bool = False,
        title: Optional[str] = None,
        style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Plot bar or stacked bar charts on existing axes.

        Parameters
        ----------
        category : str
            Column to plot along the x-axis.
        value : str
            Column representing bar heights.
        group : str, optional
            Optional grouping column for multiple series.
        stacked : bool, optional
            Whether to stack grouped bars. The default is ``False``.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is ``DISTRIBUTION_STYLE_TEMPLATE``.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses the current axes.

        Returns
        -------
        Axes
            The Matplotlib axes object with the bar chart.
        """
        return aplot_bar(
            pd_df=self._obj,
            category=category,
            value=value,
            group=group,
            stacked=stacked,
            title=title,
            style=style,
            ax=ax,
        )

    def fplot_bar(
        self,
        category: str,
        value: str,
        group: Optional[str] = None,
        stacked: bool = False,
        title: Optional[str] = None,
        style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
        figsize: Tuple[float, float] = (19.2, 10.8),
    ) -> Figure:
        """Plot bar or stacked bar charts on a new figure.

        Parameters
        ----------
        category : str
            Column to plot along the x-axis.
        value : str
            Column representing bar heights.
        group : str, optional
            Optional grouping column for multiple series.
        stacked : bool, optional
            Whether to stack grouped bars. The default is ``False``.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is ``DISTRIBUTION_STYLE_TEMPLATE``.
        figsize : tuple[float, float], optional
            Figure size. The default is (19.2, 10.8).

        Returns
        -------
        Figure
            The new Matplotlib figure with the bar chart.
        """
        return fplot_bar(
            pd_df=self._obj,
            category=category,
            value=value,
            group=group,
            stacked=stacked,
            title=title,
            style=style,
            figsize=figsize,
        )

    def aplot_histogram_kde(
        self,
        column: str,
        bins: int = 20,
        kde: bool = True,
        title: Optional[str] = None,
        style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Plot a histogram with an optional KDE overlay.

        Parameters
        ----------
        column : str
            Column to plot.
        bins : int, optional
            Number of bins. The default is 20.
        kde : bool, optional
            Whether to add a KDE line. The default is ``True``.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is ``DISTRIBUTION_STYLE_TEMPLATE``.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses the current axes.

        Returns
        -------
        Axes
            The Matplotlib axes object with the histogram.
        """
        return aplot_histogram_kde(
            pd_df=self._obj,
            column=column,
            bins=bins,
            kde=kde,
            title=title,
            style=style,
            ax=ax,
        )

    def fplot_histogram_kde(
        self,
        column: str,
        bins: int = 20,
        kde: bool = True,
        title: Optional[str] = None,
        style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
        figsize: Tuple[float, float] = (19.2, 10.8),
    ) -> Figure:
        """Plot a histogram with an optional KDE on a new figure.

        Parameters
        ----------
        column : str
            Column to plot.
        bins : int, optional
            Number of bins. The default is 20.
        kde : bool, optional
            Whether to add a KDE line. The default is ``True``.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is ``DISTRIBUTION_STYLE_TEMPLATE``.
        figsize : tuple[float, float], optional
            Figure size. The default is (19.2, 10.8).

        Returns
        -------
        Figure
            The new Matplotlib figure with the histogram.
        """
        return fplot_histogram_kde(
            pd_df=self._obj,
            column=column,
            bins=bins,
            kde=kde,
            title=title,
            style=style,
            figsize=figsize,
        )

    def aplot_box_violin(
        self,
        column: str,
        by: Optional[str] = None,
        violin: bool = False,
        title: Optional[str] = None,
        style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Plot box or violin charts on existing axes.

        Parameters
        ----------
        column : str
            Column to summarize.
        by : str, optional
            Optional grouping column.
        violin : bool, optional
            Whether to draw a violin plot. The default is ``False``.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is ``DISTRIBUTION_STYLE_TEMPLATE``.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses the current axes.

        Returns
        -------
        Axes
            The Matplotlib axes object with the distribution summary.
        """
        return aplot_box_violin(
            pd_df=self._obj,
            column=column,
            by=by,
            violin=violin,
            title=title,
            style=style,
            ax=ax,
        )

    def fplot_box_violin(
        self,
        column: str,
        by: Optional[str] = None,
        violin: bool = False,
        title: Optional[str] = None,
        style: StyleTemplate = DISTRIBUTION_STYLE_TEMPLATE,
        figsize: Tuple[float, float] = (19.2, 10.8),
    ) -> Figure:
        """Plot box or violin charts on a new figure.

        Parameters
        ----------
        column : str
            Column to summarize.
        by : str, optional
            Optional grouping column.
        violin : bool, optional
            Whether to draw a violin plot. The default is ``False``.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is ``DISTRIBUTION_STYLE_TEMPLATE``.
        figsize : tuple[float, float], optional
            Figure size. The default is (19.2, 10.8).

        Returns
        -------
        Figure
            The new Matplotlib figure with the distribution summary.
        """
        return fplot_box_violin(
            pd_df=self._obj,
            column=column,
            by=by,
            violin=violin,
            title=title,
            style=style,
            figsize=figsize,
        )

    def aplot_heatmap(
        self,
        x: str,
        y: str,
        value: str,
        title: Optional[str] = None,
        style: StyleTemplate = HEATMAP_STYLE_TEMPLATE,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Plot a heatmap for dense categorical combinations.

        Parameters
        ----------
        x : str
            Column for heatmap columns.
        y : str
            Column for heatmap rows.
        value : str
            Column for heatmap values.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is ``HEATMAP_STYLE_TEMPLATE``.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses the current axes.

        Returns
        -------
        Axes
            The Matplotlib axes object with the heatmap.
        """
        return aplot_heatmap(
            pd_df=self._obj,
            x=x,
            y=y,
            value=value,
            title=title,
            style=style,
            ax=ax,
        )

    def fplot_heatmap(
        self,
        x: str,
        y: str,
        value: str,
        title: Optional[str] = None,
        style: StyleTemplate = HEATMAP_STYLE_TEMPLATE,
        figsize: Tuple[float, float] = (19.2, 10.8),
    ) -> Figure:
        """Plot a heatmap on a new figure.

        Parameters
        ----------
        x : str
            Column for heatmap columns.
        y : str
            Column for heatmap rows.
        value : str
            Column for heatmap values.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is ``HEATMAP_STYLE_TEMPLATE``.
        figsize : tuple[float, float], optional
            Figure size. The default is (19.2, 10.8).

        Returns
        -------
        Figure
            The new Matplotlib figure with the heatmap.
        """
        return fplot_heatmap(
            pd_df=self._obj,
            x=x,
            y=y,
            value=value,
            title=title,
            style=style,
            figsize=figsize,
        )

    def aplot_correlation_matrix(
        self,
        columns: Optional[List[str]] = None,
        method: str = "pearson",
        title: Optional[str] = None,
        style: StyleTemplate = HEATMAP_STYLE_TEMPLATE,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Plot a correlation matrix heatmap.

        Parameters
        ----------
        columns : list[str], optional
            Numeric columns to include. The default is ``None`` for all numeric columns.
        method : str, optional
            Correlation method. The default is "pearson".
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is ``HEATMAP_STYLE_TEMPLATE``.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses the current axes.

        Returns
        -------
        Axes
            The Matplotlib axes object with the correlation matrix.
        """
        return aplot_correlation_matrix(
            pd_df=self._obj,
            columns=columns,
            method=method,
            title=title,
            style=style,
            ax=ax,
        )

    def fplot_correlation_matrix(
        self,
        columns: Optional[List[str]] = None,
        method: str = "pearson",
        title: Optional[str] = None,
        style: StyleTemplate = HEATMAP_STYLE_TEMPLATE,
        figsize: Tuple[float, float] = (19.2, 10.8),
    ) -> Figure:
        """Plot a correlation matrix heatmap on a new figure.

        Parameters
        ----------
        columns : list[str], optional
            Numeric columns to include. The default is ``None`` for all numeric columns.
        method : str, optional
            Correlation method. The default is "pearson".
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is ``HEATMAP_STYLE_TEMPLATE``.
        figsize : tuple[float, float], optional
            Figure size. The default is (19.2, 10.8).

        Returns
        -------
        Figure
            The new Matplotlib figure with the correlation matrix.
        """
        return fplot_correlation_matrix(
            pd_df=self._obj,
            columns=columns,
            method=method,
            title=title,
            style=style,
            figsize=figsize,
        )

    def aplot_area(
        self,
        x: str,
        y: str,
        label: Optional[str] = None,
        stacked: bool = True,
        title: Optional[str] = None,
        style: StyleTemplate = AREA_STYLE_TEMPLATE,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Plot an area chart on existing axes.

        Parameters
        ----------
        x : str
            Column for the x-axis.
        y : str
            Column for the area heights.
        label : str, optional
            Optional grouping column.
        stacked : bool, optional
            Whether to stack grouped areas. The default is ``True``.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is ``AREA_STYLE_TEMPLATE``.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses the current axes.

        Returns
        -------
        Axes
            The Matplotlib axes object with the area chart.
        """
        return aplot_area(
            pd_df=self._obj,
            x=x,
            y=y,
            label=label,
            stacked=stacked,
            title=title,
            style=style,
            ax=ax,
        )

    def fplot_area(
        self,
        x: str,
        y: str,
        label: Optional[str] = None,
        stacked: bool = True,
        title: Optional[str] = None,
        style: StyleTemplate = AREA_STYLE_TEMPLATE,
        figsize: Tuple[float, float] = (19.2, 10.8),
    ) -> Figure:
        """Plot an area chart on a new figure.

        Parameters
        ----------
        x : str
            Column for the x-axis.
        y : str
            Column for the area heights.
        label : str, optional
            Optional grouping column.
        stacked : bool, optional
            Whether to stack grouped areas. The default is ``True``.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is ``AREA_STYLE_TEMPLATE``.
        figsize : tuple[float, float], optional
            Figure size. The default is (19.2, 10.8).

        Returns
        -------
        Figure
            The new Matplotlib figure with the area chart.
        """
        return fplot_area(
            pd_df=self._obj,
            x=x,
            y=y,
            label=label,
            stacked=stacked,
            title=title,
            style=style,
            figsize=figsize,
        )

    def aplot_pie_donut(
        self,
        category: str,
        value: str,
        donut: bool = False,
        title: Optional[str] = None,
        style: StyleTemplate = PIE_STYLE_TEMPLATE,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Plot pie or donut charts for categorical shares.

        Parameters
        ----------
        category : str
            Column for slice labels.
        value : str
            Column for slice sizes.
        donut : bool, optional
            Whether to draw a donut chart. The default is ``False``.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is ``PIE_STYLE_TEMPLATE``.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses the current axes.

        Returns
        -------
        Axes
            The Matplotlib axes object with the pie or donut chart.
        """
        return aplot_pie_donut(
            pd_df=self._obj,
            category=category,
            value=value,
            donut=donut,
            title=title,
            style=style,
            ax=ax,
        )

    def fplot_pie_donut(
        self,
        category: str,
        value: str,
        donut: bool = False,
        title: Optional[str] = None,
        style: StyleTemplate = PIE_STYLE_TEMPLATE,
        figsize: Tuple[float, float] = (19.2, 10.8),
    ) -> Figure:
        """Plot pie or donut charts on a new figure.

        Parameters
        ----------
        category : str
            Column for slice labels.
        value : str
            Column for slice sizes.
        donut : bool, optional
            Whether to draw a donut chart. The default is ``False``.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is ``PIE_STYLE_TEMPLATE``.
        figsize : tuple[float, float], optional
            Figure size. The default is (19.2, 10.8).

        Returns
        -------
        Figure
            The new Matplotlib figure with the pie or donut chart.
        """
        return fplot_pie_donut(
            pd_df=self._obj,
            category=category,
            value=value,
            donut=donut,
            title=title,
            style=style,
            figsize=figsize,
        )

    def aplot_waffle(
        self,
        category: str,
        value: str,
        rows: int = 10,
        title: Optional[str] = None,
        style: StyleTemplate = PIE_STYLE_TEMPLATE,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Plot waffle charts for categorical proportions.

        Parameters
        ----------
        category : str
            Column for segment labels.
        value : str
            Column for segment sizes.
        rows : int, optional
            Number of grid rows. The default is 10.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is ``PIE_STYLE_TEMPLATE``.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses the current axes.

        Returns
        -------
        Axes
            The Matplotlib axes object with the waffle chart.
        """
        return aplot_waffle(
            pd_df=self._obj,
            category=category,
            value=value,
            rows=rows,
            title=title,
            style=style,
            ax=ax,
        )

    def fplot_waffle(
        self,
        category: str,
        value: str,
        rows: int = 10,
        title: Optional[str] = None,
        style: StyleTemplate = PIE_STYLE_TEMPLATE,
        figsize: Tuple[float, float] = (19.2, 10.8),
    ) -> Figure:
        """Plot waffle charts on a new figure.

        Parameters
        ----------
        category : str
            Column for segment labels.
        value : str
            Column for segment sizes.
        rows : int, optional
            Number of grid rows. The default is 10.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is ``PIE_STYLE_TEMPLATE``.
        figsize : tuple[float, float], optional
            Figure size. The default is (19.2, 10.8).

        Returns
        -------
        Figure
            The new Matplotlib figure with the waffle chart.
        """
        return fplot_waffle(
            pd_df=self._obj,
            category=category,
            value=value,
            rows=rows,
            title=title,
            style=style,
            figsize=figsize,
        )

    def fplot_sankey(
        self,
        source: str,
        target: str,
        value: str,
        title: Optional[str] = None,
        style: StyleTemplate = SANKEY_STYLE_TEMPLATE,
    ) -> go.Figure:
        """Plot a Sankey diagram for flow data.

        Parameters
        ----------
        source : str
            Column for source nodes.
        target : str
            Column for target nodes.
        value : str
            Column for flow weights.
        title : str, optional
            Diagram title.
        style : StyleTemplate, optional
            Styling template. The default is ``SANKEY_STYLE_TEMPLATE``.

        Returns
        -------
        go.Figure
            The Plotly Sankey figure.
        """
        return fplot_sankey(
            pd_df=self._obj,
            source=source,
            target=target,
            value=value,
            title=title,
            style=style,
        )

    def aplot_table(
        self,
        cols: List[str],
        title: Optional[str] = None,
        style: StyleTemplate = TABLE_STYLE_TEMPLATE,
        max_values: int = 20,
        col_width: float = 0.2,
        padding: float = 0.1,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Plot a table of the DataFrame's data on Matplotlib axes.

        Parameters
        ----------
        cols : list[str]
            Columns to include in the table.
        title : str, optional
            Table title.
        style : StyleTemplate, optional
            Styling template. The default is `TABLE_STYLE_TEMPLATE`.
        max_values : int, optional
            Maximum number of rows to show. The default is 20.
        col_width : float, optional
            Width of each column. The default is 0.2.
        padding : float, optional
            Padding between table elements. The default is 0.1.
        sort_by : str, optional
            Column to sort by.
        ascending : bool, optional
            Sort order. The default is `False`.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses the current axes.

        Returns
        -------
        Axes
            The Matplotlib axes object with the plot.
        """
        return aplot_table(
            pd_df=self._obj,
            cols=cols,
            title=title,
            style=style,
            max_values=max_values,
            col_width=col_width,
            padding=padding,
            sort_by=sort_by,
            ascending=ascending,
            ax=ax,
        )

    def fplot_table(
        self,
        cols: List[str],
        title: Optional[str] = None,
        style: StyleTemplate = TABLE_STYLE_TEMPLATE,
        max_values: int = 20,
        col_width: float = 0.2,
        padding: float = 0.1,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        figsize: Tuple[float, float] = (19.2, 10.8),
    ) -> Figure:
        """Plot a table of the DataFrame's data on a new figure.

        Parameters
        ----------
        cols : list[str]
            Columns to include in the table.
        title : str, optional
            Table title.
        style : StyleTemplate, optional
            Styling template. The default is `TABLE_STYLE_TEMPLATE`.
        max_values : int, optional
            Maximum number of rows to show. The default is 20.
        col_width : float, optional
            Width of each column. The default is 0.2.
        padding : float, optional
            Padding between table elements. The default is 0.1.
        sort_by : str, optional
            Column to sort by.
        ascending : bool, optional
            Sort order. The default is `False`.
        figsize : tuple[float, float], optional
            Figure size. The default is (19.2, 10.8).

        Returns
        -------
        Figure
            The new Matplotlib figure with the table.
        """
        return fplot_table(
            pd_df=self._obj,
            cols=cols,
            title=title,
            style=style,
            max_values=max_values,
            col_width=col_width,
            padding=padding,
            sort_by=sort_by,
            ascending=ascending,
            figsize=figsize,
        )

    def aplot_timeserie(
        self,
        label: str,
        x: str,
        y: str,
        title: Optional[str] = None,
        style: StyleTemplate = TIMESERIE_STYLE_TEMPLATE,
        max_values: int = 50,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        std: bool = False,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Plot a time series on Matplotlib axes.

        Parameters
        ----------
        label : str
            Column for series labels.
        x : str
            Column for the x-axis.
        y : str
            Column for the y-axis.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is `TIMESERIE_STYLE_TEMPLATE`.
        max_values : int, optional
            Maximum number of series to show. The default is 50.
        sort_by : str, optional
            Column to sort by.
        ascending : bool, optional
            Sort order. The default is `False`.
        std : bool, optional
            If True, plot the standard deviation. The default is `False`.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses the current axes.

        Returns
        -------
        Axes
            The Matplotlib axes object with the plot.
        """
        return aplot_timeserie(
            pd_df=self._obj,
            label=label,
            x=x,
            y=y,
            title=title,
            style=style,
            max_values=max_values,
            sort_by=sort_by,
            ascending=ascending,
            std=std,
            ax=ax,
        )

    def fplot_timeserie(
        self,
        label: str,
        x: str,
        y: str,
        title: Optional[str] = None,
        style: StyleTemplate = TIMESERIE_STYLE_TEMPLATE,
        max_values: int = 50,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        std: bool = False,
        figsize: Tuple[float, float] = (19.2, 10.8),
    ) -> Figure:
        """Plot a time series on a new figure.

        Parameters
        ----------
        label : str
            Column for series labels.
        x : str
            Column for the x-axis.
        y : str
            Column for the y-axis.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is `TIMESERIE_STYLE_TEMPLATE`.
        max_values : int, optional
            Maximum number of series to show. The default is 50.
        sort_by : str, optional
            Column to sort by.
        ascending : bool, optional
            Sort order. The default is `False`.
        std : bool, optional
            If True, plot the standard deviation. The default is `False`.
        figsize : tuple[float, float], optional
            Figure size. The default is (19.2, 10.8).

        Returns
        -------
        Figure
            The new Matplotlib figure with the plot.
        """
        return fplot_timeserie(
            pd_df=self._obj,
            label=label,
            x=x,
            y=y,
            title=title,
            style=style,
            max_values=max_values,
            sort_by=sort_by,
            ascending=ascending,
            std=std,
            figsize=figsize,
        )

    def aplot_wordcloud(
        self,
        text_column: str,
        weight_column: Optional[str] = None,
        title: Optional[str] = None,
        style: StyleTemplate = WORDCLOUD_STYLE_TEMPLATE,
        max_words: int = 50,
        stopwords: Optional[List[str]] = None,
        random_state: Optional[int] = None,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Plot a word cloud on a Matplotlib axes.

        Parameters
        ----------
        text_column : str
            Column containing words or phrases.
        weight_column : str, optional
            Column containing numeric weights. The default is ``None`` for equal weights.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is `WORDCLOUD_STYLE_TEMPLATE`.
        max_words : int, optional
            Maximum number of words to display. The default is 50.
        stopwords : list[str], optional
            Words to exclude from the visualization. The default is ``None``.
        random_state : int, optional
            Seed for word placement. The default is ``None``.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses the current axes.

        Returns
        -------
        Axes
            The Matplotlib axes object with the plot.
        """
        return aplot_wordcloud(
            pd_df=self._obj,
            text_column=text_column,
            weight_column=weight_column,
            title=title,
            style=style,
            max_words=max_words,
            stopwords=stopwords,
            random_state=random_state,
            ax=ax,
        )

    def fplot_wordcloud(
        self,
        text_column: str,
        weight_column: Optional[str] = None,
        title: Optional[str] = None,
        style: StyleTemplate = WORDCLOUD_STYLE_TEMPLATE,
        max_words: int = 50,
        stopwords: Optional[List[str]] = None,
        random_state: Optional[int] = None,
        figsize: Tuple[float, float] = (19.2, 10.8),
    ) -> Figure:
        """Plot a word cloud on a new figure.

        Parameters
        ----------
        text_column : str
            Column containing words or phrases.
        weight_column : str, optional
            Column containing numeric weights. The default is ``None`` for equal weights.
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is `WORDCLOUD_STYLE_TEMPLATE`.
        max_words : int, optional
            Maximum number of words to display. The default is 50.
        stopwords : list[str], optional
            Words to exclude from the visualization. The default is ``None``.
        random_state : int, optional
            Seed for word placement. The default is ``None``.
        figsize : tuple[float, float], optional
            Figure size. The default is (19.2, 10.8).

        Returns
        -------
        Figure
            The new Matplotlib figure with the plot.
        """
        return fplot_wordcloud(
            pd_df=self._obj,
            text_column=text_column,
            weight_column=weight_column,
            title=title,
            style=style,
            max_words=max_words,
            stopwords=stopwords,
            random_state=random_state,
            figsize=figsize,
        )

    def aplot_network(
        self,
        source: str = "source",
        target: str = "target",
        weight: str = "weight",
        title: Optional[str] = None,
        style: StyleTemplate = NETWORK_STYLE_TEMPLATE,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        node_list: Optional[List] = None,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Plot a network graph on a Matplotlib axes.

        Parameters
        ----------
        source : str, optional
            Column for source nodes. The default is "source".
        target : str, optional
            Column for target nodes. The default is "target".
        weight : str, optional
            Column for edge weights. The default is "weight".
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is `NETWORK_STYLE_TEMPLATE`.
        sort_by : str, optional
            Column to sort by.
        ascending : bool, optional
            Sort order. The default is `False`.
        node_list : list, optional
            List of nodes to include. If None, all nodes are used.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses the current axes.

        Returns
        -------
        Axes
            The Matplotlib axes object with the plot.
        """
        return aplot_network(
            pd_df=self._obj,
            source=source,
            target=target,
            weight=weight,
            title=title,
            style=style,
            sort_by=sort_by,
            ascending=ascending,
            node_list=node_list,
            ax=ax,
        )

    def aplot_network_components(
        self,
        source: str = "source",
        target: str = "target",
        weight: str = "weight",
        title: Optional[str] = None,
        style: StyleTemplate = NETWORK_STYLE_TEMPLATE,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        node_list: Optional[List] = None,
        axes: Optional[np.ndarray] = None,
    ) -> None:
        """Plot connected components of a network graph on multiple axes.

        Parameters
        ----------
        source : str, optional
            Column for source nodes. The default is "source".
        target : str, optional
            Column for target nodes. The default is "target".
        weight : str, optional
            Column for edge weights. The default is "weight".
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is `NETWORK_STYLE_TEMPLATE`.
        sort_by : str, optional
            Column to sort by.
        ascending : bool, optional
            Sort order. The default is `False`.
        node_list : list, optional
            List of nodes to include. If None, all nodes are used.
        axes : np.ndarray, optional
            Numpy array of Matplotlib axes to plot on. If None, new axes are created.
        """
        aplot_network_components(
            pd_df=self._obj,
            source=source,
            target=target,
            weight=weight,
            title=title,
            style=style,
            sort_by=sort_by,
            ascending=ascending,
            node_list=node_list,
            axes=axes,
        )

    def fplot_network(
        self,
        source: str = "source",
        target: str = "target",
        weight: str = "weight",
        title: Optional[str] = None,
        style: StyleTemplate = NETWORK_STYLE_TEMPLATE,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        node_list: Optional[List] = None,
        figsize: Tuple[float, float] = (19.2, 10.8),
    ) -> Figure:
        """Plot a network graph on a new figure.

        Parameters
        ----------
        source : str, optional
            Column for source nodes. The default is "source".
        target : str, optional
            Column for target nodes. The default is "target".
        weight : str, optional
            Column for edge weights. The default is "weight".
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is `NETWORK_STYLE_TEMPLATE`.
        sort_by : str, optional
            Column to sort by.
        ascending : bool, optional
            Sort order. The default is `False`.
        node_list : list, optional
            List of nodes to include. If None, all nodes are used.
        figsize : tuple[float, float], optional
            Figure size. The default is (19.2, 10.8).

        Returns
        -------
        Figure
            The new Matplotlib figure with the plot.
        """
        return fplot_network(
            pd_df=self._obj,
            source=source,
            target=target,
            weight=weight,
            title=title,
            style=style,
            sort_by=sort_by,
            ascending=ascending,
            node_list=node_list,
            figsize=figsize,
        )

    def fplot_treemap(
        self,
        labels: str,
        parents: str,
        values: str,
        style: StyleTemplate = TREEMAP_STYLE_TEMPLATE,
        title: Optional[str] = None,
        sort_by: Optional[str] = None,
        max_values: int = 100,
        ascending: bool = False,
        fig: Optional[go.Figure] = None,
    ) -> go.Figure:
        """Plot a treemap on a new Plotly figure.

        Parameters
        ----------
        labels : str
            Column for labels.
        parents : str
            Column for parent relationships.
        values : str
            Column with values for the treemap areas.
        style : StyleTemplate, optional
            Styling template. The default is `TREEMAP_STYLE_TEMPLATE`.
        title : str, optional
            Chart title.
        sort_by : str, optional
            Column to sort by.
        max_values : int, optional
            Maximum number of items to display. The default is 100.
        ascending : bool, optional
            Sort order. The default is `False`.
        fig : go.Figure, optional
            Existing Plotly figure to add to. If None, a new figure is created.

        Returns
        -------
        go.Figure
            The Plotly figure with the treemap.
        """
        return fplot_treemap(
            pd_df=self._obj,
            labels=labels,
            parents=parents,
            values=values,
            style=style,
            title=title,
            sort_by=sort_by,
            max_values=max_values,
            ascending=ascending,
            fig=fig,
        )

    def fplot_sunburst(
        self,
        labels: str,
        parents: str,
        values: str,
        style: StyleTemplate = TREEMAP_STYLE_TEMPLATE,
        title: Optional[str] = None,
        sort_by: Optional[str] = None,
        max_values: int = 100,
        ascending: bool = False,
        fig: Optional[go.Figure] = None,
    ) -> go.Figure:
        """Plot a sunburst chart on a new Plotly figure.

        Parameters
        ----------
        labels : str
            Column for labels.
        parents : str
            Column for parent relationships.
        values : str
            Column with values for the sunburst areas.
        style : StyleTemplate, optional
            Styling template. The default is `TREEMAP_STYLE_TEMPLATE`.
        title : str, optional
            Chart title.
        sort_by : str, optional
            Column to sort by.
        max_values : int, optional
            Maximum number of items to display. The default is 100.
        ascending : bool, optional
            Sort order. The default is `False`.
        fig : go.Figure, optional
            Existing Plotly figure to add to. If None, a new figure is created.
        Returns
        -------
        go.Figure
            The Plotly figure with the sunburst chart.
        """
        return fplot_sunburst(
            pd_df=self._obj,
            labels=labels,
            parents=parents,
            values=values,
            title=title,
            style=style,
            sort_by=sort_by,
            ascending=ascending,
            max_values=max_values,
            fig=fig,
        )

    def fplot_composite_treemap(
        self,
        paths: List[str],
        values: str,
        style: StyleTemplate = TREEMAP_STYLE_TEMPLATE,
        title: Optional[str] = None,
        color: Optional[str] = None,
        sort_by: Optional[str] = None,
        max_values: int = 100,
        ascending: bool = False,
        fig: Optional[go.Figure] = None,
    ) -> Optional[go.Figure]:
        """Plot a composite treemap on a new Plotly figure.

        Parameters
        ----------
        paths : list[str]
            Columns representing the hierarchy paths for each treemap.
        values : str
            Column with values for the treemap areas.
        style : StyleTemplate, optional
            Styling template. The default is `TREEMAP_STYLE_TEMPLATE`.
        title : str, optional
            Chart title.
        color : str, optional
            Column to use for color coding.
        sort_by : str, optional
            Column to sort by.
        max_values : int, optional
            Maximum number of items to display. The default is 100.
        ascending : bool, optional
            Sort order. The default is `False`.
        fig : go.Figure, optional
            Existing Plotly figure to add to. If None, a new figure is created.

        Returns
        -------
        go.Figure, optional
            The Plotly figure with the composite treemap, or None if the input data is empty.
        """
        pd_dfs: Dict[str, pd.DataFrame] = {}
        for path in paths:
            pd_dfs[path] = self._obj

        return plot_composite_treemap(
            pd_dfs=pd_dfs,
            values=values,
            title=title,
            style=style,
            color=color,
            sort_by=sort_by,
            ascending=ascending,
            max_values=max_values,
        )


__all__ = ["DataFrameAccessor"]
