"""Public API and pandas accessor for MatplotLibAPI."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas.api.extensions import register_dataframe_accessor

from .Bubble import BUBBLE_STYLE_TEMPLATE, aplot_bubble, fplot_bubble
from .Composite import plot_composite_bubble, plot_composite_treemap
from .Network import (
    NETWORK_STYLE_TEMPLATE,
    aplot_network,
    aplot_network_components,
    fplot_network,
)
from .StyleTemplate import StyleTemplate
from .Table import TABLE_STYLE_TEMPLATE, aplot_table, fplot_table
from .Timeserie import TIMESERIE_STYLE_TEMPLATE, aplot_timeserie, fplot_timeserie
from .Treemap import TREEMAP_STYLE_TEMPLATE, aplot_treemap, fplot_treemap
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

    def aplot_table(
        self,
        cols: List[str],
        title: Optional[str] = None,
        style: StyleTemplate = TABLE_STYLE_TEMPLATE,
        max_values: int = 20,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Plot a table of the DataFrame's data on a Matplotlib axes.

        Parameters
        ----------
        cols : list[str]
            List of columns to include in the table.
        title : str, optional
            Table title.
        style : StyleTemplate, optional
            Styling template. The default is `TABLE_STYLE_TEMPLATE`.
        max_values : int, optional
            Maximum number of rows to display. The default is 20.
        sort_by : str, optional
            Column to sort by.
        ascending : bool, optional
            Sort order. The default is `False`.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses the current axes.

        Returns
        -------
        Axes
            The Matplotlib axes object with the table.
        """
        return aplot_table(
            pd_df=self._obj,
            cols=cols,
            title=title,
            style=style,
            max_values=max_values,
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
        sort_by: Optional[str] = None,
        ascending: bool = False,
        figsize: Tuple[float, float] = (19.2, 10.8),
    ) -> Figure:
        """Plot a table of the DataFrame's data on a new figure.

        Parameters
        ----------
        cols : list[str]
            List of columns to include in the table.
        title : str, optional
            Table title.
        style : StyleTemplate, optional
            Styling template. The default is `TABLE_STYLE_TEMPLATE`.
        max_values : int, optional
            Maximum number of rows to display. The default is 20.
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
        max_values: int = 100,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        std: bool = False,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Plot a time series on a Matplotlib axes.

        Parameters
        ----------
        label : str
            Column to use for the series label.
        x : str
            Column to use for the x-axis (time).
        y : str
            Column to use for the y-axis (values).
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is `TIMESERIE_STYLE_TEMPLATE`.
        max_values : int, optional
            Maximum number of data points to plot. The default is 100.
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
        max_values: int = 100,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        std: bool = False,
        figsize: Tuple[float, float] = (19.2, 10.8),
    ) -> Figure:
        """Plot a time series on a new figure.

        Parameters
        ----------
        label : str
            Column to use for the series label.
        x : str
            Column to use for the x-axis (time).
        y : str
            Column to use for the y-axis (values).
        title : str, optional
            Chart title.
        style : StyleTemplate, optional
            Styling template. The default is `TIMESERIE_STYLE_TEMPLATE`.
        max_values : int, optional
            Maximum number of data points to plot. The default is 100.
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
        )

    def fplot_treemap(
        self,
        path: str,
        values: str,
        style: StyleTemplate = TREEMAP_STYLE_TEMPLATE,
        title: Optional[str] = None,
        color: Optional[str] = None,
        sort_by: Optional[str] = None,
        max_values: int = 100,
        ascending: bool = False,
        fig: Optional[go.Figure] = None,
    ) -> go.Figure:
        """Plot a treemap on a new Plotly figure.

        Parameters
        ----------
        path : str
            Column representing the hierarchy path.
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
        go.Figure
            The Plotly figure with the treemap.
        """
        return fplot_treemap(
            pd_df=self._obj,
            path=path,
            values=values,
            title=title,
            style=style,
            color=color,
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


__all__ = [
    "DataFrameAccessor",
    "StyleTemplate",
    "aplot_bubble",
    "aplot_network",
    "aplot_network_components",
    "aplot_table",
    "aplot_timeserie",
    "aplot_wordcloud",
    "fplot_bubble",
    "fplot_network",
    "fplot_table",
    "fplot_timeserie",
    "fplot_wordcloud",
    "fplot_treemap",
    "plot_composite_bubble",
    "plot_composite_treemap",
]
