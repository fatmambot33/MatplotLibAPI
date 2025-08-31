"""Public API and pandas accessor for MatplotLibAPI."""

from .StyleTemplate import StyleTemplate
from .Bubble import aplot_bubble, fplot_bubble, BUBBLE_STYLE_TEMPLATE
from .Composite import plot_composite_bubble, plot_composite_treemap
from .Timeserie import aplot_timeserie, fplot_timeserie, TIMESERIE_STYLE_TEMPLATE
from .Table import aplot_table, fplot_table, TABLE_STYLE_TEMPLATE
from .Network import (
    aplot_network,
    aplot_network_components,
    fplot_network,
    NETWORK_STYLE_TEMPLATE,
)
from .Treemap import fplot_treemap, aplot_treemap, TREEMAP_STYLE_TEMPLATE
from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
from pandas.api.extensions import register_dataframe_accessor

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import plotly.graph_objects as go


@register_dataframe_accessor("mpl")
class DataFrameAccessor:
    """Expose MatplotLibAPI plotting helpers as a pandas accessor."""

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
            Chart title. Default is None.
        style : StyleTemplate, optional
            Styling template. Default is BUBBLE_STYLE_TEMPLATE.
        max_values : int, optional
            Maximum number of bubbles to plot. Default is 50.
        center_to_mean : bool, optional
            If True, center x-axis values around the mean. Default is False.
        sort_by : str, optional
            Column to sort by before plotting. Default is None.
        ascending : bool, optional
            Sort order. Default is False.
        hline : bool, optional
            If True, draw a horizontal line at the mean of y-values. Default is False.
        vline : bool, optional
            If True, draw a vertical line at the mean of x-values. Default is False.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses the current axes. Default is None.

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
            Chart title. Default is None.
        style : StyleTemplate, optional
            Styling template. Default is BUBBLE_STYLE_TEMPLATE.
        max_values : int, optional
            Maximum number of bubbles to plot. Default is 50.
        center_to_mean : bool, optional
            If True, center x-axis values around the mean. Default is False.
        sort_by : str, optional
            Column to sort by before plotting. Default is None.
        ascending : bool, optional
            Sort order. Default is False.
        hline : bool, optional
            If True, draw a horizontal line at the mean of y-values. Default is False.
        vline : bool, optional
            If True, draw a vertical line at the mean of x-values. Default is False.
        figsize : tuple[float, float], optional
            Figure size. Default is (19.2, 10.8).

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
            Chart title. Default is None.
        style : StyleTemplate, optional
            Styling template. Default is BUBBLE_STYLE_TEMPLATE.
        max_values : int, optional
            Maximum number of bubbles to plot. Default is 100.
        center_to_mean : bool, optional
            If True, center x-axis values around the mean. Default is False.
        sort_by : str, optional
            Column to sort by before plotting. Default is None.
        ascending : bool, optional
            Sort order. Default is False.
        table_rows : int, optional
            Number of rows for the summary tables. Default is 10.

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
            Table title. Default is None.
        style : StyleTemplate, optional
            Styling template. Default is TABLE_STYLE_TEMPLATE.
        max_values : int, optional
            Maximum number of rows to display. Default is 20.
        sort_by : str, optional
            Column to sort by. Default is None.
        ascending : bool, optional
            Sort order. Default is False.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses the current axes. Default is None.

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
            Table title. Default is None.
        style : StyleTemplate, optional
            Styling template. Default is TABLE_STYLE_TEMPLATE.
        max_values : int, optional
            Maximum number of rows to display. Default is 20.
        sort_by : str, optional
            Column to sort by. Default is None.
        ascending : bool, optional
            Sort order. Default is False.
        figsize : tuple[float, float], optional
            Figure size. Default is (19.2, 10.8).

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
            Chart title. Default is None.
        style : StyleTemplate, optional
            Styling template. Default is TIMESERIE_STYLE_TEMPLATE.
        max_values : int, optional
            Maximum number of data points to plot. Default is 100.
        sort_by : str, optional
            Column to sort by. Default is None.
        ascending : bool, optional
            Sort order. Default is False.
        std : bool, optional
            If True, plot the standard deviation. Default is False.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses the current axes. Default is None.

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
            Chart title. Default is None.
        style : StyleTemplate, optional
            Styling template. Default is TIMESERIE_STYLE_TEMPLATE.
        max_values : int, optional
            Maximum number of data points to plot. Default is 100.
        sort_by : str, optional
            Column to sort by. Default is None.
        ascending : bool, optional
            Sort order. Default is False.
        std : bool, optional
            If True, plot the standard deviation. Default is False.
        figsize : tuple[float, float], optional
            Figure size. Default is (19.2, 10.8).

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
            Column for source nodes. Default is "source".
        target : str, optional
            Column for target nodes. Default is "target".
        weight : str, optional
            Column for edge weights. Default is "weight".
        title : str, optional
            Chart title. Default is None.
        style : StyleTemplate, optional
            Styling template. Default is NETWORK_STYLE_TEMPLATE.
        sort_by : str, optional
            Column to sort by. Default is None.
        ascending : bool, optional
            Sort order. Default is False.
        node_list : list, optional
            List of nodes to include. If None, all nodes are used. Default is None.
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses the current axes. Default is None.

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
            Column for source nodes. Default is "source".
        target : str, optional
            Column for target nodes. Default is "target".
        weight : str, optional
            Column for edge weights. Default is "weight".
        title : str, optional
            Chart title. Default is None.
        style : StyleTemplate, optional
            Styling template. Default is NETWORK_STYLE_TEMPLATE.
        sort_by : str, optional
            Column to sort by. Default is None.
        ascending : bool, optional
            Sort order. Default is False.
        node_list : list, optional
            List of nodes to include. If None, all nodes are used. Default is None.
        axes : np.ndarray, optional
            Numpy array of Matplotlib axes to plot on. If None, new axes are created. Default is None.
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
            Column for source nodes. Default is "source".
        target : str, optional
            Column for target nodes. Default is "target".
        weight : str, optional
            Column for edge weights. Default is "weight".
        title : str, optional
            Chart title. Default is None.
        style : StyleTemplate, optional
            Styling template. Default is NETWORK_STYLE_TEMPLATE.
        sort_by : str, optional
            Column to sort by. Default is None.
        ascending : bool, optional
            Sort order. Default is False.
        node_list : list, optional
            List of nodes to include. If None, all nodes are used. Default is None.
        figsize : tuple[float, float], optional
            Figure size. Default is (19.2, 10.8).

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
            Styling template. Default is TREEMAP_STYLE_TEMPLATE.
        title : str, optional
            Chart title. Default is None.
        color : str, optional
            Column to use for color coding. Default is None.
        sort_by : str, optional
            Column to sort by. Default is None.
        max_values : int, optional
            Maximum number of items to display. Default is 100.
        ascending : bool, optional
            Sort order. Default is False.
        fig : go.Figure, optional
            Existing Plotly figure to add to. If None, a new figure is created. Default is None.

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
        pathes: List[str],
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
        pathes : list[str]
            List of columns representing the hierarchy paths for each treemap.
        values : str
            Column with values for the treemap areas.
        style : StyleTemplate, optional
            Styling template. Default is TREEMAP_STYLE_TEMPLATE.
        title : str, optional
            Chart title. Default is None.
        color : str, optional
            Column to use for color coding. Default is None.
        sort_by : str, optional
            Column to sort by. Default is None.
        max_values : int, optional
            Maximum number of items to display. Default is 100.
        ascending : bool, optional
            Sort order. Default is False.
        fig : go.Figure, optional
            Existing Plotly figure to add to. If None, a new figure is created. Default is None.

        Returns
        -------
        go.Figure | None
            The Plotly figure with the composite treemap, or None if the input data is empty.
        """
        pd_dfs: Dict[str, pd.DataFrame] = {}
        for path in pathes:
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
    "aplot_bubble",
    "aplot_timeserie",
    "aplot_table",
    "aplot_network",
    "aplot_network_components",
    "fplot_network",
    "fplot_treemap",
    "plot_composite_bubble",
    "plot_composite_treemap",
    "StyleTemplate",
    "DataFrameAccessor",
]
