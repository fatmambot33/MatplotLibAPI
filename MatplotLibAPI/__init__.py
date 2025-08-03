"""Public API and pandas accessor for MatplotLibAPI."""

from .StyleTemplate import StyleTemplate
from .Bubble import aplot_bubble, fplot_bubble, BUBBLE_STYLE_TEMPLATE
from .Composite import plot_composite_bubble,plot_composite_treemap
from .Timeserie import aplot_timeserie, fplot_timeserie, TIMESERIE_STYLE_TEMPLATE
from .Table import aplot_table, fplot_table, TABLE_STYLE_TEMPLATE
from .Network import aplot_network, aplot_network_components, fplot_network, NETWORK_STYLE_TEMPLATE
from .Treemap import fplot_treemap, aplot_treemap, TREEMAP_STYLE_TEMPLATE
from typing import List, Optional, Tuple,Dict
import pandas as pd
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

    def aplot_bubble(self,
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
                     ax: Optional[Axes] = None) -> Axes:
        """Plot a bubble chart using the underlying DataFrame."""
        return aplot_bubble(pd_df=self._obj,
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
                            ax=ax)

    def fplot_bubble(self,
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
                     figsize: Tuple[float, float] = (19.2, 10.8)) -> Figure:
        """Return a figure with a bubble chart."""
        return fplot_bubble(pd_df=self._obj,
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
                            figsize=figsize)

    def fplot_composite_bubble(self,
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
                               table_rows:int=10) -> Figure:
        """Return a composite bubble chart with summary tables."""
        return plot_composite_bubble(pd_df=self._obj,
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
                                     table_rows=table_rows)

    def aplot_table(self,
                    cols: List[str],
                    title: Optional[str] = None,
                    style: StyleTemplate = TABLE_STYLE_TEMPLATE,
                    max_values: int = 20,
                    sort_by: Optional[str] = None,
                    ascending: bool = False,
                    ax: Optional[Axes] = None) -> Axes:
        """Plot a table using the DataFrame's data."""
        return aplot_table(pd_df=self._obj,
                           cols=cols,
                           title=title,
                           style=style,
                           max_values=max_values,
                           sort_by=sort_by,
                           ascending=ascending,
                           ax=ax)

    def fplot_table(self,
                    cols: List[str],
                    title: Optional[str] = None,
                    style: StyleTemplate = TABLE_STYLE_TEMPLATE,
                    max_values: int = 20,
                    sort_by: Optional[str] = None,
                    ascending: bool = False,
                    figsize: Tuple[float, float] = (19.2, 10.8)) -> Figure:
        """Return a figure containing the table plot."""
        return fplot_table(pd_df=self._obj,
                           cols=cols,
                           title=title,
                           style=style,
                           max_values=max_values,
                           sort_by=sort_by,
                           ascending=ascending,
                           figsize=figsize)

    def aplot_timeserie(self,
                        label: str,
                        x: str,
                        y: str,
                        title: Optional[str] = None,
                        style: StyleTemplate = TIMESERIE_STYLE_TEMPLATE,
                        max_values: int = 100,
                        sort_by: Optional[str] = None,
                        ascending: bool = False,
                        std: bool = False,
                        ax: Optional[Axes] = None) -> Axes:
        """Plot a time series using the stored DataFrame."""
        return aplot_timeserie(pd_df=self._obj,
                               label=label,
                               x=x,
                               y=y,
                               title=title,
                               style=style,
                               max_values=max_values,
                               sort_by=sort_by,
                               ascending=ascending,
                               std=std,
                               ax=ax)

    def fplot_timeserie(self,
                        label: str,
                        x: str,
                        y: str,
                        title: Optional[str] = None,
                        style: StyleTemplate = TIMESERIE_STYLE_TEMPLATE,
                        max_values: int = 100,
                        sort_by: Optional[str] = None,
                        ascending: bool = False,
                        std: bool = False,
                        figsize: Tuple[float, float] = (19.2, 10.8)) -> Figure:
        """Return a figure plotting the time series."""
        return fplot_timeserie(pd_df=self._obj,
                               label=label,
                               x=x,
                               y=y,
                               title=title,
                               style=style,
                               max_values=max_values,
                               sort_by=sort_by,
                               ascending=ascending,
                               std=std,
                               figsize=figsize)

    def aplot_network(self,
                      source: str = "source",
                      target: str = "target",
                      weight: str = "weight",
                      title: Optional[str] = None,
                      style: StyleTemplate = NETWORK_STYLE_TEMPLATE,
                      sort_by: Optional[str] = None,
                      ascending: bool = False,
                      node_list: Optional[List] = None,
                      ax: Optional[Axes] = None) -> Axes:
        """Plot a network graph on the given axes."""
        return aplot_network(pd_df=self._obj,
                             source=source,
                             target=target,
                             weight=weight,
                             title=title,
                             style=style,
                             sort_by=sort_by,
                             ascending=ascending,
                             node_list=node_list,
                             ax=ax)

    def aplot_network_components(self,
                                 source: str = "source",
                                 target: str = "target",
                                 weight: str = "weight",
                                 title: Optional[str] = None,
                                style: StyleTemplate = NETWORK_STYLE_TEMPLATE,
                                sort_by: Optional[str] = None,
                                ascending: bool = False,
                                node_list: Optional[List] = None,
                                ax: Optional[List[Axes]] = None) -> List[Axes]:
        """Plot connected components of the network graph."""
        return aplot_network_components(pd_df=self._obj,
                                        source=source,
                                        target=target,
                                        weight=weight,
                                        title=title,
                                        style=style,
                                        sort_by=sort_by,
                                        ascending=ascending,
                                        node_list=node_list,
                                        ax=ax)

    def fplot_network(self,
                      source: str = "source",
                      target: str = "target",
                      weight: str = "weight",
                      title: Optional[str] = None,
                      style: StyleTemplate = NETWORK_STYLE_TEMPLATE,
                      sort_by: Optional[str] = None,
                      ascending: bool = False,
                      node_list: Optional[List] = None) -> Figure:
        """Return a network plot as a new figure."""
        return fplot_network(pd_df=self._obj,
                             source=source,
                             target=target,
                             weight=weight,
                             title=title,
                             style=style,
                             sort_by=sort_by,
                             ascending=ascending,
                             node_list=node_list)

    def fplot_treemap(self,
                      path: str,
                      values: str,
                      style: StyleTemplate = TREEMAP_STYLE_TEMPLATE,
                      title: Optional[str] = None,
                      color: Optional[str] = None,
                      sort_by: Optional[str] = None,
                      max_values: int = 100,
                      ascending: bool = False,
                      fig: Optional[go.Figure] = None) -> go.Figure:
        """Return a figure with a treemap plot."""
        return fplot_treemap(pd_df=self._obj,
                             path=path,
                             values=values,
                             title=title,
                             style=style,
                             color=color,
                             sort_by=sort_by,
                             ascending=ascending,
                             max_values=max_values,
                             fig=fig)

    def aplot_treemap(self,
                      path: str,
                      values: str,
                      style: StyleTemplate = TREEMAP_STYLE_TEMPLATE,
                      title: Optional[str] = None,
                      color: Optional[str] = None,
                      sort_by: Optional[str] = None,
                      max_values: int = 100,
                      ascending: bool = False,
                      fig: Optional[go.Figure] = None) -> go.Trace:
        """Add a treemap trace to an existing figure or create a new one."""
        return aplot_treemap(pd_df=self._obj,
                             path=path,
                             values=values,
                             title=title,
                             style=style,
                             color=color,
                             sort_by=sort_by,
                             ascending=ascending,
                             max_values=max_values)

    def fplot_composite_treemap(self,
                      pathes: List[str],
                      values: str,
                      style: StyleTemplate = TREEMAP_STYLE_TEMPLATE,
                      title: Optional[str] = None,
                      color: Optional[str] = None,
                      sort_by: Optional[str] = None,
                      max_values: int = 100,
                      ascending: bool = False,
                      fig: Optional[go.Figure] = None) -> Optional[go.Figure]:
        """Return a figure composed of multiple treemap plots."""
        pd_dfs: Dict[str, pd.DataFrame] = {}
        for path in pathes:
            pd_dfs[path] = self._obj

        return plot_composite_treemap(pd_dfs=pd_dfs,
                             values=values,
                             title=title,
                             style=style,
                             color=color,
                             sort_by=sort_by,
                             ascending=ascending,
                             max_values=max_values)


__all__ = [
    "aplot_bubble",
    "aplot_timeserie",
    "aplot_table",
    "aplot_network",
    "aplot_network_components",
    "fplot_network",
    "fplot_treemap",
    "aplot_treemap",
    "plot_composite_bubble",
    "plot_composite_treemap",
    "StyleTemplate",
    "DataFrameAccessor",
]
