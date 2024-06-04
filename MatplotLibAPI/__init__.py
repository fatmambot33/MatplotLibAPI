
from .StyleTemplate import StyleTemplate
from .Bubble import plot_bubble, BUBBLE_STYLE_TEMPLATE
from .Composite import plot_composite_bubble
from .Timeserie import plot_timeserie, TIMESERIE_STYLE_TEMPLATE
from .Table import plot_table, TABLE_STYLE_TEMPLATE
from .Network import plot_network, plot_network_components, NETWORK_STYLE_TEMPLATE
from .Treemap import plot_treemap, TREEMAP_STYLE_TEMPLATE
from typing import List, Optional
import pandas as pd
from pandas.api.extensions import register_dataframe_accessor

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import plotly.graph_objects as go


@register_dataframe_accessor("mpl")
class DataFrameAccessor:

    def __init__(self, pd_df: pd.DataFrame):
        self._obj = pd_df

    def plot_bubble(self,
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
                    ax: Optional[Axes] = None) -> Axes:

        return plot_bubble(pd_df=self._obj,
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
                           ax=ax)

    def plot_composite_bubble(self,
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
                              ax: Optional[Axes] = None) -> Figure:

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
                                     ax=ax)

    def plot_table(self,
                   cols: List[str],
                   title: Optional[str] = None,
                   style: StyleTemplate = TABLE_STYLE_TEMPLATE,
                   max_values: int = 20,
                   sort_by: Optional[str] = None,
                   ascending: bool = False,
                   ax: Optional[Axes] = None) -> Axes:

        return plot_table(pd_df=self._obj,
                          cols=cols,
                          title=title,
                          style=style,
                          max_values=max_values,
                          sort_by=sort_by,
                          ascending=ascending,
                          ax=ax)

    def plot_timeserie(self,
                       label: str,
                       x: str,
                       y: str,
                       title: Optional[str] = None,
                       style: StyleTemplate = TIMESERIE_STYLE_TEMPLATE,
                       max_values: int = 100,
                       sort_by: Optional[str] = None,
                       ascending: bool = False,
                       ax: Optional[Axes] = None) -> Axes:

        return plot_timeserie(pd_df=self._obj,
                              label=label,
                              x=x,
                              y=y,
                              title=title,
                              style=style,
                              max_values=max_values,
                              sort_by=sort_by,
                              ascending=ascending,
                              ax=ax)

    def plot_network(self,
                     source: str = "source",
                     target: str = "target",
                     weight: str = "weight",
                     title: Optional[str] = None,
                     style: StyleTemplate = NETWORK_STYLE_TEMPLATE,
                     sort_by: Optional[str] = None,
                     ascending: bool = False,
                     node_list: Optional[List] = None,
                     ax: Optional[Axes] = None) -> Axes:

        return plot_network(df=self._obj,
                            source=source,
                            target=target,
                            weight=weight,
                            title=title,
                            style=style,
                            sort_by=sort_by,
                            ascending=ascending,
                            node_list=node_list,
                            ax=ax)

    def plot_network_components(self,
                                source: str = "source",
                                target: str = "target",
                                weight: str = "weight",
                                title: Optional[str] = None,
                                style: StyleTemplate = NETWORK_STYLE_TEMPLATE,
                                sort_by: Optional[str] = None,
                                ascending: bool = False,
                                node_list: Optional[List] = None,
                                ax: Optional[Axes] = None) -> Axes:

        return plot_network_components(df=self._obj,
                                       source=source,
                                       target=target,
                                       weight=weight,
                                       title=title,
                                       style=style,
                                       sort_by=sort_by,
                                       ascending=ascending,
                                       node_list=node_list,
                                       ax=ax)

    def plot_treemap(self,
                     path: str,
                     values: str,
                     style: StyleTemplate = TREEMAP_STYLE_TEMPLATE,
                     title: Optional[str] = None,
                     color: Optional[str] = None,
                     max_values: int = 100,
                     sort_by: Optional[str] = None,
                     ascending: bool = False) -> go.Figure:
        return plot_treemap(pd_df=self._obj,
                            path=path,
                            values=values,
                            title=title,
                            style=style,
                            color=color,
                            max_values=max_values,
                            sort_by=sort_by,
                            ascending=ascending)


__all__ = ["validate_dataframe", "plot_bubble", "plot_timeserie", "plot_table", "plot_network", "plot_network_components",
           "plot_pivotbar", "plot_treemap", "plot_composite_bubble", "StyleTemplate", "DataFrameAccessor"]
