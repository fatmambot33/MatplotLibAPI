
from typing import List,Optional
import pandas as pd

def validate_dataframe(pd_df: pd.DataFrame,
                    cols: List[str],
                    sort_by: Optional[str] = None):
    _columns = cols.copy()
    if sort_by and sort_by not in _columns:
        _columns.append(sort_by)
    for col in _columns:
        if col not in pd_df.columns:
            raise AttributeError(f"{col} is not a DataFrame's column")
        
from .Table import plot_table
from .Timeserie import plot_timeserie
from .Bubble import plot_bubble
from .Network import plot_network, plot_network_components
from .Pivot import plot_pivotbar
from .Treemap import plot_treemap
from .Composite import plot_composite_bubble
from .Style import StyleTemplate


from pandas.api.extensions import register_dataframe_accessor
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import plotly.graph_objects as go
from .Style import StyleTemplate
from .Bubble import plot_bubble, BUBBLE_STYLE_TEMPLATE
from .Composite import plot_composite_bubble
from .Timeserie import plot_timeserie, TIMESERIE_STYLE_TEMPLATE
from .Table import plot_table, TABLE_STYLE_TEMPLATE
from .Network import Graph
from .Treemap import plot_treemap, TREEMAP_STYLE_TEMPLATE

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
                    ascending: bool = False) -> Axes:

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
                           ascending=ascending)

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
                              ascending: bool = False) -> Figure:

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
                                     ascending=ascending)

    def plot_table(self,
                   cols: List[str],
                   title: Optional[str] = None,
                   style: StyleTemplate = TABLE_STYLE_TEMPLATE,
                   max_values: int = 20,
                   sort_by: Optional[str] = None,
                   ascending: bool = False) -> Axes:

        return plot_table(pd_df=self._obj,
                          cols=cols,
                          title=title,
                          style=style,
                          max_values=max_values,
                          sort_by=sort_by,
                          ascending=ascending)

    def plot_timeserie(self,
                       label: str,
                       x: str,
                       y: str,
                       title: Optional[str] = None,
                       style: StyleTemplate = TIMESERIE_STYLE_TEMPLATE,
                       max_values: int = 100,
                       sort_by: Optional[str] = None,
                       ascending: bool = False) -> Axes:

        return plot_timeserie(pd_df=self._obj,
                              label=label,
                              x=x,
                              y=y,
                              title=title,
                              style=style,
                              max_values=max_values,
                              sort_by=sort_by,
                              ascending=ascending)

    def plot_network(self,
                     source: str = "source",
                     target: str = "target",
                     weight: str = "weight",
                     title: Optional[str] = None,
                     style: StyleTemplate = TIMESERIE_STYLE_TEMPLATE,
                     max_values: int = 20,
                     sort_by: Optional[str] = None,
                     ascending: bool = False) -> Axes:

        graph = Graph.from_pandas_edgelist(df=self._obj,
                                           source=source,
                                           target=target,
                                           weight=weight)

        return graph.plotX(title, style)
    
    def plot_treemap(self,
                 path: str,
                 values: str,
                 style: StyleTemplate = TREEMAP_STYLE_TEMPLATE,
                 title: Optional[str] = None,
                 color: Optional[str] = None, 
                 max_values: int = 100,
                 sort_by: Optional[str] = None,
                 ascending: bool = False) ->go.Figure:
        return plot_treemap(pd_df=self._obj,
                              path=path,
                              values=values,
                              title=title,
                              style=style,
                              color=color,
                              max_values=max_values,
                              sort_by=sort_by,
                              ascending=ascending)

__all__ = ["validate_dataframe","plot_bubble", "plot_timeserie", "plot_table", "plot_network","plot_network_components",
           "plot_pivotbar", "plot_treemap", "plot_composite_bubble", "StyleTemplate", "DataFrameAccessor"]

