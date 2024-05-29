
import logging
import warnings
from typing import Optional, List
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
from .Style import StyleTemplate
from .Bubble import plot_bubble, BUBBLE_STYLE_TEMPLATE
from .Composite import plot_composite_bubble
from .Timeserie import plot_timeserie, TIMESERIE_STYLE_TEMPLATE
from .Table import plot_table, TABLE_STYLE_TEMPLATE
from .Network import (Graph)


warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.WARNING)


@pd.api.extensions.register_dataframe_accessor("mpl")
class MatPlotLibAccessor:

    def __init__(self, pd_df: pd.DataFrame):
        self._obj = pd_df

    def plot_bubble(self,
                    label: str,
                    x: str,
                    y: str,
                    z: str,
                    title: str = "Test",
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

    def plot_bubble_composite(self,
                              label: str,
                              x: str,
                              y: str,
                              z: str,
                              title: Optional[str] = None,
                              style: StyleTemplate = BUBBLE_STYLE_TEMPLATE,
                              max_values: int = 50,
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
                       max_values: int = 20,
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
