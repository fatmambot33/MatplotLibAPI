
import logging
import warnings
from typing import Optional, List
from matplotlib.axes import Axes
import pandas as pd
from .Utils import (StyleTemplate, BUBBLE_STYLE_TEMPLATE,TIMESERIE_STYLE_TEMPLATE,TABLE_STYLE_TEMPLATE)
from .Bubble import (plot_bubble)
from .Timeserie import (plot_timeserie)
from .Table import (plot_table)




warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.WARNING)


@pd.api.extensions.register_dataframe_accessor("mpl")
class MatPlotLibAccessor:

    def __init__(self, pd_df: pd.DataFrame):
        self._obj = pd_df

    @staticmethod
    def _validate(pd_df: pd.DataFrame,
                  cols: List[str]):
        for col in cols:
            if col not in pd_df.columns:
                raise AttributeError(f"{col} is not a DataFrame's column")

    def plot_bubble(self,
                    label: str,
                    x: str,
                    y: str,
                    z: str,
                    title: str = "Test",
                    style:StyleTemplate = BUBBLE_STYLE_TEMPLATE,
                    max_values: int = 50,
                    center_to_mean: bool = False)->Axes:

        MatPlotLibAccessor._validate(pd_df=self._obj, cols=[label, x, y, z])
        return plot_bubble(pd_df=self._obj,
                          label=label,
                          x=x,
                          y=y,
                          z=z,
                          title=title,
                          style=style,
                          max_values=max_values,
                          center_to_mean=center_to_mean)

    def plot_table(self,
               cols: List[str],
               title: str = "test",
               style:StyleTemplate=TABLE_STYLE_TEMPLATE,
               max_values:int=20,
               sort_by: Optional[str] = None,
               ascending:bool=False):

        MatPlotLibAccessor._validate(pd_df=self._obj,
                                     cols=cols)
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
                       title: str = "Test",
                       style: StyleTemplate = TIMESERIE_STYLE_TEMPLATE):

        MatPlotLibAccessor._validate(pd_df=self._obj, cols=[label, x, y])
        return plot_timeserie(pd_df=self._obj,
                              label=label,
                              x=x,
                              y=y,
                              title=title,
                              style=style)
    
    