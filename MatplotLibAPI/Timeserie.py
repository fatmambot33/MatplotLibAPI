# Hint for Visual Code Python Interactive window
# %%
from typing import Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

from .StyleTemplate import DynamicFuncFormatter, StyleTemplate, string_formatter, bmk_formatter, format_func, validate_dataframe


TIMESERIE_STYLE_TEMPLATE = StyleTemplate(
    palette='rocket',
    format_funcs={"y": bmk_formatter, "label": string_formatter}
)

# region Line


def aplot_timeserie(pd_df: pd.DataFrame,
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

    validate_dataframe(pd_df, cols=[label, x, y], sort_by=sort_by)
    style.format_funcs = format_func(style.format_funcs, label=label, x=x, y=y)

    df = pd_df[[label, x, y]].sort_values(by=[label, x])
    df[x] = pd.to_datetime(df[x])
    df.set_index(x, inplace=True)

    sns.set_palette(style.palette)
    # Colors for each group
    colors = sns.color_palette(n_colors=len(df.columns))
    if ax is None:
        ax = plt.gca()

    # Get unique dimension_types
    label_types = df[label].unique()

    # Colors for each group
    colors = sns.color_palette(n_colors=len(label_types))

    for label_type, color in zip(label_types, colors):
        temp_df = df[df[label] == label_type].sort_values(by=x)

        if style.format_funcs.get("label"):
            label_type = style.format_funcs.get("label")(label_type)
        if std:
            ma = temp_df[y].rolling(window=7, min_periods=1).mean()
            std_dev = temp_df[y].rolling(window=7, min_periods=1).std()
            # Calculate the last moving average value to include in the legend
            last_ma_value = ma.iloc[-1]
            # Dynamically creating the legend label
            label_str = f"{string_formatter(label_type)} (avg 7d: {style.format_funcs[y](last_ma_value)})"
            # Plot moving average and include the last MA value in the label for the legend
            ax.plot(temp_df.index, ma, color=color,
                    linestyle='--', label=label_str)

            ax.fill_between(temp_df.index, ma - std_dev, ma +
                            std_dev, color=color, alpha=0.2, label='_nolegend_')
        else:
            label_str = f"{string_formatter(label_type)}"
            # Plot moving average and include the last MA value in the label for the legend
            ax.plot(temp_df.index, temp_df[y], color=color, label=label_str)

    ax.legend(
        title=label,
        fontsize=style.font_size-4,
        title_fontsize=style.font_size,
        labelcolor='linecolor',
        facecolor=style.background_color)

    ax.set_xlabel(string_formatter(x), color=style.font_color)
    if style.format_funcs.get("x"):
        ax.xaxis.set_major_formatter(
            DynamicFuncFormatter(style.format_funcs.get("x")))
    ax.tick_params(axis='x', colors=style.font_color,
                   labelrotation=45, labelsize=style.font_size-4)

    ax.set_ylabel(string_formatter(y), color=style.font_color)
    if style.format_funcs.get("y"):
        ax.yaxis.set_major_formatter(
            DynamicFuncFormatter(style.format_funcs.get("y")))
    ax.tick_params(axis='y', colors=style.font_color,
                   labelsize=style.font_size-4)
    ax.set_facecolor(style.background_color)
    ax.grid(True)
    if title:
        ax.set_title(title, color=style.font_color, fontsize=style.font_size+4)
    return ax


def fplot_timeserie(pd_df: pd.DataFrame,
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
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(style.background_color)
    ax = fig.add_subplot()
    ax = aplot_timeserie(pd_df=pd_df,
                           label=label,
                           x=x,
                           y=y,
                           title=title,
                           style=style,
                           max_values=max_values,
                           std=std,
                           sort_by=sort_by,
                           ascending=ascending,
                           ax=ax)
    return fig
# endregion
