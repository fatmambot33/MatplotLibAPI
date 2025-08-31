"""Timeserie plotting helpers."""

from typing import Optional, Tuple, cast
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

from .StyleTemplate import (
    DynamicFuncFormatter,
    StyleTemplate,
    string_formatter,
    bmk_formatter,
    format_func,
    validate_dataframe,
    FormatterFunc,
)


TIMESERIE_STYLE_TEMPLATE = StyleTemplate(
    palette="rocket",
    format_funcs=cast(
        dict[str, Optional[FormatterFunc]],
        {"y": bmk_formatter, "label": string_formatter},
    ),
)

# region Line


def _prepare_timeserie_data(
    pd_df: pd.DataFrame, label: str, x: str, y: str, sort_by: Optional[str]
) -> pd.DataFrame:
    """Prepare data for time series plotting."""
    validate_dataframe(pd_df, cols=[label, x, y], sort_by=sort_by)
    df = pd_df[[label, x, y]].sort_values(by=[label, x])  # type: ignore
    df[x] = pd.to_datetime(df[x])
    df.set_index(x, inplace=True)
    return df


def _plot_timeserie_lines(
    ax: Axes,
    df: pd.DataFrame,
    label: str,
    x: str,
    y: str,
    std: bool,
    style: StyleTemplate,
    format_funcs: Optional[dict[str, Optional[FormatterFunc]]],
) -> None:
    """Plot the time series lines on the axes."""
    sns.set_palette(style.palette)
    # Get unique dimension_types
    label_types = df[label].unique()
    # Colors for each group
    colors = sns.color_palette(n_colors=len(label_types))

    for label_type, color in zip(label_types, colors):
        temp_df = df[df[label] == label_type].sort_values(by=x)  # type: ignore

        if format_funcs and (label_formatter := format_funcs.get("label")):
            label_type = label_formatter(label_type, None)
        if std:
            ma = temp_df[y].rolling(window=7, min_periods=1).mean()
            std_dev = temp_df[y].rolling(window=7, min_periods=1).std()
            # Calculate the last moving average value to include in the legend
            last_ma_value = cast(pd.Series, ma).iloc[-1]
            # Dynamically creating the legend label
            if format_funcs and (y_formatter := format_funcs.get(y)):
                label_str = f"{string_formatter(label_type)} (avg 7d: {y_formatter(last_ma_value, None)})"
            else:
                label_str = f"{string_formatter(label_type)} (avg 7d: {last_ma_value})"
            # Plot moving average and include the last MA value in the label for the legend
            ax.plot(temp_df.index, ma, color=color, linestyle="--", label=label_str)

            ax.fill_between(
                temp_df.index,
                ma - std_dev,
                ma + std_dev,
                color=color,
                alpha=0.2,
                label="_nolegend_",
            )
        else:
            label_str = f"{string_formatter(label_type)}"
            # Plot moving average and include the last MA value in the label for the legend
            ax.plot(temp_df.index, temp_df[y], color=color, label=label_str)


def _setup_timeserie_axes(
    ax: Axes,
    label: str,
    x: str,
    y: str,
    style: StyleTemplate,
    format_funcs: Optional[dict[str, Optional[FormatterFunc]]],
) -> None:
    """Configure the axes for the time series plot."""
    ax.legend(
        title=label,
        fontsize=style.font_size - 4,
        title_fontsize=style.font_size,
        labelcolor="linecolor",
        facecolor=style.background_color,
    )

    ax.set_xlabel(string_formatter(x), color=style.font_color)
    if format_funcs and (x_formatter := format_funcs.get("x")):
        ax.xaxis.set_major_formatter(DynamicFuncFormatter(x_formatter))
    ax.tick_params(
        axis="x",
        colors=style.font_color,
        labelrotation=45,
        labelsize=style.font_size - 4,
    )

    ax.set_ylabel(string_formatter(y), color=style.font_color)
    if format_funcs and (y_formatter := format_funcs.get("y")):
        ax.yaxis.set_major_formatter(DynamicFuncFormatter(y_formatter))
    ax.tick_params(axis="y", colors=style.font_color, labelsize=style.font_size - 4)
    ax.set_facecolor(style.background_color)
    ax.grid(True)


def aplot_timeserie(
    pd_df: pd.DataFrame,
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
    """Plot a time series on the provided axes.

    Parameters
    ----------
    pd_df : pd.DataFrame
        DataFrame containing the data to plot.
    label : str
        Column used to group series.
    x : str
        Column for the x-axis values.
    y : str
        Column for the y-axis values.
    title : str, optional
        Plot title. Default is None.
    style : StyleTemplate, optional
        Style configuration. Default is TIMESERIE_STYLE_TEMPLATE.
    max_values : int, optional
        Maximum number of rows to plot. Default is 100.
    sort_by : str, optional
        Column used to sort the data. Default is None.
    ascending : bool, optional
        Sort order for the data. Default is False.
    std : bool, optional
        Whether to plot rolling standard deviation. Default is False.
    ax : Axes, optional
        Axes to draw on. Default is None.

    Returns
    -------
    Axes
        Matplotlib axes with the time series plot.
    """
    if ax is None:
        ax = plt.gca()

    df = _prepare_timeserie_data(pd_df, label, x, y, sort_by)

    format_funcs = format_func(style.format_funcs, label=label, x=x, y=y)

    _plot_timeserie_lines(ax, df, label, x, y, std, style, format_funcs)

    _setup_timeserie_axes(ax, label, x, y, style, format_funcs)

    if title:
        ax.set_title(title, color=style.font_color, fontsize=style.font_size + 4)
    return ax


def fplot_timeserie(
    pd_df: pd.DataFrame,
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
    """Return a figure plotting the time series.

    Parameters
    ----------
    pd_df : pd.DataFrame
        DataFrame containing the data to plot.
    label : str
        Column used to group series.
    x : str
        Column for the x-axis values.
    y : str
        Column for the y-axis values.
    title : str, optional
        Plot title. Default is None.
    style : StyleTemplate, optional
        Style configuration. Default is TIMESERIE_STYLE_TEMPLATE.
    max_values : int, optional
        Maximum number of rows to plot. Default is 100.
    sort_by : str, optional
        Column used to sort the data. Default is None.
    ascending : bool, optional
        Sort order for the data. Default is False.
    std : bool, optional
        Whether to plot rolling standard deviation. Default is False.
    figsize : tuple[float, float], optional
        Size of the created figure. Default is (19.2, 10.8).

    Returns
    -------
    Figure
        Matplotlib figure containing the time series plot.
    """
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(style.background_color)
    ax = fig.add_subplot()
    ax = aplot_timeserie(
        pd_df=pd_df,
        label=label,
        x=x,
        y=y,
        title=title,
        style=style,
        max_values=max_values,
        std=std,
        sort_by=sort_by,
        ascending=ascending,
        ax=ax,
    )
    return fig


# endregion
