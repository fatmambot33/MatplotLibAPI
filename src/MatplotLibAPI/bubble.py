"""Bubble chart plotting helpers.

Provides a Bubble class to create and render bubble charts using seaborn and matplotlib,
with customizable styling via `StyleTemplate`.
"""

from typing import Any, Dict, Optional, Tuple, cast

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.extensions import register_dataframe_accessor
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator

from .base_plot import BasePlot

from .style_template import (
    BUBBLE_STYLE_TEMPLATE,
    FIG_SIZE,
    MAX_RESULTS,
    TITLE_SCALE_FACTOR,
    StyleTemplate,
    format_func,
    generate_ticks,
    validate_dataframe,
    DynamicFuncFormatter,
    FormatterFunc,
)

__all__ = ["BUBBLE_STYLE_TEMPLATE", "Bubble"]


@register_dataframe_accessor("bubble")
class Bubble(BasePlot):
    """Bubble chart plot implementing the BasePlot interface.

    This class provides methods to plot bubble charts on existing or new
    Matplotlib figures with customizable styling.

    Methods
    -------
    aplot
        Plot a bubble chart on existing Matplotlib axes.
    fplot
        Plot a bubble chart on a new Matplotlib figure.
    """

    def __init__(
        self,
        pd_df: pd.DataFrame,
        label: str,
        x: str,
        y: str,
        z: str,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        max_values: int = MAX_RESULTS,
        center_to_mean: bool = False,
    ):
        """Initialize the Bubble plot accessor."""
        plot_df = self._prepare_data(
            pd_df,
            label=label,
            x=x,
            y=y,
            z=z,
            sort_by=sort_by,
            ascending=ascending,
            max_values=max_values,
            center_to_mean=center_to_mean,
        )
        self.x = x
        self.y = y
        self.z = z
        self.label = label
        super().__init__(plot_df)

    @staticmethod
    def _prepare_data(
        pd_df: pd.DataFrame,
        label: str,
        x: str,
        y: str,
        z: str,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        max_values: int = MAX_RESULTS,
        center_to_mean: bool = False,
    ) -> pd.DataFrame:
        """Prepare data for bubble chart.

        Parameters
        ----------
        pd_df : pd.DataFrame
            Input DataFrame.
        label : str
            Column name for bubble labels.
        x : str
            Column name for x-axis values.
        y : str
            Column name for y-axis values.
        z : str
            Column name for bubble sizes.
        sort_by : Optional[str]
            Column to sort by.
        ascending : bool
            Sort order.
        max_values : int
            Maximum number of bubbles to display.
        center_to_mean : bool
            Whether to center x-axis values around the mean.


        Returns
        -------
        pd.DataFrame
            Prepared DataFrame for plotting.

        Raises
        ------
        AttributeError
            If required columns are missing from the DataFrame.
        """
        validate_dataframe(pd_df, cols=[label, x, y, z], sort_by=sort_by)
        sort_col = sort_by or z

        plot_df = (
            pd_df[[label, x, y, z]]
            .sort_values(by=[sort_col], ascending=ascending)  # type: ignore
            .head(max_values)
            .copy()
        )

        if center_to_mean:
            plot_df[x] -= plot_df[x].mean()

        plot_df["quintile"] = pd.qcut(plot_df[z], 5, labels=False, duplicates="drop")
        plot_df[f"{x}_mean"] = plot_df[x].mean()
        plot_df[f"{y}_mean"] = plot_df[y].mean()
        plot_df[f"{z}_mean"] = plot_df[z].mean()
        return plot_df

    @staticmethod
    def _setup_axes(
        ax: Axes,
        style: StyleTemplate,
        pd_df: pd.DataFrame,
        x: str,
        y: str,
        format_funcs: Optional[Dict[str, Optional[FormatterFunc]]],
    ) -> None:
        """Configure axes for the bubble chart.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object.
        style : StyleTemplate
            Styling for the plot.
        pd_df : pd.DataFrame
            DataFrame used for plotting.
        x : str
            Column name for x-axis values.
        y : str
            Column name for y-axis values.
        format_funcs : Optional[Dict[str, Optional[FormatterFunc]]]
            Functions to format axis tick labels.
        """
        ax.set_facecolor(style.background_color)

        if style.xscale:
            ax.set(xscale=style.xscale)
        if style.yscale:
            ax.set(yscale=style.yscale)

        # X-axis ticks and formatting
        x_min, x_max = cast(float, pd_df[x].min()), cast(float, pd_df[x].max())
        ax.xaxis.set_ticks(
            generate_ticks(
                x_min, x_max, num_ticks=style.x_ticks
            )  # pyright: ignore[reportArgumentType]
        )
        ax.xaxis.grid(True, "major", linewidth=0.5, color=style.font_color)
        if format_funcs and (fmt_x := format_funcs.get(x)):
            ax.xaxis.set_major_formatter(DynamicFuncFormatter(fmt_x))

        # Y-axis ticks and formatting
        y_min, y_max = cast(float, pd_df[y].min()), cast(float, pd_df[y].max())
        ax.yaxis.set_ticks(
            generate_ticks(
                y_min, y_max, num_ticks=style.y_ticks
            )  # pyright: ignore[reportArgumentType]
        )
        if style.yscale == "log":
            ax.yaxis.set_minor_locator(NullLocator())
        else:
            ax.minorticks_off()
        ax.yaxis.grid(True, "major", linewidth=0.5, color=style.font_color)
        if format_funcs and (fmt_y := format_funcs.get(y)):
            ax.yaxis.set_major_formatter(DynamicFuncFormatter(fmt_y))

        ax.tick_params(
            axis="both",
            which="major",
            colors=style.font_color,
            labelsize=style.font_size,
        )

    @staticmethod
    def _draw_bubbles(
        ax: Axes, plot_df: pd.DataFrame, x: str, y: str, z: str, style: StyleTemplate
    ) -> None:
        """Draw bubbles on the axes.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object.
        plot_df : pd.DataFrame
            DataFrame with data for plotting.
        x : str
            Column name for x-axis values.
        y : str
            Column name for y-axis values.
        z : str
            Column name for bubble sizes.
        style : StyleTemplate
            Styling for the plot.
        """
        sns.scatterplot(
            data=plot_df,
            x=x,
            y=y,
            size=z,
            hue="quintile",
            sizes=(100, 2000),
            legend=False,
            palette=sns.color_palette(style.palette, as_cmap=True),
            edgecolor=style.background_color,
            ax=ax,
        )

    @staticmethod
    def _draw_labels(
        ax: Axes,
        plot_df: pd.DataFrame,
        label: str,
        x: str,
        y: str,
        style: StyleTemplate,
        format_funcs: Optional[Dict[str, Optional[FormatterFunc]]],
    ) -> None:
        """Draw labels for each bubble.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object.
        plot_df : pd.DataFrame
            DataFrame with data for plotting.
        label : str
            Column name for bubble labels.
        x : str
            Column name for x-axis values.
        y : str
            Column name for y-axis values.
        style : StyleTemplate
            Styling for the plot.
        format_funcs : Optional[Dict[str, Optional[FormatterFunc]]]
            Functions to format bubble labels.
        """
        for _, row in plot_df.iterrows():
            x_val, y_val, label_val = row[x], row[y], str(row[label])
            if format_funcs and (fmt_label := format_funcs.get(label)):
                label_val = fmt_label(label_val, None)
            quintile = cast(int, row["quintile"])
            ax.text(
                cast(float, x_val),
                cast(float, y_val),
                label_val,
                ha="center",
                fontsize=style.font_mapping.get(quintile, style.font_size),
                color=style.font_color,
            )

    @staticmethod
    def _draw_lines(
        ax: Axes,
        plot_df: pd.DataFrame,
        x: str,
        y: str,
        hline: bool,
        vline: bool,
        style: StyleTemplate,
    ) -> None:
        """Draw horizontal and vertical mean lines.

        Parameters
        ----------
        ax : Axes
            Matplotlib axes object.
        plot_df : pd.DataFrame
            DataFrame with data for plotting.
        x : str
            Column name for x-axis values.
        y : str
            Column name for y-axis values.
        hline : bool
            Whether to draw a horizontal line at the mean of y.
        vline : bool
            Whether to draw a vertical line at the mean of x.
        style : StyleTemplate
            Styling for the plot.
        """
        if vline:
            ax.axvline(
                int(cast(float, plot_df[x].mean())),
                linestyle="--",
                color=style.font_color,
            )
        if hline:
            ax.axhline(
                int(cast(float, plot_df[y].mean())),
                linestyle="--",
                color=style.font_color,
            )

    def aplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = BUBBLE_STYLE_TEMPLATE,
        hline: bool = False,
        vline: bool = False,
        ax: Optional[Axes] = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot a bubble chart on existing axes.

        Parameters
        ----------
        pd_df : pd.DataFrame
            DataFrame containing the data to plot.
        label : str
            Column name used for labeling bubbles.
        x : str
            Column name for x-axis values.
        y : str
            Column name for y-axis values.
        z : str
            Column name for bubble sizes.
        title : str, optional
            Plot title. The default is None.
        style : StyleTemplate, optional
            Plot styling. The default is `BUBBLE_STYLE_TEMPLATE`.
        max_values : int, optional
            Max number of rows to display. The default is `MAX_RESULTS`.
        center_to_mean : bool, optional
            Whether to center x values around their mean. The default is False.
        sort_by : str, optional
            Column to sort by before slicing. The default is None.
        ascending : bool, optional
            Sort order. The default is False.
        hline : bool, optional
            Whether to draw a horizontal line at the mean of y. The default is False.
        vline : bool, optional
            Whether to draw a vertical line at the mean of x. The default is False.
        ax : Axes, optional
            Existing matplotlib axes to use. If None, uses current axes.
        **kwargs : Any
            Additional keyword arguments (unused, for interface compatibility).

        Returns
        -------
        Axes
            The Matplotlib axes object with the bubble chart.

        Raises
        ------
        AttributeError
            If required columns are not in the DataFrame.

        Examples
        --------
        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt
        >>> from MatplotLibAPI import Bubble
        >>> data = {
        ...     'country': ['A', 'B', 'C', 'D'],
        ...     'gdp_per_capita': [45000, 42000, 52000, 48000],
        ...     'life_expectancy': [81, 78, 83, 82],
        ...     'population': [10, 20, 5, 30]
        ... }
        >>> df = pd.DataFrame(data)
        >>> fig, ax = plt.subplots()
        >>> Bubble.aplot(df, label='country', x='gdp_per_capita',
        ...              y='life_expectancy', z='population', ax=ax)
        """
        if ax is None:
            ax = cast(Axes, plt.gca())

        format_funcs = format_func(
            style.format_funcs, label=self.label, x=self.x, y=self.y, z=self.z
        )

        Bubble._setup_axes(ax, style, self._obj, self.x, self.y, format_funcs)

        Bubble._draw_bubbles(ax, self._obj, self.x, self.y, self.z, style)
        Bubble._draw_lines(ax, self._obj, self.x, self.y, hline, vline, style)
        Bubble._draw_labels(
            ax, self._obj, self.label, self.x, self.y, style, format_funcs
        )

        if title:
            ax.set_title(
                title,
                color=style.font_color,
                fontsize=style.font_size * TITLE_SCALE_FACTOR,
            )

        return ax

    def fplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = BUBBLE_STYLE_TEMPLATE,
        hline: bool = False,
        vline: bool = False,
        figsize: Tuple[float, float] = FIG_SIZE,
    ) -> Figure:
        """Plot a bubble chart on a new figure.

        Parameters
        ----------
        pd_df : pd.DataFrame
            DataFrame containing the data to plot.
        label : str
            Column name for bubble labels.
        x : str
            Column name for x-axis values.
        y : str
            Column name for y-axis values.
        z : str
            Column name for bubble sizes.
        title : str, optional
            Plot title. The default is None.
        style : StyleTemplate, optional
            Plot styling. The default is `BUBBLE_STYLE_TEMPLATE`.
        max_values : int, optional
            Max number of rows to display. The default is `MAX_RESULTS`.
        center_to_mean : bool, optional
            Whether to center x around its mean. The default is False.
        sort_by : str, optional
            Column to sort by. The default is None.
        ascending : bool, optional
            Sort order. The default is False.
        hline : bool, optional
            Draw horizontal line at mean y. The default is False.
        vline : bool, optional
            Draw vertical line at mean x. The default is False.
        figsize : tuple[float, float], optional
            Size of the figure. The default is FIG_SIZE.


        Returns
        -------
        Figure
            A matplotlib Figure object with the bubble chart.

        Raises
        ------
        AttributeError
            If required columns are not in the DataFrame.

        Examples
        --------
        >>> import pandas as pd
        >>> from MatplotLibAPI import Bubble
        >>> data = {
        ...     'country': ['A', 'B', 'C', 'D'],
        ...     'gdp_per_capita': [45000, 42000, 52000, 48000],
        ...     'life_expectancy': [81, 78, 83, 82],
        ...     'population': [10, 20, 5, 30]
        ... }
        >>> df = pd.DataFrame(data)
        >>> fig = Bubble.fplot(df, label='country', x='gdp_per_capita',
        ...                    y='life_expectancy', z='population')
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(style.background_color)

        self.aplot(
            title=title,
            style=style,
            hline=hline,
            vline=vline,
            ax=ax,
        )
        return fig


def aplot_bubble(
    pd_df: pd.DataFrame,
    label: str,
    x: str,
    y: str,
    z: str,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    max_values: int = MAX_RESULTS,
    center_to_mean: bool = False,
    title: Optional[str] = None,
    style: StyleTemplate = BUBBLE_STYLE_TEMPLATE,
    hline: bool = False,
    vline: bool = False,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Plot a bubble chart on existing axes."""
    return Bubble(
        pd_df=pd_df,
        label=label,
        x=x,
        y=y,
        z=z,
        sort_by=sort_by,
        ascending=ascending,
        max_values=max_values,
        center_to_mean=center_to_mean,
    ).aplot(title=title, style=style, hline=hline, vline=vline, ax=ax, **kwargs)


def fplot_bubble(
    pd_df: pd.DataFrame,
    label: str,
    x: str,
    y: str,
    z: str,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    max_values: int = MAX_RESULTS,
    center_to_mean: bool = False,
    title: Optional[str] = None,
    style: StyleTemplate = BUBBLE_STYLE_TEMPLATE,
    hline: bool = False,
    vline: bool = False,
    figsize: Tuple[float, float] = FIG_SIZE,
    **kwargs: Any,
) -> Figure:
    """Plot a bubble chart on a new figure."""
    return Bubble(
        pd_df=pd_df,
        label=label,
        x=x,
        y=y,
        z=z,
        sort_by=sort_by,
        ascending=ascending,
        max_values=max_values,
        center_to_mean=center_to_mean,
    ).fplot(
        title=title, style=style, hline=hline, vline=vline, figsize=figsize, **kwargs
    )
