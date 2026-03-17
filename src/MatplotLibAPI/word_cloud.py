"""Word cloud plotting utilities."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from matplotlib.transforms import BboxBase
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.figure import Figure, SubFigure
from wordcloud import WordCloud

from .base_plot import BasePlot

from .utils import _get_axis

from .style_template import (
    FIG_SIZE,
    MAX_RESULTS,
    TITLE_SCALE_FACTOR,
    WORDCLOUD_STYLE_TEMPLATE,
    StyleTemplate,
    string_formatter,
    validate_dataframe,
)


def _filter_stopwords(
    words: Iterable[str], stopwords: Optional[Iterable[str]]
) -> np.ndarray:
    """Remove stopwords from a sequence of words.

    Parameters
    ----------
    words : Iterable[str]
        Words to filter.
    stopwords : Iterable[str], optional
        Collection of stopwords to exclude. Defaults to ``None``.

    Returns
    -------
    numpy.ndarray
        Filtered words.
    """
    if stopwords is None:
        return np.array(list(words))

    stop_set: set[str] = {word.lower() for word in stopwords}
    return np.array([word for word in words if word.lower() not in stop_set])


def from_pandas_nodelist(
    nodes_df: pd.DataFrame,
    node_col: str = "node",
    weight_col: str = "weight",
):
    """Create SankeyData from a DataFrame."""
    validate_dataframe(nodes_df, cols=[node_col, weight_col])
    return nodes_df[[node_col, weight_col]].set_index(node_col)[weight_col].to_dict()


def _prepare_word_frequencies(
    pd_df: pd.DataFrame,
    text_column: str,
    weight_column: str,
    max_words: int,
    stopwords: Optional[Iterable[str]],
) -> Tuple[list[str], list[float]]:
    """Aggregate and filter word frequencies.

    Parameters
    ----------
    pd_df : pandas.DataFrame
        Input DataFrame containing word data.
    text_column : str
        Column containing words or phrases.
    weight_column : str, optional
        Column containing numeric weights. Defaults to ``None``.
    max_words : int
        Maximum number of words to include.
    stopwords : Iterable[str], optional
        Words to exclude from the visualization. Defaults to ``None``.

    Returns
    -------
    tuple of list
        Lists of filtered words and their corresponding weights.

    Raises
    ------
    AttributeError
        If required columns are missing from the DataFrame.
    """
    validate_dataframe(pd_df, cols=[text_column], sort_by=weight_column)
    word_frequencies = from_pandas_nodelist(
        pd_df, node_col=text_column, weight_col=weight_column
    )
    words: np.ndarray = np.asarray(list(word_frequencies.keys()), dtype=np.str_)
    weights: np.ndarray = np.asarray(list(word_frequencies.values()), dtype=np.float64)

    filtered_words = _filter_stopwords(words, stopwords)
    mask: np.ndarray = np.asarray(np.isin(words, filtered_words), dtype=bool)
    filtered_weights: np.ndarray = np.asarray(weights[mask], dtype=np.float64)

    sorted_indices = np.argsort(filtered_weights)[::-1]
    sorted_words = filtered_words[sorted_indices][:max_words].tolist()
    sorted_weights = filtered_weights[sorted_indices][:max_words].tolist()

    return sorted_words, sorted_weights


def _create_circular_mask(size: int = 300, radius: Optional[int] = None) -> np.ndarray:
    """Construct a binary mask with a circular opening for a word cloud.

    Parameters
    ----------
    size : int, optional
        Width and height of the mask in pixels. Defaults to ``300``.
    radius : int, optional
        Radius of the circular opening in pixels. Defaults to ``size // 2``.

    Returns
    -------
    numpy.ndarray
        Two-dimensional array suitable for the ``mask`` argument of
        ``wordcloud.WordCloud`` where ``0`` values define the drawable region.

    Raises
    ------
    ValueError
        If ``size`` or ``radius`` are non-positive.
    """
    if size <= 0:
        raise ValueError("size must be a positive integer.")

    resolved_radius: int = radius if radius is not None else size // 2
    if resolved_radius <= 0:
        raise ValueError("radius must be a positive integer.")

    center: float = (size - 1) / 2
    x, y = np.ogrid[:size, :size]
    mask_region = (x - center) ** 2 + (y - center) ** 2 > resolved_radius**2
    return 255 * mask_region.astype(np.uint8)


def _plot_words(
    ax: Axes,
    words: Sequence[str],
    weights: Sequence[float],
    style: StyleTemplate,
    title: Optional[str],
    random_state: Optional[int],
    mask: Optional[np.ndarray],
) -> Axes:
    """Render words on the provided axes with sizes proportional to weights.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw.
    words : Sequence[str]
        Words to render.
    weights : Sequence[float]
        Corresponding weights for sizing.
    style : StyleTemplate
        Style configuration for the plot.
    title : str, optional
        Title of the plot. Defaults to ``None``.
    random_state : int, optional
        Seed for reproducible placement. Defaults to ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the rendered word cloud.
    """
    ax.set_facecolor(style.background_color)
    ax.axis("off")

    if not words:
        if title:
            ax.set_title(
                title,
                color=style.font_color,
                fontsize=style.font_size * TITLE_SCALE_FACTOR * 0.75,
            )
        return ax

    fig_raw = ax.figure
    if not isinstance(fig_raw, (Figure, SubFigure)):
        raise RuntimeError("Axes is not associated with a Figure.")

    fig_obj: Figure | SubFigure = fig_raw
    canvas: FigureCanvasBase = fig_obj.canvas
    if canvas is None:
        raise RuntimeError("Figure does not have an attached canvas.")

    canvas.draw()
    ax_bbox: BboxBase = ax.get_window_extent()

    if mask is None:
        mask_dimension: int = max(int(ax_bbox.width), int(ax_bbox.height), 1)
        resolved_mask: np.ndarray = _create_circular_mask(size=mask_dimension)
    else:
        resolved_mask: np.ndarray = np.asarray(mask, dtype=np.uint8)

    if resolved_mask.ndim != 2:
        raise ValueError("mask must be a 2D array.")

    height, width = resolved_mask.shape

    frequency_map: Dict[str, float] = {
        string_formatter(word): float(weight) for word, weight in zip(words, weights)
    }
    if frequency_map and max(frequency_map.values()) <= 0:
        frequency_map = {word: 1.0 for word in frequency_map}

    max_font_size = int(style.font_mapping[max(style.font_mapping.keys())] * 20)

    wc: WordCloud = WordCloud(
        width=width,
        height=height,
        background_color=style.background_color,
        colormap=colormaps.get_cmap(style.palette),
        # min_font_size=min_font_size,
        # max_font_size=max_font_size,
        random_state=random_state,
        mask=resolved_mask,
    ).generate_from_frequencies(frequency_map, max_font_size=max_font_size)

    ax.imshow(wc.to_array(), interpolation="bilinear")

    if title:
        ax.set_title(
            title,
            color=style.font_color,
            fontsize=style.font_size * TITLE_SCALE_FACTOR * 0.75,
        )
    return ax


class WordCloudPlot(BasePlot):
    """Represent a word-cloud plot builder.

    Methods
    -------
    aplot
        Plot the word cloud on an existing Matplotlib axes.
    fplot
        Plot the word cloud on a new Matplotlib figure.
    """

    def __init__(self, pd_df: pd.DataFrame, text_column: str, weight_column: str):
        validate_dataframe(pd_df, cols=[text_column], sort_by=weight_column)
        super().__init__(pd_df=pd_df)
        self.text_column = text_column
        self.weight_column = weight_column

    def aplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = WORDCLOUD_STYLE_TEMPLATE,
        max_words: int = MAX_RESULTS,
        stopwords: Optional[Iterable[str]] = None,
        random_state: Optional[int] = None,
        ax: Optional[Axes] = None,
        mask: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot the configured word cloud on existing axes.

        Parameters
        ----------
        title : str, optional
            Plot title.
        style : StyleTemplate, optional
            Style configuration. The default is ``WORDCLOUD_STYLE_TEMPLATE``.
        max_words : int, optional
            Maximum number of words to include. The default is ``MAX_RESULTS``.
        stopwords : Iterable[str], optional
            Words to exclude from the cloud.
        random_state : int, optional
            Random seed used by word-cloud placement.
        ax : Axes, optional
            Matplotlib axes to draw on. If None, use the current axes.
        mask : np.ndarray, optional
            Binary mask controlling drawable pixels.
        **kwargs : Any
            Additional keyword arguments reserved for compatibility.

        Returns
        -------
        Axes
            Matplotlib axes containing the rendered word cloud.
        """
        words, weights = _prepare_word_frequencies(
            pd_df=self._obj,
            text_column=self.text_column,
            weight_column=self.weight_column,
            max_words=max_words,
            stopwords=stopwords,
        )
        plot_ax = _get_axis(ax)
        return _plot_words(
            plot_ax,
            words,
            weights,
            style=style,
            title=title,
            random_state=random_state,
            mask=mask,
        )

    def fplot(
        self,
        title: Optional[str] = None,
        style: StyleTemplate = WORDCLOUD_STYLE_TEMPLATE,
        max_words: int = MAX_RESULTS,
        stopwords: Optional[Iterable[str]] = None,
        random_state: Optional[int] = None,
        figsize: Tuple[float, float] = FIG_SIZE,
        mask: Optional[np.ndarray] = None,
    ) -> Figure:
        """Plot the configured word cloud on a new figure.

        Parameters
        ----------
        title : str, optional
            Plot title.
        style : StyleTemplate, optional
            Style configuration. The default is ``WORDCLOUD_STYLE_TEMPLATE``.
        max_words : int, optional
            Maximum number of words to include. The default is ``MAX_RESULTS``.
        stopwords : Iterable[str], optional
            Words to exclude from the cloud.
        random_state : int, optional
            Random seed used by word-cloud placement.
        figsize : tuple[float, float], optional
            Figure size. The default is ``FIG_SIZE``.
        mask : np.ndarray, optional
            Binary mask controlling drawable pixels.

        Returns
        -------
        Figure
            Matplotlib figure containing the rendered word cloud.
        """
        fig, ax = plt.subplots(figsize=figsize)
        self.aplot(
            title=title,
            style=style,
            max_words=max_words,
            stopwords=stopwords,
            random_state=random_state,
            ax=ax,
            mask=mask,
        )
        return fig


def aplot_wordcloud(
    pd_df: pd.DataFrame,
    text_column: str,
    weight_column: str,
    title: Optional[str] = None,
    style: StyleTemplate = WORDCLOUD_STYLE_TEMPLATE,
    max_words: int = MAX_RESULTS,
    stopwords: Optional[Iterable[str]] = None,
    random_state: Optional[int] = None,
    ax: Optional[Axes] = None,
    mask: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Axes:
    """Plot a word cloud on the provided axes.

    Parameters
    ----------
    pd_df : pandas.DataFrame
        DataFrame containing the words to visualize.
    text_column : str
        Column containing words or phrases.
    weight_column : str, optional
        Column containing numeric weights. Defaults to ``None`` for equal weights.
    title : str, optional
        Plot title. Defaults to ``None``.
    style : StyleTemplate, optional
        Styling options. Defaults to ``WORDCLOUD_STYLE_TEMPLATE``.
    max_words : int, optional
        Maximum number of words to display. Defaults to ``MAX_RESULTS``.
    stopwords : Iterable[str], optional
        Words to exclude from the visualization. Defaults to ``None``.
    random_state : int, optional
        Seed for word placement. Defaults to ``None``.
    ax : matplotlib.axes.Axes or numpy.ndarray, optional
        Axes to draw on. Defaults to ``None`` which uses the current axes.
    mask : numpy.ndarray, optional
        Two-dimensional mask array defining the drawable region of the word cloud.
        Defaults to a circular mask generated by :func:`create_circular_mask`.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the rendered word cloud.

    Raises
    ------
    AttributeError
        If required columns are missing from the DataFrame.
    TypeError
        If ``ax`` is not a ``matplotlib.axes.Axes`` instance.
    """
    return WordCloudPlot(
        pd_df=pd_df,
        text_column=text_column,
        weight_column=weight_column,
    ).aplot(
        title=title,
        style=style,
        max_words=max_words,
        stopwords=stopwords,
        random_state=random_state,
        ax=ax,
        mask=mask,
        **kwargs,
    )


def fplot_wordcloud(
    pd_df: pd.DataFrame,
    text_column: str,
    weight_column: str,
    title: Optional[str] = None,
    style: StyleTemplate = WORDCLOUD_STYLE_TEMPLATE,
    max_words: int = MAX_RESULTS,
    stopwords: Optional[Iterable[str]] = None,
    random_state: Optional[int] = None,
    figsize: Tuple[float, float] = FIG_SIZE,
    mask: Optional[np.ndarray] = None,
) -> Figure:
    """Return a new figure containing a word cloud.

    Parameters
    ----------
    pd_df : pd.DataFrame
        DataFrame containing text and optional weight columns.
    text_column : str
        Column containing text terms.
    weight_column : str, optional
        Column containing numeric weights. The default is None.
    title : str, optional
        Plot title. The default is None.
    style : StyleTemplate, optional
        Style configuration. The default is ``WORDCLOUD_STYLE_TEMPLATE``.
    max_words : int, optional
        Maximum number of words to include. The default is ``MAX_RESULTS``.
    stopwords : Iterable[str], optional
        Words to exclude from the cloud. The default is None.
    random_state : int, optional
        Random seed used by word cloud layout. The default is None.
    figsize : tuple[float, float], optional
        Size of the created figure. The default is ``FIG_SIZE``.
    mask : np.ndarray, optional
        Optional 2D mask limiting word cloud shape. The default is None.

    Returns
    -------
    Figure
        Matplotlib figure containing the word cloud.
    """
    return WordCloudPlot(
        pd_df=pd_df, text_column=text_column, weight_column=weight_column
    ).fplot(
        title=title,
        style=style,
        max_words=max_words,
        stopwords=stopwords,
        random_state=random_state,
        figsize=figsize,
        mask=mask,
    )
