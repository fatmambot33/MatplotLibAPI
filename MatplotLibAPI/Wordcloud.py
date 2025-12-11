"""Word cloud plotting utilities."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, cast

from matplotlib.transforms import Bbox
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.backend_bases import FigureCanvasBase
from wordcloud import WordCloud

from .StyleTemplate import (
    FIG_SIZE,
    MAX_RESULTS,
    StyleTemplate,
    string_formatter,
    validate_dataframe,
)

WORDCLOUD_STYLE_TEMPLATE = StyleTemplate(
    background_color="black", font_color="white", palette="plasma"
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


def _prepare_word_frequencies(
    pd_df: pd.DataFrame,
    text_column: str,
    weight_column: Optional[str],
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

    if weight_column is None:
        freq_series: pd.Series = pd_df[text_column].value_counts()
    else:
        weight_col: str = cast(str, weight_column)
        freq_series = cast(pd.Series, pd_df.groupby(text_column)[weight_col].sum())
        freq_series = cast(pd.Series, freq_series[freq_series > 0])
        freq_series = freq_series.sort_values(ascending=False)
        if freq_series.empty:
            freq_series = pd_df[text_column].value_counts()

    words_: np.ndarray[Tuple[int], np.dtype[Any]] = freq_series.index.to_numpy()
    weights_: np.ndarray[Tuple[int], np.dtype[Any]] = freq_series.to_numpy(dtype=float)

    words: np.ndarray[Tuple[Any], np.dtype[Any]] = _filter_stopwords(words_, stopwords)
    mask: np.ndarray[Tuple[Any], np.dtype[np.bool[bool]]] = np.isin(
        freq_series.index, words
    )
    weights: np.ndarray[Tuple[Any], np.dtype[Any]] = weights_[mask]

    sorted_indices: NDArray[np.intp] = np.argsort(weights)[::-1]
    sorted_words: np.ndarray = words[sorted_indices][:max_words]
    sorted_weights: np.ndarray = weights[sorted_indices][:max_words]

    words_list: list[str] = sorted_words.tolist()
    weights_list: list[float] = sorted_weights.tolist()

    return words_list, weights_list


def create_circular_mask(size: int = 300, radius: Optional[int] = None) -> np.ndarray:
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
            ax.set_title(title, color=style.font_color, fontsize=style.font_size * 1.5)
        return ax

    fig_obj: Figure | SubFigure | None = ax.get_figure()
    if not isinstance(fig_obj, Figure):
        raise RuntimeError("Axes is not associated with a Figure.")

    canvas: FigureCanvasBase = fig_obj.canvas
    if canvas is None:
        raise RuntimeError("Figure does not have an attached canvas.")

    canvas.draw()
    ax_bbox: Bbox = ax.get_window_extent()

    if mask is None:
        mask_dimension: int = max(int(ax_bbox.width), int(ax_bbox.height), 1)
        resolved_mask: NDArray[np.uint8] = create_circular_mask(size=mask_dimension)
    else:
        resolved_mask: NDArray[np.uint8] = np.asarray(mask, dtype=np.uint8)

    if resolved_mask.ndim != 2:
        raise ValueError("mask must be a 2D array.")

    height, width = resolved_mask.shape

    frequency_map: Dict[str, float] = {
        string_formatter(word): weight for word, weight in zip(words, weights)
    }

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
        ax.set_title(title, color=style.font_color, fontsize=style.font_size * 1.5)
    return ax


def aplot_wordcloud(
    pd_df: pd.DataFrame,
    text_column: str,
    weight_column: Optional[str] = None,
    title: Optional[str] = None,
    style: StyleTemplate = WORDCLOUD_STYLE_TEMPLATE,
    max_words: int = MAX_RESULTS,
    stopwords: Optional[Iterable[str]] = None,
    random_state: Optional[int] = None,
    ax: Optional[Axes | np.ndarray[Any, np.dtype[Any]]] = None,
    mask: Optional[np.ndarray] = None,
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
    if ax is None:
        ax = cast(Axes, plt.gca())
    elif isinstance(ax, np.ndarray):
        raise TypeError("ax must be a single matplotlib Axes instance, not an array.")
    elif not isinstance(ax, Axes):
        raise TypeError("ax must be a matplotlib Axes instance.")

    words, weights = _prepare_word_frequencies(
        pd_df=pd_df,
        text_column=text_column,
        weight_column=weight_column,
        max_words=max_words,
        stopwords=stopwords,
    )
    return _plot_words(
        ax,
        words,
        weights,
        style=style,
        title=title,
        random_state=random_state,
        mask=mask,
    )


def fplot_wordcloud(
    pd_df: pd.DataFrame,
    text_column: str,
    weight_column: Optional[str] = None,
    title: Optional[str] = None,
    style: StyleTemplate = WORDCLOUD_STYLE_TEMPLATE,
    max_words: int = MAX_RESULTS,
    stopwords: Optional[Iterable[str]] = None,
    random_state: Optional[int] = None,
    figsize: Tuple[float, float] = FIG_SIZE,
    save_path: Optional[str] = None,
    savefig_kwargs: Optional[Dict[str, Any]] = None,
    mask: Optional[np.ndarray] = None,
) -> Figure:
    """Create a new figure with a word cloud.

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
    figsize : tuple of float, optional
        Figure size. Defaults to ``FIG_SIZE``.
    mask : numpy.ndarray, optional
        Two-dimensional mask array defining the drawable region of the word cloud.
        Defaults to a circular mask generated by :func:`create_circular_mask`.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the rendered word cloud.

    Raises
    ------
    AttributeError
        If required columns are missing from the DataFrame.
    """
    fig_raw, ax_raw = plt.subplots(figsize=figsize)
    fig: Figure = cast(Figure, fig_raw)
    ax: Axes = cast(Axes, ax_raw)

    _plot_words(
        ax,
        *_prepare_word_frequencies(
            pd_df=pd_df,
            text_column=text_column,
            weight_column=weight_column,
            max_words=max_words,
            stopwords=stopwords,
        ),
        style=style,
        title=title,
        random_state=random_state,
        mask=mask,
    )
    fig.patch.set_facecolor(style.background_color)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, **(savefig_kwargs or {}))
    return fig
