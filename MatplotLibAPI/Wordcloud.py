"""Word cloud plotting utilities."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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


def _normalize_weights(weights: Sequence[float], base_size: int) -> np.ndarray:
    """Normalize weights to a range of font sizes.

    Parameters
    ----------
    weights : Sequence[float]
        Sequence of weights representing word importance.
    base_size : int
        Base font size used as the lower bound for scaling.

    Returns
    -------
    numpy.ndarray
        Array of font sizes corresponding to the provided weights.
    """
    numeric_weights = np.asarray(weights, dtype=float)
    if numeric_weights.size == 0:
        return np.array([], dtype=float)
    min_weight = numeric_weights.min()
    max_weight = numeric_weights.max()
    if min_weight == max_weight:
        return np.full_like(numeric_weights, fill_value=base_size, dtype=float)

    min_size, max_size = base_size, base_size * 4
    return np.interp(numeric_weights, (min_weight, max_weight), (min_size, max_size))


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

    stop_set = {word.lower() for word in stopwords}
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
        weight_col = cast(str, weight_column)
        freq_series = cast(pd.Series, pd_df.groupby(text_column)[weight_col].sum())
        freq_series = freq_series.sort_values(ascending=False)

    words = freq_series.index.to_numpy()
    weights = freq_series.to_numpy(dtype=float)

    words = _filter_stopwords(words, stopwords)
    mask = np.isin(freq_series.index, words)
    weights = weights[mask]

    sorted_indices = np.argsort(weights)[::-1]
    words = words[sorted_indices][:max_words].tolist()
    weights = weights[sorted_indices][:max_words].tolist()

    return words, weights


def _plot_words(
    ax: Axes,
    words: Sequence[str],
    weights: Sequence[float],
    style: StyleTemplate,
    title: Optional[str],
    random_state: Optional[int],
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

    ax.figure.canvas.draw()
    ax_bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans)
    width = max(int(ax_bbox.width), 1)
    height = max(int(ax_bbox.height), 1)

    frequency_map = {
        string_formatter(word): weight for word, weight in zip(words, weights)
    }

    font_sizes = _normalize_weights(weights, base_size=style.font_size)
    wc = WordCloud(
        width=width,
        height=height,
        background_color=style.background_color,
        colormap=colormaps.get_cmap(style.palette),
        min_font_size=int(font_sizes.min(initial=style.font_size)),
        max_font_size=int(font_sizes.max(initial=style.font_size * 4)),
        random_state=random_state,
    ).generate_from_frequencies(frequency_map)

    ax.imshow(wc, interpolation="bilinear")

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
    ax: Optional[Axes] = None,
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
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Defaults to ``None`` which uses the current axes.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the rendered word cloud.

    Raises
    ------
    AttributeError
        If required columns are missing from the DataFrame.
    """
    if ax is None:
        ax = cast(Axes, plt.gca())

    words, weights = _prepare_word_frequencies(
        pd_df=pd_df,
        text_column=text_column,
        weight_column=weight_column,
        max_words=max_words,
        stopwords=stopwords,
    )
    return _plot_words(
        ax, words, weights, style=style, title=title, random_state=random_state
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
    fig = cast(Figure, fig_raw)
    ax = cast(Axes, ax_raw)

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
    )
    fig.patch.set_facecolor(style.background_color)
    fig.tight_layout()
    return fig
