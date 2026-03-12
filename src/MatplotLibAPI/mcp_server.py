"""MCP server helpers for exposing MatplotLibAPI plotting tools.

This module provides a minimal Model Context Protocol (MCP) server focused on
bubble charts so language-model agents can request chart generation from either
CSV input or in-memory tabular records.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .area import fplot_area
from .bar import fplot_bar
from .box_violin import fplot_box_violin
from .bubble import aplot_bubble
from .heatmap import fplot_correlation_matrix, fplot_heatmap
from .histogram import fplot_histogram_kde
from .network import fplot_network
from .pie import fplot_pie_donut
from .sankey import fplot_sankey
from .sunburst import fplot_sunburst
from .table import fplot_table
from .timeserie import fplot_timeserie
from .treemap import fplot_treemap
from .waffle import fplot_waffle
from .word_cloud import fplot_wordcloud

TableRecords = list[dict[str, Any]]


def _load_bubble_dataframe(
    csv_path: Optional[str] = None, table: Optional[TableRecords] = None
) -> pd.DataFrame:
    """Load bubble chart input data from CSV path or in-memory table records.

    Parameters
    ----------
    csv_path : str, optional
        Path to a CSV file containing plotting data. The default is None.
    table : list[dict[str, Any]], optional
        In-memory row records with column names as keys. The default is None.

    Returns
    -------
    pd.DataFrame
        Loaded tabular data for plotting.

    Raises
    ------
    ValueError
        If both ``csv_path`` and ``table`` are missing.
    """
    if csv_path is None and table is None:
        raise ValueError("Provide either `csv_path` or `table`.")
    if table is not None:
        return pd.DataFrame(table)

    data_path = Path(str(csv_path)).expanduser().resolve()
    return pd.read_csv(data_path)


def _build_bubble_chart_figure(
    label: str,
    x: str,
    y: str,
    z: str,
    csv_path: Optional[str] = None,
    table: Optional[TableRecords] = None,
    title: Optional[str] = None,
    max_values: int = 50,
    center_to_mean: bool = False,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    hline: bool = False,
    vline: bool = False,
) -> Figure:
    """Create a bubble chart figure from tabular input.

    Parameters
    ----------
    label : str
        Column name used for bubble labels.
    x : str
        Column name for the x-axis values.
    y : str
        Column name for the y-axis values.
    z : str
        Column name used for bubble size.
    csv_path : str, optional
        Path to a CSV file containing plotting data. The default is None.
    table : list[dict[str, Any]], optional
        In-memory row records with column names as keys. The default is None.
    title : str, optional
        Chart title. The default is None.
    max_values : int, optional
        Maximum number of rows to include in the plot. The default is 50.
    center_to_mean : bool, optional
        Whether to center x-axis values around their mean. The default is False.
    sort_by : str, optional
        Column used to sort before selecting rows. The default is None.
    ascending : bool, optional
        Sort order used with ``sort_by``. The default is False.
    hline : bool, optional
        Whether to draw a horizontal mean line for y values. The default is False.
    vline : bool, optional
        Whether to draw a vertical mean line for x values. The default is False.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the rendered bubble chart.
    """
    pd_df = _load_bubble_dataframe(csv_path=csv_path, table=table)
    fig, ax = plt.subplots()
    aplot_bubble(
        pd_df=pd_df,
        label=label,
        x=x,
        y=y,
        z=z,
        title=title,
        max_values=max_values,
        center_to_mean=center_to_mean,
        sort_by=sort_by,
        ascending=ascending,
        hline=hline,
        vline=vline,
        ax=ax,
    )
    return fig


def _render_bubble_chart_octet_from_source(
    label: str,
    x: str,
    y: str,
    z: str,
    csv_path: Optional[str] = None,
    table: Optional[TableRecords] = None,
    title: Optional[str] = None,
    max_values: int = 50,
    center_to_mean: bool = False,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    hline: bool = False,
    vline: bool = False,
) -> bytes:
    """Render a bubble chart from tabular input and return PNG octets.

    Parameters
    ----------
    label : str
        Column name used for bubble labels.
    x : str
        Column name for the x-axis values.
    y : str
        Column name for the y-axis values.
    z : str
        Column name used for bubble size.
    csv_path : str, optional
        Path to a CSV file containing plotting data. The default is None.
    table : list[dict[str, Any]], optional
        In-memory row records with column names as keys. The default is None.
    title : str, optional
        Chart title. The default is None.
    max_values : int, optional
        Maximum number of rows to include in the plot. The default is 50.
    center_to_mean : bool, optional
        Whether to center x-axis values around their mean. The default is False.
    sort_by : str, optional
        Column used to sort before selecting rows. The default is None.
    ascending : bool, optional
        Sort order used with ``sort_by``. The default is False.
    hline : bool, optional
        Whether to draw a horizontal mean line for y values. The default is False.
    vline : bool, optional
        Whether to draw a vertical mean line for x values. The default is False.

    Returns
    -------
    bytes
        PNG payload bytes suitable for ``application/octet-stream`` responses.
    """
    fig = _build_bubble_chart_figure(
        csv_path=csv_path,
        table=table,
        label=label,
        x=x,
        y=y,
        z=z,
        title=title,
        max_values=max_values,
        center_to_mean=center_to_mean,
        sort_by=sort_by,
        ascending=ascending,
        hline=hline,
        vline=vline,
    )
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def render_bubble_chart(
    output_path: str,
    label: str,
    x: str,
    y: str,
    z: str,
    csv_path: Optional[str] = None,
    table: Optional[TableRecords] = None,
    title: Optional[str] = None,
    max_values: int = 50,
    center_to_mean: bool = False,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    hline: bool = False,
    vline: bool = False,
) -> str:
    """Render a bubble chart from table data and write it to disk.

    Parameters
    ----------
    output_path : str
        Path where the generated chart image is saved.
    label : str
        Column name used for bubble labels.
    x : str
        Column name for the x-axis values.
    y : str
        Column name for the y-axis values.
    z : str
        Column name used for bubble size.
    csv_path : str, optional
        Path to a CSV file containing plotting data. The default is None.
    table : list[dict[str, Any]], optional
        In-memory row records with column names as keys. The default is None.
    title : str, optional
        Chart title. The default is None.
    max_values : int, optional
        Maximum number of rows to include in the plot. The default is 50.
    center_to_mean : bool, optional
        Whether to center x-axis values around their mean. The default is False.
    sort_by : str, optional
        Column used to sort before selecting rows. The default is None.
    ascending : bool, optional
        Sort order used with ``sort_by``. The default is False.
    hline : bool, optional
        Whether to draw a horizontal mean line for y values. The default is False.
    vline : bool, optional
        Whether to draw a vertical mean line for x values. The default is False.

    Returns
    -------
    str
        The resolved output path for the generated image.
    """
    out_path = Path(output_path).expanduser().resolve()
    if out_path.suffix.lower() != ".png":
        out_path = out_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _render_bubble_chart_octet_from_source(
        csv_path=csv_path,
        table=table,
        label=label,
        x=x,
        y=y,
        z=z,
        title=title,
        max_values=max_values,
        center_to_mean=center_to_mean,
        sort_by=sort_by,
        ascending=ascending,
        hline=hline,
        vline=vline,
    )
    out_path.write_bytes(payload)
    return str(out_path)


def render_bubble_chart_octet(
    label: str,
    x: str,
    y: str,
    z: str,
    csv_path: Optional[str] = None,
    table: Optional[TableRecords] = None,
    title: Optional[str] = None,
    max_values: int = 50,
    center_to_mean: bool = False,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    hline: bool = False,
    vline: bool = False,
) -> bytes:
    """Render a bubble chart and return PNG bytes as an octet payload.

    Parameters
    ----------
    label : str
        Column name used for bubble labels.
    x : str
        Column name for the x-axis values.
    y : str
        Column name for the y-axis values.
    z : str
        Column name used for bubble size.
    csv_path : str, optional
        Path to a CSV file containing plotting data. The default is None.
    table : list[dict[str, Any]], optional
        In-memory row records with column names as keys. The default is None.
    title : str, optional
        Chart title. The default is None.
    max_values : int, optional
        Maximum number of rows to include in the plot. The default is 50.
    center_to_mean : bool, optional
        Whether to center x-axis values around their mean. The default is False.
    sort_by : str, optional
        Column used to sort before selecting rows. The default is None.
    ascending : bool, optional
        Sort order used with ``sort_by``. The default is False.
    hline : bool, optional
        Whether to draw a horizontal mean line for y values. The default is False.
    vline : bool, optional
        Whether to draw a vertical mean line for x values. The default is False.

    Returns
    -------
    bytes
        PNG payload bytes suitable for ``application/octet-stream`` responses.
    """
    return _render_bubble_chart_octet_from_source(
        csv_path=csv_path,
        table=table,
        label=label,
        x=x,
        y=y,
        z=z,
        title=title,
        max_values=max_values,
        center_to_mean=center_to_mean,
        sort_by=sort_by,
        ascending=ascending,
        hline=hline,
        vline=vline,
    )


def _build_network_chart_figure(
    csv_path: Optional[str] = None,
    table: Optional[TableRecords] = None,
    edge_source_col: str = "source",
    edge_target_col: str = "target",
    edge_weight_col: str = "weight",
    title: Optional[str] = None,
) -> Figure:
    """Create a network chart figure from tabular input.

    Parameters
    ----------
    csv_path : str, optional
        Path to a CSV file containing edge data. The default is None.
    table : list[dict[str, Any]], optional
        In-memory row records with edge columns as keys. The default is None.
    edge_source_col : str, optional
        Column name for source nodes. The default is ``"source"``.
    edge_target_col : str, optional
        Column name for target nodes. The default is ``"target"``.
    edge_weight_col : str, optional
        Column name for edge weights. The default is ``"weight"``.
    title : str, optional
        Chart title. The default is None.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the rendered network chart.
    """
    pd_df = _load_bubble_dataframe(csv_path=csv_path, table=table)
    return fplot_network(
        pd_df=pd_df,
        edge_source_col=edge_source_col,
        edge_target_col=edge_target_col,
        edge_weight_col=edge_weight_col,
        title=title,
    )


def _render_network_chart_octet_from_source(
    csv_path: Optional[str] = None,
    table: Optional[TableRecords] = None,
    edge_source_col: str = "source",
    edge_target_col: str = "target",
    edge_weight_col: str = "weight",
    title: Optional[str] = None,
) -> bytes:
    """Render a network chart from tabular input and return PNG octets.

    Parameters
    ----------
    csv_path : str, optional
        Path to a CSV file containing edge data. The default is None.
    table : list[dict[str, Any]], optional
        In-memory row records with edge columns as keys. The default is None.
    edge_source_col : str, optional
        Column name for source nodes. The default is ``"source"``.
    edge_target_col : str, optional
        Column name for target nodes. The default is ``"target"``.
    edge_weight_col : str, optional
        Column name for edge weights. The default is ``"weight"``.
    title : str, optional
        Chart title. The default is None.

    Returns
    -------
    bytes
        PNG payload bytes suitable for ``application/octet-stream`` responses.
    """
    fig = _build_network_chart_figure(
        csv_path=csv_path,
        table=table,
        edge_source_col=edge_source_col,
        edge_target_col=edge_target_col,
        edge_weight_col=edge_weight_col,
        title=title,
    )
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def render_network_chart_octet(
    csv_path: Optional[str] = None,
    table: Optional[TableRecords] = None,
    edge_source_col: str = "source",
    edge_target_col: str = "target",
    edge_weight_col: str = "weight",
    title: Optional[str] = None,
) -> bytes:
    """Render a network chart and return PNG bytes as an octet payload.

    Parameters
    ----------
    csv_path : str, optional
        Path to a CSV file containing edge data. The default is None.
    table : list[dict[str, Any]], optional
        In-memory row records with edge columns as keys. The default is None.
    edge_source_col : str, optional
        Column name for source nodes. The default is ``"source"``.
    edge_target_col : str, optional
        Column name for target nodes. The default is ``"target"``.
    edge_weight_col : str, optional
        Column name for edge weights. The default is ``"weight"``.
    title : str, optional
        Chart title. The default is None.

    Returns
    -------
    bytes
        PNG payload bytes suitable for ``application/octet-stream`` responses.
    """
    return _render_network_chart_octet_from_source(
        csv_path=csv_path,
        table=table,
        edge_source_col=edge_source_col,
        edge_target_col=edge_target_col,
        edge_weight_col=edge_weight_col,
        title=title,
    )


def _render_plotly_octet(fig: Any) -> bytes:
    """Render a Plotly figure into PNG octets.

    Parameters
    ----------
    fig : Any
        Plotly figure object supporting ``to_image``.

    Returns
    -------
    bytes
        PNG payload bytes suitable for ``application/octet-stream`` responses.
    """
    return bytes(fig.to_image(format="png"))


def render_plot_module_octet(
    plot_module: str,
    params: dict[str, Any],
    csv_path: Optional[str] = None,
    table: Optional[TableRecords] = None,
) -> bytes:
    """Render a supported plot module and return PNG bytes.

    Parameters
    ----------
    plot_module : str
        Plot module key. Supported values are ``bubble``, ``network``, ``bar``,
        ``histogram``, ``box_violin``, ``heatmap``, ``correlation_matrix``,
        ``area``, ``pie``, ``waffle``, ``sankey``, ``table``, ``timeserie``,
        ``wordcloud``, ``treemap``, and ``sunburst``.
    params : dict[str, Any]
        Module-specific plotting parameters excluding the DataFrame argument.
    csv_path : str, optional
        Path to a CSV file containing source data. The default is None.
    table : list[dict[str, Any]], optional
        In-memory row records with source columns as keys. The default is None.

    Returns
    -------
    bytes
        PNG payload bytes suitable for ``application/octet-stream`` responses.

    Raises
    ------
    ValueError
        If ``plot_module`` is not supported.
    """
    pd_df = _load_bubble_dataframe(csv_path=csv_path, table=table)

    if plot_module == "bubble":
        return _render_bubble_chart_octet_from_source(
            csv_path=csv_path,
            table=table,
            **params,
        )
    if plot_module == "network":
        return _render_network_chart_octet_from_source(
            csv_path=csv_path,
            table=table,
            **params,
        )

    matplotlib_renderers: dict[str, Any] = {
        "bar": fplot_bar,
        "histogram": fplot_histogram_kde,
        "box_violin": fplot_box_violin,
        "heatmap": fplot_heatmap,
        "correlation_matrix": fplot_correlation_matrix,
        "area": fplot_area,
        "pie": fplot_pie_donut,
        "waffle": fplot_waffle,
        "table": fplot_table,
        "timeserie": fplot_timeserie,
        "wordcloud": fplot_wordcloud,
    }
    plotly_renderers: dict[str, Any] = {
        "sankey": fplot_sankey,
        "treemap": fplot_treemap,
        "sunburst": fplot_sunburst,
    }

    if plot_module in matplotlib_renderers:
        fig = matplotlib_renderers[plot_module](pd_df=pd_df, **params)
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return buffer.getvalue()

    if plot_module in plotly_renderers:
        fig = plotly_renderers[plot_module](pd_df=pd_df, **params)
        return _render_plotly_octet(fig)

    supported = sorted(
        list(matplotlib_renderers.keys())
        + list(plotly_renderers.keys())
        + ["bubble", "network"]
    )
    raise ValueError(f"Unsupported plot_module '{plot_module}'. Supported: {supported}")


def create_bubble_mcp_server() -> Any:
    """Create an MCP server exposing bubble plotting as a tool.

    Returns
    -------
    FastMCP
        A configured FastMCP server instance.

    Raises
    ------
    ImportError
        If the optional ``mcp`` package is not installed.
    """
    try:
        from mcp.server.fastmcp import FastMCP  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - exercised when optional dep missing
        raise ImportError(
            "Install MatplotLibAPI with MCP support: `pip install MatplotLibAPI[mcp]`."
        ) from exc

    mcp = FastMCP("MatplotLibAPI Bubble")

    @mcp.tool()
    def plot_bubble(
        label: str,
        x: str,
        y: str,
        z: str,
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
        title: Optional[str] = None,
        max_values: int = 50,
        center_to_mean: bool = False,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        hline: bool = False,
        vline: bool = False,
    ) -> bytes:
        """Generate a bubble chart image from table input and return octets.

        Parameters
        ----------
        label : str
            Column name used for bubble labels.
        x : str
            Column name for x-axis values.
        y : str
            Column name for y-axis values.
        z : str
            Column name used for bubble size.
        csv_path : str, optional
            Path to a CSV file containing plotting data. The default is None.
        table : list[dict[str, Any]], optional
            In-memory row records with column names as keys. The default is None.
        title : str, optional
            Chart title. The default is None.
        max_values : int, optional
            Maximum number of points to include. The default is 50.
        center_to_mean : bool, optional
            Whether to center x-axis values around their mean. The default is False.
        sort_by : str, optional
            Column used to sort before plotting. The default is None.
        ascending : bool, optional
            Sort order used with ``sort_by``. The default is False.
        hline : bool, optional
            Whether to draw a horizontal mean line. The default is False.
        vline : bool, optional
            Whether to draw a vertical mean line. The default is False.

        Returns
        -------
        bytes
            PNG octet payload of the rendered bubble chart.
        """
        return _render_bubble_chart_octet_from_source(
            csv_path=csv_path,
            table=table,
            label=label,
            x=x,
            y=y,
            z=z,
            title=title,
            max_values=max_values,
            center_to_mean=center_to_mean,
            sort_by=sort_by,
            ascending=ascending,
            hline=hline,
            vline=vline,
        )

    @mcp.tool()
    def plot_network(
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
        edge_source_col: str = "source",
        edge_target_col: str = "target",
        edge_weight_col: str = "weight",
        title: Optional[str] = None,
    ) -> bytes:
        """Generate a network chart image from table input and return octets.

        Parameters
        ----------
        csv_path : str, optional
            Path to a CSV file containing edge data. The default is None.
        table : list[dict[str, Any]], optional
            In-memory row records with edge columns as keys. The default is None.
        edge_source_col : str, optional
            Column name for source nodes. The default is ``"source"``.
        edge_target_col : str, optional
            Column name for target nodes. The default is ``"target"``.
        edge_weight_col : str, optional
            Column name for edge weights. The default is ``"weight"``.
        title : str, optional
            Chart title. The default is None.

        Returns
        -------
        bytes
            PNG octet payload of the rendered network chart.
        """
        return _render_network_chart_octet_from_source(
            csv_path=csv_path,
            table=table,
            edge_source_col=edge_source_col,
            edge_target_col=edge_target_col,
            edge_weight_col=edge_weight_col,
            title=title,
        )

    @mcp.tool()
    def plot_module(
        plot_module: str,
        params: dict[str, Any],
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> bytes:
        """Generate a chart for any supported plot module and return octets.

        Parameters
        ----------
        plot_module : str
            Plot module key to render.
        params : dict[str, Any]
            Module-specific plotting parameters excluding the DataFrame argument.
        csv_path : str, optional
            Path to a CSV file containing source data. The default is None.
        table : list[dict[str, Any]], optional
            In-memory row records with source columns as keys. The default is None.

        Returns
        -------
        bytes
            PNG octet payload of the rendered chart.
        """
        return render_plot_module_octet(
            plot_module=plot_module,
            params=params,
            csv_path=csv_path,
            table=table,
        )

    return mcp


def main() -> None:
    """Run the MCP server over stdio transport."""
    server = create_bubble_mcp_server()
    server.run(transport="stdio")


def main_bubble() -> None:
    """Run the bubble MCP entry point over stdio transport."""
    main()


def main_network() -> None:
    """Run the network MCP entry point over stdio transport."""
    main()


def main_bar() -> None:
    """Run the bar MCP entry point over stdio transport."""
    main()


def main_histogram() -> None:
    """Run the histogram MCP entry point over stdio transport."""
    main()


def main_box_violin() -> None:
    """Run the box/violin MCP entry point over stdio transport."""
    main()


def main_heatmap() -> None:
    """Run the heatmap MCP entry point over stdio transport."""
    main()


def main_area() -> None:
    """Run the area MCP entry point over stdio transport."""
    main()


def main_pie() -> None:
    """Run the pie MCP entry point over stdio transport."""
    main()


def main_waffle() -> None:
    """Run the waffle MCP entry point over stdio transport."""
    main()


def main_sankey() -> None:
    """Run the sankey MCP entry point over stdio transport."""
    main()


def main_table() -> None:
    """Run the table MCP entry point over stdio transport."""
    main()


def main_timeserie() -> None:
    """Run the timeserie MCP entry point over stdio transport."""
    main()


def main_wordcloud() -> None:
    """Run the wordcloud MCP entry point over stdio transport."""
    main()


def main_treemap() -> None:
    """Run the treemap MCP entry point over stdio transport."""
    main()


def main_sunburst() -> None:
    """Run the sunburst MCP entry point over stdio transport."""
    main()


if __name__ == "__main__":  # pragma: no cover
    main()
