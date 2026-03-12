"""MCP server helpers for exposing MatplotLibAPI plotting tools."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional

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
Renderer = Callable[..., Any]

SHARED_INPUT_CONTRACT: dict[str, str] = {
    "csv_path": "Path to a CSV file with source data.",
    "table": "In-memory table as list[dict[str, Any]] records.",
    "return": "PNG bytes suitable for application/octet-stream.",
}

PLOT_MODULE_PARAMETER_HINTS: dict[str, dict[str, str]] = {
    "bubble": {
        "label": "Bubble label column name.",
        "x": "X-axis column name.",
        "y": "Y-axis column name.",
        "z": "Bubble size column name.",
    },
    "network": {
        "edge_source_col": "Source-node column name (default: source).",
        "edge_target_col": "Target-node column name (default: target).",
        "edge_weight_col": "Edge-weight column name (default: weight).",
    },
    "bar": {
        "category": "Category column name.",
        "value": "Numeric value column name.",
    },
    "histogram": {"value": "Numeric value column name."},
    "box_violin": {"y": "Numeric value column name."},
    "heatmap": {
        "x": "X-axis category column name.",
        "y": "Y-axis category column name.",
        "value": "Cell value column name.",
    },
    "correlation_matrix": {
        "features": "List of numeric columns used in correlation matrix.",
    },
    "area": {
        "x": "X-axis column name.",
        "y": "List of stacked y-axis columns.",
    },
    "pie": {
        "label": "Category label column name.",
        "value": "Numeric value column name.",
    },
    "waffle": {
        "label": "Category label column name.",
        "value": "Numeric value column name.",
    },
    "sankey": {
        "source": "Source-node column name.",
        "target": "Target-node column name.",
        "value": "Flow/weight column name.",
    },
    "table": {},
    "timeserie": {
        "x": "Datetime or ordered x-axis column name.",
        "y": "List of y-axis series columns.",
    },
    "wordcloud": {
        "text_column": "Text/token column name.",
        "weight_column": "Weight/frequency column name (optional).",
    },
    "treemap": {
        "path": "Hierarchy column names in order.",
        "values": "Numeric value column name.",
    },
    "sunburst": {
        "path": "Hierarchy column names in order.",
        "values": "Numeric value column name.",
    },
}


def _load_dataframe(
    csv_path: Optional[str] = None,
    table: Optional[TableRecords] = None,
) -> pd.DataFrame:
    """Load plotting data from either a CSV file or table records."""
    if csv_path is None and table is None:
        raise ValueError("Provide either `csv_path` or `table`.")

    if table is not None:
        return pd.DataFrame(table)

    data_path = Path(str(csv_path)).expanduser().resolve()
    return pd.read_csv(data_path)


def _figure_to_png_bytes(fig: Figure) -> bytes:
    """Serialize a Matplotlib figure to PNG bytes and close it."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


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
    """Create a bubble chart figure from tabular input."""
    pd_df = _load_dataframe(csv_path=csv_path, table=table)
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


def _build_network_chart_figure(
    csv_path: Optional[str] = None,
    table: Optional[TableRecords] = None,
    edge_source_col: str = "source",
    edge_target_col: str = "target",
    edge_weight_col: str = "weight",
    title: Optional[str] = None,
) -> Figure:
    """Create a network chart figure from tabular input."""
    pd_df = _load_dataframe(csv_path=csv_path, table=table)
    return fplot_network(
        pd_df=pd_df,
        edge_source_col=edge_source_col,
        edge_target_col=edge_target_col,
        edge_weight_col=edge_weight_col,
        title=title,
    )


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

    out_path.write_bytes(
        render_bubble_chart_octet(
            label=label,
            x=x,
            y=y,
            z=z,
            csv_path=csv_path,
            table=table,
            title=title,
            max_values=max_values,
            center_to_mean=center_to_mean,
            sort_by=sort_by,
            ascending=ascending,
            hline=hline,
            vline=vline,
        )
    )
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
    fig = _build_bubble_chart_figure(
        label=label,
        x=x,
        y=y,
        z=z,
        csv_path=csv_path,
        table=table,
        title=title,
        max_values=max_values,
        center_to_mean=center_to_mean,
        sort_by=sort_by,
        ascending=ascending,
        hline=hline,
        vline=vline,
    )
    return _figure_to_png_bytes(fig)


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
    fig = _build_network_chart_figure(
        csv_path=csv_path,
        table=table,
        edge_source_col=edge_source_col,
        edge_target_col=edge_target_col,
        edge_weight_col=edge_weight_col,
        title=title,
    )
    return _figure_to_png_bytes(fig)


_MATPLOTLIB_RENDERERS: dict[str, Renderer] = {
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

_PLOTLY_RENDERERS: dict[str, Renderer] = {
    "sankey": fplot_sankey,
    "treemap": fplot_treemap,
    "sunburst": fplot_sunburst,
}

_SUPPORTED_PLOT_MODULES = sorted(
    ["bubble", "network", *_MATPLOTLIB_RENDERERS.keys(), *_PLOTLY_RENDERERS.keys()]
)


_DEDICATED_PLOT_TOOLS = {
    "plot_bubble": "bubble",
    "plot_network": "network",
    "plot_bar": "bar",
    "plot_histogram": "histogram",
    "plot_box_violin": "box_violin",
    "plot_heatmap": "heatmap",
    "plot_correlation_matrix": "correlation_matrix",
    "plot_area": "area",
    "plot_pie": "pie",
    "plot_waffle": "waffle",
    "plot_sankey": "sankey",
    "plot_table": "table",
    "plot_timeserie": "timeserie",
    "plot_wordcloud": "wordcloud",
    "plot_treemap": "treemap",
    "plot_sunburst": "sunburst",
}


def get_plot_module_metadata() -> dict[str, Any]:
    """Return MCP metadata for module discoverability and exploration.

    Returns
    -------
    dict[str, Any]
        Discoverability metadata containing supported modules, shared input
        contract, and parameter hints per module.
    """
    return {
        "supported_plot_modules": _SUPPORTED_PLOT_MODULES,
        "shared_input_contract": SHARED_INPUT_CONTRACT,
        "parameter_hints": PLOT_MODULE_PARAMETER_HINTS,
        "dedicated_tools": _DEDICATED_PLOT_TOOLS,
    }


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
        Plot module key.
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
    if plot_module == "bubble":
        return render_bubble_chart_octet(csv_path=csv_path, table=table, **params)
    if plot_module == "network":
        return render_network_chart_octet(csv_path=csv_path, table=table, **params)

    pd_df = _load_dataframe(csv_path=csv_path, table=table)

    if plot_module in _MATPLOTLIB_RENDERERS:
        fig = _MATPLOTLIB_RENDERERS[plot_module](pd_df=pd_df, **params)
        return _figure_to_png_bytes(fig)

    if plot_module in _PLOTLY_RENDERERS:
        fig = _PLOTLY_RENDERERS[plot_module](pd_df=pd_df, **params)
        return bytes(fig.to_image(format="png"))

    raise ValueError(
        f"Unsupported plot_module '{plot_module}'. Supported: {_SUPPORTED_PLOT_MODULES}"
    )


def create_bubble_mcp_server() -> Any:
    """Create an MCP server exposing MatplotLibAPI plotting tools.

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
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install MatplotLibAPI with MCP support: `pip install MatplotLibAPI[mcp]`."
        ) from exc

    mcp = FastMCP("MatplotLibAPI")

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
        """Generate a bubble chart and return PNG octets."""
        return render_bubble_chart_octet(
            label=label,
            x=x,
            y=y,
            z=z,
            csv_path=csv_path,
            table=table,
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
        """Generate a network chart and return PNG octets."""
        return render_network_chart_octet(
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
        """Generate a chart for any supported plot module and return octets."""
        return render_plot_module_octet(
            plot_module=plot_module,
            params=params,
            csv_path=csv_path,
            table=table,
        )

    def _render_module(
        module_name: str,
        params: dict[str, Any],
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> bytes:
        """Render a module-specific chart as PNG octets."""
        return render_plot_module_octet(
            plot_module=module_name,
            params=params,
            csv_path=csv_path,
            table=table,
        )

    @mcp.tool()
    def plot_bar(
        params: dict[str, Any],
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> bytes:
        """Generate a bar chart and return PNG octets."""
        return _render_module("bar", params=params, csv_path=csv_path, table=table)

    @mcp.tool()
    def plot_histogram(
        params: dict[str, Any],
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> bytes:
        """Generate a histogram chart and return PNG octets."""
        return _render_module(
            "histogram", params=params, csv_path=csv_path, table=table
        )

    @mcp.tool()
    def plot_box_violin(
        params: dict[str, Any],
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> bytes:
        """Generate a box/violin chart and return PNG octets."""
        return _render_module(
            "box_violin", params=params, csv_path=csv_path, table=table
        )

    @mcp.tool()
    def plot_heatmap(
        params: dict[str, Any],
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> bytes:
        """Generate a heatmap chart and return PNG octets."""
        return _render_module("heatmap", params=params, csv_path=csv_path, table=table)

    @mcp.tool()
    def plot_correlation_matrix(
        params: dict[str, Any],
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> bytes:
        """Generate a correlation matrix chart and return PNG octets."""
        return _render_module(
            "correlation_matrix", params=params, csv_path=csv_path, table=table
        )

    @mcp.tool()
    def plot_area(
        params: dict[str, Any],
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> bytes:
        """Generate an area chart and return PNG octets."""
        return _render_module("area", params=params, csv_path=csv_path, table=table)

    @mcp.tool()
    def plot_pie(
        params: dict[str, Any],
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> bytes:
        """Generate a pie chart and return PNG octets."""
        return _render_module("pie", params=params, csv_path=csv_path, table=table)

    @mcp.tool()
    def plot_waffle(
        params: dict[str, Any],
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> bytes:
        """Generate a waffle chart and return PNG octets."""
        return _render_module("waffle", params=params, csv_path=csv_path, table=table)

    @mcp.tool()
    def plot_sankey(
        params: dict[str, Any],
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> bytes:
        """Generate a sankey chart and return PNG octets."""
        return _render_module("sankey", params=params, csv_path=csv_path, table=table)

    @mcp.tool()
    def plot_table(
        params: dict[str, Any],
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> bytes:
        """Generate a table chart and return PNG octets."""
        return _render_module("table", params=params, csv_path=csv_path, table=table)

    @mcp.tool()
    def plot_timeserie(
        params: dict[str, Any],
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> bytes:
        """Generate a timeserie chart and return PNG octets."""
        return _render_module(
            "timeserie", params=params, csv_path=csv_path, table=table
        )

    @mcp.tool()
    def plot_wordcloud(
        params: dict[str, Any],
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> bytes:
        """Generate a wordcloud chart and return PNG octets."""
        return _render_module(
            "wordcloud", params=params, csv_path=csv_path, table=table
        )

    @mcp.tool()
    def plot_treemap(
        params: dict[str, Any],
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> bytes:
        """Generate a treemap chart and return PNG octets."""
        return _render_module("treemap", params=params, csv_path=csv_path, table=table)

    @mcp.tool()
    def plot_sunburst(
        params: dict[str, Any],
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> bytes:
        """Generate a sunburst chart and return PNG octets."""
        return _render_module("sunburst", params=params, csv_path=csv_path, table=table)

    @mcp.tool()
    def describe_plot_modules() -> dict[str, Any]:
        """Describe MCP plot-module capabilities for tool exploration.

        Returns
        -------
        dict[str, Any]
            Metadata including supported module names, shared input contract,
            and parameter-hint dictionaries.
        """
        return get_plot_module_metadata()

    return mcp


def main() -> None:
    """Run the MCP server over stdio transport."""
    server = create_bubble_mcp_server()
    server.run(transport="stdio")


main_bubble = main
main_network = main
main_bar = main
main_histogram = main
main_box_violin = main
main_heatmap = main
main_area = main
main_pie = main
main_waffle = main
main_sankey = main
main_table = main
main_timeserie = main
main_wordcloud = main
main_treemap = main
main_sunburst = main


if __name__ == "__main__":  # pragma: no cover
    main()
