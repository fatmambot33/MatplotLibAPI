"""MCP server helpers for exposing MatplotLibAPI plotting tools.

This module provides a minimal Model Context Protocol (MCP) server focused on
bubble charts so language-model agents can request chart generation from either
CSV input or in-memory tabular records.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .bubble import aplot_bubble

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
    fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
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
    with NamedTemporaryFile(suffix=".png") as tmp_file:
        fig.savefig(tmp_file.name, format="png", dpi=300, bbox_inches="tight")
        tmp_file.seek(0)
        payload = tmp_file.read()
    plt.close(fig)
    return payload


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
        return render_bubble_chart_octet(
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

    return mcp


def main() -> None:
    """Run the bubble MCP server over stdio transport."""
    server = create_bubble_mcp_server()
    server.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
