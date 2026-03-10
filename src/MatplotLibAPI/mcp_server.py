"""MCP server helpers for exposing MatplotLibAPI plotting tools.

This module provides a minimal Model Context Protocol (MCP) server focused on
bubble charts so language-model agents can request chart generation from CSV
input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .bubble import fplot_bubble


def render_bubble_chart(
    csv_path: str,
    output_path: str,
    label: str,
    x: str,
    y: str,
    z: str,
    title: Optional[str] = None,
    max_values: int = 50,
    center_to_mean: bool = False,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    hline: bool = False,
    vline: bool = False,
) -> str:
    """Render a bubble chart from CSV data and write it to disk.

    Parameters
    ----------
    csv_path : str
        Path to a CSV file containing the plotting data.
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
    data_path = Path(csv_path).expanduser().resolve()
    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pd_df = pd.read_csv(data_path)
    fplot_bubble(
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
    )

    return str(out_path)


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
    def plot_bubble_from_csv(
        csv_path: str,
        output_path: str,
        label: str,
        x: str,
        y: str,
        z: str,
        title: Optional[str] = None,
        max_values: int = 50,
        center_to_mean: bool = False,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        hline: bool = False,
        vline: bool = False,
    ) -> str:
        """Generate a bubble chart image from CSV data.

        Parameters
        ----------
        csv_path : str
            Path to the source CSV file.
        output_path : str
            Destination file path for the chart image.
        label : str
            Column name used for bubble labels.
        x : str
            Column name for x-axis values.
        y : str
            Column name for y-axis values.
        z : str
            Column name used for bubble size.
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
        str
            The output path where the chart image was written.
        """
        return render_bubble_chart(
            csv_path=csv_path,
            output_path=output_path,
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
