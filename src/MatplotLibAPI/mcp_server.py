"""MCP server helpers backed by the canonical plotting executor."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .bubble import Bubble
from .executor import (
    RenderPolicy,
    execute_plot,
    inspect_dataframe,
    recommend_plot,
)
from .intelligence import suggest_plot_spec_repairs
from .migration import v5_compatibility_status
from .mcp.metadata import (
    DEDICATED_PLOT_TOOLS,
    PLOT_MODULE_PARAMETER_HINTS,
    SHARED_INPUT_CONTRACT,
)
from .mcp.renderers import SUPPORTED_PLOT_MODULES
from .network import fplot_network
from .specs import PlotSpec
from .style_template import BUBBLE_STYLE_TEMPLATE, StyleTemplate

TableRecords = List[Dict[str, Any]]
_MCP_ALIASES = {
    "histogram": "histogram_kde",
    "pie": "pie_donut",
    "timeserie": "timeseries",
}


def _load_dataframe(
    csv_path: Optional[str] = None,
    table: Optional[TableRecords] = None,
) -> pd.DataFrame:
    """Load plotting data from either a CSV file or table records."""
    if csv_path is None and table is None:
        raise ValueError("Provide either `csv_path` or `table`.")
    if csv_path is not None and table is not None:
        raise ValueError("Provide either `csv_path` or `table`, not both.")
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
    style: StyleTemplate = BUBBLE_STYLE_TEMPLATE,
    hline: bool = False,
    vline: bool = False,
) -> Figure:
    """Create a bubble chart figure from tabular input."""
    pd_df = _load_dataframe(csv_path=csv_path, table=table)
    return Bubble(
        pd_df=pd_df,
        label=label,
        x=x,
        y=y,
        z=z,
        max_values=max_values,
        center_to_mean=center_to_mean,
        sort_by=sort_by,
        ascending=ascending,
    ).fplot(title=title, hline=hline, vline=vline, style=style)


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
    """Render a bubble chart and write a PNG to disk."""
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
    """Render a bubble chart and return PNG bytes."""
    figure = _build_bubble_chart_figure(
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
    return _figure_to_png_bytes(figure)


def render_network_chart_octet(
    csv_path: Optional[str] = None,
    table: Optional[TableRecords] = None,
    edge_source_col: str = "source",
    edge_target_col: str = "target",
    edge_weight_col: str = "weight",
    title: Optional[str] = None,
) -> bytes:
    """Render a network chart and return PNG bytes."""
    figure = _build_network_chart_figure(
        csv_path=csv_path,
        table=table,
        edge_source_col=edge_source_col,
        edge_target_col=edge_target_col,
        edge_weight_col=edge_weight_col,
        title=title,
    )
    return _figure_to_png_bytes(figure)


def get_plot_module_metadata() -> Dict[str, Any]:
    """Return MCP and canonical registry discovery metadata."""
    from .plugins import create_registry

    registry = create_registry(discover=False)
    return {
        "supported_plot_modules": SUPPORTED_PLOT_MODULES,
        "shared_input_contract": SHARED_INPUT_CONTRACT,
        "parameter_hints": PLOT_MODULE_PARAMETER_HINTS,
        "dedicated_tools": DEDICATED_PLOT_TOOLS,
        "plot_descriptors": registry.context.list_descriptors(),
        "openai_tools": registry.context.openai_tools(),
        "plugin_compatibility": registry.compatibility_report(),
        "v5_compatibility": v5_compatibility_status(),
    }


def inspect_plot_data(
    csv_path: Optional[str] = None,
    table: Optional[TableRecords] = None,
) -> Dict[str, Any]:
    """Return bounded deterministic profiling metadata for MCP clients."""
    return inspect_dataframe(_load_dataframe(csv_path=csv_path, table=table))


def recommend_plot_data(
    csv_path: Optional[str] = None,
    table: Optional[TableRecords] = None,
) -> Dict[str, Any]:
    """Return ranked deterministic chart recommendations for MCP clients."""
    return recommend_plot(_load_dataframe(csv_path=csv_path, table=table))


def repair_plot_spec_data(
    spec: Dict[str, Any],
    csv_path: Optional[str] = None,
    table: Optional[TableRecords] = None,
) -> Dict[str, Any]:
    """Return opt-in PlotSpec repair suggestions without mutating input."""
    from .plugins import create_registry

    resolved_spec = PlotSpec.from_dict(spec)
    frame = _load_dataframe(csv_path=csv_path, table=table)
    suggestions = suggest_plot_spec_repairs(
        resolved_spec,
        frame,
        registry=create_registry(),
    )
    return {
        "spec": resolved_spec.to_dict(),
        "suggestions": [item.to_dict() for item in suggestions],
        "applied": False,
    }


def render_plot_module_octet(
    plot_module: str,
    params: Dict[str, Any],
    csv_path: Optional[str] = None,
    table: Optional[TableRecords] = None,
) -> bytes:
    """Render any supported module through one validated execution contract."""
    if plot_module == "bubble":
        return render_bubble_chart_octet(csv_path=csv_path, table=table, **params)
    if plot_module == "network":
        return render_network_chart_octet(csv_path=csv_path, table=table, **params)
    if plot_module not in SUPPORTED_PLOT_MODULES:
        raise ValueError(
            f"Unsupported plot_module '{plot_module}'. Supported: "
            f"{SUPPORTED_PLOT_MODULES}"
        )

    frame = _load_dataframe(csv_path=csv_path, table=table)
    chart = _MCP_ALIASES.get(plot_module, plot_module)
    spec = PlotSpec.from_dict(
        {
            "chart": chart,
            "options": params,
            "output": {"format": "png", "dpi": 300},
        }
    )
    result = execute_plot(
        spec,
        frame,
        policy=RenderPolicy(
            workspace=Path.cwd(),
            allow_absolute_paths=True,
        ),
    )
    if result.payload is None:  # pragma: no cover - executor contract guard
        raise RuntimeError("The plotting executor returned no PNG payload.")
    return result.payload


def create_mcp_server() -> Any:
    """Create an MCP server exposing schema-backed plotting tools."""
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
        params: Dict[str, Any],
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> bytes:
        """Generate a chart for any supported plot module."""
        return render_plot_module_octet(
            plot_module=plot_module,
            params=params,
            csv_path=csv_path,
            table=table,
        )

    @mcp.tool()
    def inspect_data(
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> Dict[str, Any]:
        """Profile local tabular data with deterministic bounded summaries."""
        return inspect_plot_data(csv_path=csv_path, table=table)

    @mcp.tool()
    def recommend_chart(
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> Dict[str, Any]:
        """Recommend charts with scores, reasons, and warnings."""
        return recommend_plot_data(csv_path=csv_path, table=table)

    @mcp.tool()
    def suggest_plot_repairs(
        spec: Dict[str, Any],
        csv_path: Optional[str] = None,
        table: Optional[TableRecords] = None,
    ) -> Dict[str, Any]:
        """Suggest safe opt-in PlotSpec repairs."""
        return repair_plot_spec_data(
            spec,
            csv_path=csv_path,
            table=table,
        )

    @mcp.tool()
    def compatibility_status() -> Dict[str, Any]:
        """Return plugin and 5.0 migration compatibility gates."""
        from .plugins import create_registry

        registry = create_registry()
        return {
            "plugins": registry.compatibility_report(),
            "v5": v5_compatibility_status(),
        }

    @mcp.tool()
    def describe_plot_modules() -> Dict[str, Any]:
        """Describe MCP tools and canonical plot schemas."""
        return get_plot_module_metadata()

    def make_handler(module_name: str) -> Any:
        """Create one module-specific MCP tool handler."""

        def handler(
            params: Dict[str, Any],
            csv_path: Optional[str] = None,
            table: Optional[TableRecords] = None,
        ) -> bytes:
            """Render one dedicated plot module through the executor."""
            return render_plot_module_octet(
                plot_module=module_name,
                params=params,
                csv_path=csv_path,
                table=table,
            )

        return handler

    for tool_name, module_name in sorted(DEDICATED_PLOT_TOOLS.items()):
        if tool_name in {"plot_bubble", "plot_network"}:
            continue
        handler = make_handler(module_name)
        handler.__name__ = tool_name
        handler.__doc__ = f"Generate a {module_name} chart and return PNG octets."
        mcp.tool()(handler)

    return mcp


def main() -> None:
    """Run the MCP server over stdio transport."""
    create_mcp_server().run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
