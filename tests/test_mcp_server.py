"""Tests for MCP server helpers."""

from pathlib import Path

import pytest

from MatplotLibAPI.mcp_server import (
    render_bubble_chart,
    render_bubble_chart_octet,
    render_network_chart_octet,
)


def test_render_bubble_chart_from_csv(load_sample_df, tmp_path: Path):
    """Render a bubble chart image using CSV input."""
    df = load_sample_df("bubble.csv")
    df["score"] = df["population"] / 10
    csv_path = tmp_path / "bubble.csv"
    out_path = tmp_path / "charts" / "bubble.png"
    df.to_csv(csv_path, index=False)

    result = render_bubble_chart(
        csv_path=str(csv_path),
        output_path=str(out_path),
        label="country",
        x="gdp_per_capita",
        y="population",
        z="score",
        title="Bubble via MCP",
    )

    assert Path(result).exists()
    assert Path(result).suffix == ".png"


def test_render_bubble_chart_accepts_table_input(load_sample_df, tmp_path: Path):
    """Render a bubble chart image to disk using table records."""
    df = load_sample_df("bubble.csv")
    df["score"] = df["population"] / 10
    out_path = tmp_path / "charts" / "bubble-table.png"

    result = render_bubble_chart(
        table=df.to_dict(orient="records"),
        output_path=str(out_path),
        label="country",
        x="gdp_per_capita",
        y="population",
        z="score",
        title="Bubble via MCP",
    )

    assert Path(result).exists()
    assert Path(result).suffix == ".png"


def test_render_bubble_chart_octet_returns_png_payload(load_sample_df, tmp_path: Path):
    """Return PNG bytes for MCP octet-stream responses using CSV input."""
    df = load_sample_df("bubble.csv")
    df["score"] = df["population"] / 10
    csv_path = tmp_path / "bubble.csv"
    df.to_csv(csv_path, index=False)

    payload = render_bubble_chart_octet(
        csv_path=str(csv_path),
        label="country",
        x="gdp_per_capita",
        y="population",
        z="score",
        title="Bubble via MCP",
    )

    assert payload.startswith(b"\x89PNG\r\n\x1a\n")


def test_render_bubble_chart_octet_accepts_table_input(load_sample_df):
    """Return PNG bytes for MCP octet-stream responses using table records."""
    df = load_sample_df("bubble.csv")
    df["score"] = df["population"] / 10

    payload = render_bubble_chart_octet(
        table=df.to_dict(orient="records"),
        label="country",
        x="gdp_per_capita",
        y="population",
        z="score",
        title="Bubble via MCP",
    )

    assert payload.startswith(b"\x89PNG\r\n\x1a\n")


def test_render_bubble_chart_octet_requires_input_source():
    """Raise an error when no CSV path or table input is provided."""
    with pytest.raises(ValueError, match="Provide either `csv_path` or `table`"):
        render_bubble_chart_octet(
            label="country",
            x="gdp_per_capita",
            y="population",
            z="score",
        )


def test_render_network_chart_octet_returns_png_payload(load_sample_df, tmp_path: Path):
    """Return PNG bytes for network octet responses using CSV input."""
    df = load_sample_df("network.csv")
    csv_path = tmp_path / "network.csv"
    df.to_csv(csv_path, index=False)

    payload = render_network_chart_octet(
        csv_path=str(csv_path),
        edge_source_col="city_a",
        edge_target_col="city_b",
        edge_weight_col="distance_km",
        title="Network via MCP",
    )

    assert payload.startswith(b"\x89PNG\r\n\x1a\n")


def test_render_network_chart_octet_accepts_table_input(load_sample_df):
    """Return PNG bytes for network octet responses using table records."""
    df = load_sample_df("network.csv")

    payload = render_network_chart_octet(
        table=df.to_dict(orient="records"),
        edge_source_col="city_a",
        edge_target_col="city_b",
        edge_weight_col="distance_km",
        title="Network via MCP",
    )

    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
