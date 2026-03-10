"""Tests for MCP server helpers."""

from pathlib import Path

from MatplotLibAPI.mcp_server import render_bubble_chart


def test_render_bubble_chart_from_csv(load_sample_df, tmp_path: Path):
    """Render a bubble chart image using the MCP helper function."""
    df = load_sample_df("bubble.csv")
    csv_path = tmp_path / "bubble.csv"
    out_path = tmp_path / "charts" / "bubble.png"
    df.to_csv(csv_path, index=False)

    result = render_bubble_chart(
        csv_path=str(csv_path),
        output_path=str(out_path),
        label="country",
        x="gdp_per_capita",
        y="population",
        z="population",
        title="Bubble via MCP",
    )

    assert Path(result).exists()
    assert Path(result).suffix == ".png"
