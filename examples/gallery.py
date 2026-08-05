"""Generate a compact headless gallery for every public plotting helper."""

from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.figure import Figure as MatplotlibFigure
from plotly.graph_objects import Figure as PlotlyFigure

matplotlib.use("Agg")

from MatplotLibAPI import (  # noqa: E402
    fplot_area,
    fplot_bar,
    fplot_box_violin,
    fplot_correlation_matrix,
    fplot_heatmap,
    fplot_histogram_kde,
    fplot_pie_donut,
    fplot_sankey,
    fplot_sunburst,
    fplot_table,
    fplot_timeserie,
    fplot_treemap,
    fplot_waffle,
    fplot_wordcloud,
)


def _save(name: str, figure: object, output_dir: Path) -> None:
    """Save a Matplotlib or Plotly figure using a deterministic filename."""
    if isinstance(figure, MatplotlibFigure):
        figure.savefig(output_dir / f"{name}.png", dpi=100)
        return
    if isinstance(figure, PlotlyFigure):
        (output_dir / f"{name}.html").write_text(
            figure.to_html(include_plotlyjs="cdn"), encoding="utf-8"
        )
        return
    raise TypeError(f"Unsupported gallery figure: {type(figure)!r}")


def main() -> None:
    """Render one deterministic example for every public plotting helper."""
    output_dir = Path("build/gallery")
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = pd.DataFrame(
        {
            "category": ["A", "A", "B", "B"],
            "group": ["North", "South", "North", "South"],
            "value": [12, 9, 15, 11],
        }
    )
    timeline = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-01-01", "2026-02-01"]),
            "group": ["A", "A", "B", "B"],
            "value": [10, 14, 8, 13],
        }
    )
    hierarchy = pd.DataFrame(
        {"labels": ["All", "A", "B"], "parents": ["", "All", "All"], "values": [30, 10, 20]}
    )
    words = pd.DataFrame({"word": ["simple", "reliable", "typed"], "weight": [5, 3, 2]})
    flows = pd.DataFrame(
        {"source": ["Visit", "Visit"], "target": ["Buy", "Leave"], "value": [35, 65]}
    )

    figures = {
        "area": fplot_area(categories, x="category", y="value", label="group", stacked=True),
        "bar": fplot_bar(categories, category="category", value="value", group="group", stacked=True),
        "box_violin": fplot_box_violin(categories, column="value", category="category", use_violin=True),
        "correlation": fplot_correlation_matrix(categories[["value"]].assign(other=[1, 2, 3, 4])),
        "heatmap": fplot_heatmap(categories, index="category", columns="group", values="value"),
        "histogram": fplot_histogram_kde(categories, column="value", bins=4, kde=True),
        "pie": fplot_pie_donut(categories.groupby("category", as_index=False)["value"].sum(), category="category", value="value", donut=True),
        "sankey": fplot_sankey(flows, source="source", target="target", value="value"),
        "sunburst": fplot_sunburst(hierarchy, labels="labels", parents="parents", values="values"),
        "table": fplot_table(pd_df=categories, cols=["category", "group", "value"]),
        "timeserie": fplot_timeserie(pd_df=timeline, label="group", x="date", y="value"),
        "treemap": fplot_treemap(pd_df=hierarchy, path="labels", values="values"),
        "waffle": fplot_waffle(categories.groupby("category", as_index=False)["value"].sum(), category="category", value="value"),
        "wordcloud": fplot_wordcloud(words, text_column="word", weight_column="weight"),
    }

    for name, figure in figures.items():
        _save(name, figure, output_dir)


if __name__ == "__main__":
    main()
