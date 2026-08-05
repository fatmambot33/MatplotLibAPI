"""Release-level public API, documentation, and compatibility contracts."""

from pathlib import Path

import MatplotLibAPI


def test_every_public_export_is_importable_and_documented() -> None:
    """Keep the package-root API and generated reference synchronized."""
    reference = Path("docs/API_REFERENCE.md").read_text(encoding="utf-8")
    for name in MatplotLibAPI.__all__:
        assert getattr(MatplotLibAPI, name) is not None
        assert f"`{name}`" in reference


def test_documented_legacy_module_imports_match_root_exports() -> None:
    """Preserve supported 4.0.x module imports through the 4.x line."""
    from MatplotLibAPI.area import fplot_area
    from MatplotLibAPI.bar import fplot_bar
    from MatplotLibAPI.heatmap import fplot_heatmap
    from MatplotLibAPI.histogram import fplot_histogram
    from MatplotLibAPI.pie import fplot_pie

    assert fplot_area is MatplotLibAPI.fplot_area
    assert fplot_bar is MatplotLibAPI.fplot_bar
    assert fplot_heatmap is MatplotLibAPI.fplot_heatmap
    assert fplot_histogram is MatplotLibAPI.fplot_histogram_kde
    assert fplot_pie is MatplotLibAPI.fplot_pie_donut


def test_compatibility_policy_defines_removal_boundary() -> None:
    """Require actionable deprecation and removal guidance."""
    policy = Path("docs/COMPATIBILITY.md").read_text(encoding="utf-8")
    assert "DeprecationWarning" in policy
    assert "major release" in policy
    assert "4.0.x" in policy
