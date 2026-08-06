"""Tests for the local command-line contract."""

import json

from MatplotLibAPI.cli import main


def test_cli_prints_plot_spec_schema(capsys) -> None:
    """The CLI should expose the same canonical schema as Python."""
    exit_code = main(["schema", "plot-spec"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["title"] == "MatplotLibAPI PlotSpec"


def test_cli_eval_is_local_and_deterministic(capsys) -> None:
    """The CLI evaluation command must not require external services."""
    exit_code = main(["eval"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["failed"] == 0
    assert output["llm_required"] is False


def test_cli_lists_presentation_presets(capsys) -> None:
    """The CLI should expose accessible and semantic presets."""
    exit_code = main(["presets", "list"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert "colorblind" in output["accessibility"]
    assert "currency" in output["number_formats"]


def test_cli_reports_compatibility_gate(capsys) -> None:
    """The CLI should expose the explicit 5.0 removal gate."""
    exit_code = main(["compatibility"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["v5"]["target"] == "5.0.0"
    assert output["plugins"]["canonical_plugin_api"] == "2"
