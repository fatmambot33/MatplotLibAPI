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
