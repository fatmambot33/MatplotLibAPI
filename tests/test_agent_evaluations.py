"""Credential-free evaluation coverage for agent-facing behavior."""

from MatplotLibAPI.evaluations import run_agent_evaluations


def test_agent_evaluations_pass_without_external_services() -> None:
    """The deterministic baseline should pass without an LLM or credentials."""
    result = run_agent_evaluations()

    assert result["failed"] == 0
    assert result["llm_required"] is False
    assert result["credentials_required"] is False
