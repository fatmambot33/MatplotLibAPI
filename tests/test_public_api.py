"""Regression tests for the supported package-root API."""

import MatplotLibAPI


def test_public_api_is_explicit() -> None:
    """Expose only the documented package-root symbols."""
    assert MatplotLibAPI.__all__ == ["DataFrameAccessor", "CorrelationMethod"]


def test_public_api_symbols_are_importable() -> None:
    """Keep supported root imports available to downstream users."""
    for name in MatplotLibAPI.__all__:
        assert getattr(MatplotLibAPI, name) is not None
