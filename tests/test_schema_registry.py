"""Tests for schema-rich plugin discovery."""

from MatplotLibAPI.plugins import PluginContext


def _plot(pd_df, x: str, y: str, stacked: bool = False):
    return pd_df, x, y, stacked


def test_descriptor_is_inferred_from_signature() -> None:
    """Registered callables should expose deterministic parameter schemas."""
    context = PluginContext()
    context.register_plot(
        "bar",
        _plot,
        description="Render a bar chart.",
        aliases=("bars",),
    )

    descriptor = context.get_descriptor("bars")

    assert descriptor.name == "bar"
    assert descriptor.parameter_schema["required"] == ["x", "y"]
    assert descriptor.parameter_schema["properties"]["stacked"]["type"] == "boolean"
    assert context.get_plot("bars") is _plot


def test_openai_tools_are_generated_from_descriptors() -> None:
    """Agent tool definitions must come from the same canonical registry."""
    context = PluginContext()
    context.register_plot("bar", _plot, description="Render a bar chart.")

    tool = context.openai_tools()[0]

    assert tool["type"] == "function"
    assert tool["function"]["name"] == "plot_bar"
    assert tool["function"]["parameters"] == context.get_descriptor(
        "bar"
    ).parameter_schema
