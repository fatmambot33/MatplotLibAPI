"""Contract tests for the first-class plugin surface."""

import pytest

from MatplotLibAPI.plugins import (
    PLUGIN_API_VERSION,
    DuplicatePluginError,
    PluginContext,
    PluginError,
    PluginRegistry,
    create_registry,
)


class ExamplePlugin:
    """Small plugin used by the registry contract tests."""

    name = "example"
    api_version = PLUGIN_API_VERSION

    def setup(self, context: PluginContext) -> None:
        """Register one example plot."""
        context.register_plot("example", lambda: "ok")


def test_core_registry_exposes_stable_plots() -> None:
    """The built-in plugin should expose the stable plotting API."""
    registry = create_registry(discover=False)

    assert registry.list_plugins() == ["core"]
    assert "bar" in registry.context.list_plots()
    assert callable(registry.context.get_plot("bar"))


def test_custom_plugin_registration() -> None:
    """Third-party plugins should register deterministic plot names."""
    registry = PluginRegistry()
    registry.register(ExamplePlugin())

    assert registry.list_plugins() == ["example"]
    assert registry.context.get_plot("example")() == "ok"


def test_duplicate_plugin_is_rejected() -> None:
    """A plugin name may only be registered once."""
    registry = PluginRegistry()
    registry.register(ExamplePlugin())

    with pytest.raises(DuplicatePluginError):
        registry.register(ExamplePlugin())


def test_failed_setup_rolls_back_registered_plots() -> None:
    """Partial setup must not mutate the registry."""

    class BrokenPlugin:
        name = "broken"
        api_version = PLUGIN_API_VERSION

        def setup(self, context: PluginContext) -> None:
            context.register_plot("partial", lambda: None)
            raise RuntimeError("boom")

    registry = PluginRegistry()
    with pytest.raises(RuntimeError):
        registry.register(BrokenPlugin())

    assert registry.context.list_plots() == []


def test_unknown_plot_has_clear_error() -> None:
    """Unknown plot lookup should fail with a plugin-specific error."""
    with pytest.raises(PluginError, match="Unknown plot"):
        PluginContext().get_plot("missing")
