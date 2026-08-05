"""Regression tests for the stable package-root plotting contract."""

from inspect import Parameter, Signature, signature
from typing import get_type_hints

from matplotlib.figure import Figure as MatplotlibFigure
from plotly.graph_objects import Figure as PlotlyFigure

import MatplotLibAPI


PLOT_HELPERS = {
    name: getattr(MatplotLibAPI, name)
    for name in MatplotLibAPI.__all__
    if name.startswith("fplot_")
}
FIGURE_TYPES = {MatplotlibFigure, PlotlyFigure}


def test_public_plot_helpers_use_dataframe_first() -> None:
    """Require one predictable DataFrame entry point for every figure helper."""
    for name, helper in PLOT_HELPERS.items():
        parameters = list(signature(helper).parameters.values())

        assert parameters, f"{name} must accept a DataFrame"
        assert parameters[0].name == "pd_df", f"{name} must start with pd_df"
        assert parameters[0].kind in {
            Parameter.POSITIONAL_ONLY,
            Parameter.POSITIONAL_OR_KEYWORD,
        }


def test_public_plot_helpers_return_figures() -> None:
    """Keep the package-root return policy explicit and machine-readable."""
    for name, helper in PLOT_HELPERS.items():
        hints = get_type_hints(helper)

        assert hints.get("return") in FIGURE_TYPES, f"{name} must return a Figure"


def test_public_plot_helpers_have_explicit_signatures() -> None:
    """Prevent generic variadic wrappers from replacing documented APIs."""
    for name, helper in PLOT_HELPERS.items():
        helper_signature: Signature = signature(helper)
        parameter_kinds = {
            parameter.kind for parameter in helper_signature.parameters.values()
        }

        assert (
            Parameter.VAR_POSITIONAL not in parameter_kinds
        ), f"{name} must expose named parameters instead of *args"
