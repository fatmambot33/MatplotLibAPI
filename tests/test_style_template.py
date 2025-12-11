"""Tests for :class:`MatplotLibAPI.StyleTemplate.StyleTemplate`."""

from MatplotLibAPI.StyleTemplate import StyleTemplate


def test_font_mapping_scales_with_small_fonts() -> None:
    """Ensure mapping stays readable when the base font is very small."""

    style = StyleTemplate(font_size=2)

    mapping = style.font_mapping

    assert mapping[0] == 1
    assert mapping[2] == 2
    assert mapping[4] >= mapping[3] >= mapping[2]


def test_font_mapping_uses_dynamic_steps() -> None:
    """Check that font mapping grows symmetrically around the base size."""

    style = StyleTemplate(font_size=20)

    mapping = style.font_mapping

    assert mapping[2] == 20
    assert mapping[1] < mapping[2] < mapping[3]
    assert mapping[2] - mapping[0] == mapping[4] - mapping[2]
