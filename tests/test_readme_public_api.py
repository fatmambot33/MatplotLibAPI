"""Keep README imports aligned with the supported public API."""

from __future__ import annotations

import ast
from pathlib import Path

import MatplotLibAPI


README = Path(__file__).parents[1] / "README.md"
SPECIALIZED_MODULES = {
    "MatplotLibAPI.Pivot": {"plot_pivoted_bars"},
    "MatplotLibAPI.bubble": {"Bubble"},
    "MatplotLibAPI.network": {"NetworkGraph"},
}


def _python_blocks(markdown: str) -> list[str]:
    """Return Python fenced code blocks from Markdown text."""
    blocks: list[str] = []
    in_python_block = False
    current: list[str] = []

    for line in markdown.splitlines():
        if line.strip() == "```python":
            in_python_block = True
            current = []
            continue
        if in_python_block and line.strip() == "```":
            blocks.append("\n".join(current))
            in_python_block = False
            continue
        if in_python_block:
            current.append(line)

    return blocks


def test_readme_python_blocks_are_valid_syntax() -> None:
    """Ensure every documented Python example parses successfully."""
    for block in _python_blocks(README.read_text(encoding="utf-8")):
        ast.parse(block)


def test_readme_package_root_imports_are_public() -> None:
    """Ensure root imports only use names declared in ``__all__``."""
    public_names = set(MatplotLibAPI.__all__)

    for block in _python_blocks(README.read_text(encoding="utf-8")):
        tree = ast.parse(block)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "MatplotLibAPI":
                imported_names = {alias.name for alias in node.names}
                assert imported_names <= public_names


def test_readme_specialized_imports_are_intentional() -> None:
    """Limit documented module imports to the named specialized APIs."""
    for block in _python_blocks(README.read_text(encoding="utf-8")):
        tree = ast.parse(block)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not node.module or not node.module.startswith("MatplotLibAPI."):
                continue

            assert node.module in SPECIALIZED_MODULES
            imported_names = {alias.name for alias in node.names}
            assert imported_names <= SPECIALIZED_MODULES[node.module]
