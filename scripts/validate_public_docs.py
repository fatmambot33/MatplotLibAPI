"""Validate documentation coverage for the supported public API."""

from pathlib import Path

import MatplotLibAPI


def main() -> None:
    """Fail when a supported package-root export is undocumented."""
    reference = Path("docs/API_REFERENCE.md").read_text(encoding="utf-8")
    missing = [name for name in MatplotLibAPI.__all__ if f"`{name}`" not in reference]
    if missing:
        raise SystemExit(f"Undocumented public exports: {', '.join(missing)}")


if __name__ == "__main__":
    main()
