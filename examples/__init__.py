"""Global example fixtures and configuration."""

from __future__ import annotations

import sys
from pathlib import Path


# Ensure the src directory is on the Python path for src layout
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
