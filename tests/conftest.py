"""Shared fixtures for visualization tests."""

import os
from pathlib import Path
from typing import Any, Callable, Generator

import pandas as pd
import pytest

from scripts import generate_sample_data


@pytest.fixture(scope="session")
def sample_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate sample CSV files in an isolated directory for tests."""
    base_dir = tmp_path_factory.mktemp("sample_data")
    cwd = os.getcwd()
    os.chdir(base_dir)
    try:
        generate_sample_data.main()
    finally:
        os.chdir(cwd)

    return base_dir / "data"


@pytest.fixture(scope="session")
def load_sample_df(
    sample_data_dir: Path,
) -> Callable[..., pd.DataFrame]:
    """Return a loader that reads generated sample CSVs into dataframes."""

    def _loader(filename: str, **kwargs: Any) -> pd.DataFrame:
        return pd.read_csv(sample_data_dir / filename, **kwargs)

    return _loader


@pytest.fixture(autouse=True)
def close_matplotlib_figures() -> Generator[None, None, None]:
    """Close any matplotlib figures created during a test."""
    yield

    import matplotlib.pyplot as plt

    plt.close("all")
