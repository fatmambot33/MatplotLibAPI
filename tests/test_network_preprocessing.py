"""Tests for network edge-list preprocessing helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from MatplotLibAPI.network import trim_low_degree_nodes


def test_trim_low_degree_nodes_single_pass():
    """Trim low-degree nodes using one-pass degree computation."""
    edge_df = pd.DataFrame(
        {
            "source": ["A", "B", "C"],
            "target": ["B", "C", "D"],
            "weight": [1, 2, 3],
        }
    )

    filtered = trim_low_degree_nodes(edge_df, min_degree=2, recursive=False)

    assert filtered[["source", "target"]].to_dict(orient="records") == [
        {"source": "B", "target": "C"}
    ]


def test_trim_low_degree_nodes_recursive_until_stable():
    """Recursively trim until no node remains below ``min_degree``."""
    edge_df = pd.DataFrame(
        {
            "source": ["A", "B", "C"],
            "target": ["B", "C", "D"],
        }
    )

    filtered = trim_low_degree_nodes(edge_df, min_degree=2, recursive=True)

    assert filtered.empty


def test_trim_low_degree_nodes_preserves_all_columns():
    """Preserve unrelated edge metadata columns while filtering rows."""
    edge_df = pd.DataFrame(
        {
            "source": ["A", "B", "C"],
            "target": ["B", "C", "D"],
            "weight": [0.1, 0.2, 0.3],
            "label": ["x", "y", "z"],
        }
    )

    filtered = trim_low_degree_nodes(edge_df, min_degree=2, recursive=False)

    assert list(filtered.columns) == list(edge_df.columns)
    assert filtered.iloc[0]["label"] == "y"


def test_trim_low_degree_nodes_raises_for_missing_columns():
    """Raise an informative error when source/target columns are missing."""
    edge_df = pd.DataFrame({"src": ["A"], "dst": ["B"]})

    with pytest.raises(AttributeError):
        trim_low_degree_nodes(edge_df, source="source", target="target")


def test_trim_low_degree_nodes_handles_empty_dataframe():
    """Return an empty copy safely when the edge list is empty."""
    edge_df = pd.DataFrame(columns=["source", "target", "weight"])

    filtered = trim_low_degree_nodes(edge_df, min_degree=2)

    assert filtered.empty
    assert list(filtered.columns) == ["source", "target", "weight"]
    assert filtered is not edge_df


def test_trim_low_degree_nodes_handles_missing_endpoints():
    """Drop rows with missing endpoints when degree filtering is enabled."""
    edge_df = pd.DataFrame(
        {
            "source": ["A", None, "B"],
            "target": ["B", "C", "A"],
            "kind": ["valid", "missing", "valid"],
        }
    )

    filtered = trim_low_degree_nodes(edge_df, min_degree=1, recursive=False)

    assert filtered["kind"].tolist() == ["valid", "valid"]


def test_trim_low_degree_nodes_counts_duplicate_edges():
    """Count duplicate edges toward undirected degree totals."""
    edge_df = pd.DataFrame(
        {
            "source": ["A", "A", "B"],
            "target": ["B", "B", "C"],
        }
    )

    filtered = trim_low_degree_nodes(edge_df, min_degree=2, recursive=False)

    assert len(filtered) == 2
    assert (filtered["source"] == "A").all()
    assert (filtered["target"] == "B").all()


def test_trim_low_degree_nodes_counts_self_loop_as_two():
    """Retain self-loop edge at ``min_degree=2`` because it contributes degree two."""
    edge_df = pd.DataFrame({"source": ["A"], "target": ["A"], "weight": [1.0]})

    filtered = trim_low_degree_nodes(edge_df, min_degree=2, recursive=False)

    assert len(filtered) == 1
    assert filtered.iloc[0]["source"] == "A"
    assert filtered.iloc[0]["target"] == "A"
