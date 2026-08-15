"""Tests for batl.logging_utils."""

import numpy as np
import pytest

from batl.utils.logging_utils import (
    per_query_recall_stats,
    print_query_progress,
    standard_run_metadata,
)


def test_standard_run_metadata() -> None:
    metadata = standard_run_metadata("cpu")
    assert "environment" in metadata
    assert "hardware" in metadata
    assert metadata["hardware"]["device"] == "cpu"


def test_print_query_progress(capsys: pytest.CaptureFixture) -> None:
    print_query_progress(label="test", done=10, total=100, elapsed_s=2.0)
    captured = capsys.readouterr()
    assert "test: queries done 10/100, remaining 90" in captured.out
    assert "5.00 q/s" in captured.out


def test_per_query_recall_stats_basic() -> None:
    data = np.array([0.0, 0.5, 1.0, 0.25, 0.75], dtype=np.float64)
    stats = per_query_recall_stats(data)
    assert "p5" in stats and "p95" in stats
    assert stats["min"] == pytest.approx(0.0)
    assert stats["max"] == pytest.approx(1.0)
    assert stats["zero_count"] == 1
    assert stats["below_half_count"] == 2


def test_per_query_recall_stats_2d_raises() -> None:
    data = np.array([[0.5, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="1D"):
        per_query_recall_stats(data)


def test_per_query_recall_stats_empty_raises() -> None:
    data = np.array([], dtype=np.float64)
    with pytest.raises(ValueError, match="non-empty"):
        per_query_recall_stats(data)
