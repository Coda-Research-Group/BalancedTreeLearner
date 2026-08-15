import time
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from batl.profiling import (
    StageProfiler,
    device_metadata,
    memory_metadata,
    peak_host_rss_bytes,
    repetition_summary,
    stage_reconciliation,
)


def test_disabled_profiler_records_nothing_and_returns_a_shared_noop() -> None:
    profiler = StageProfiler(enabled=False)

    with profiler.stage("anything"):
        pass

    assert profiler.to_dict() == {}
    assert profiler.total_s == 0.0
    # The no-op must not allocate a fresh context manager per call: `stage` sits
    # inside the per-query search loop.
    assert profiler.stage("a") is profiler.stage("b")
    assert isinstance(profiler.stage("a"), nullcontext)


def test_disabled_profiler_never_synchronizes_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Acceptance 1: profiling off must not add synchronization to the fast path."""
    calls: list[object] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: calls.append(device))

    profiler = StageProfiler(enabled=False, device="cuda")
    with profiler.stage("search.beam_decode"):
        pass

    assert calls == []


def test_enabled_profiler_synchronizes_around_cuda_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: calls.append(device))

    profiler = StageProfiler(enabled=True, device="cuda")
    with profiler.stage("search.rerank"):
        pass

    assert len(calls) == 2  # once before, once after


def test_enabled_profiler_accumulates_seconds_and_call_counts() -> None:
    profiler = StageProfiler(enabled=True)

    for _ in range(3):
        with profiler.stage("build.train_epoch"):
            pass
    with profiler.stage("build.label_mining"):
        pass

    stages = profiler.to_dict()
    assert set(stages) == {"build.label_mining", "build.train_epoch"}
    assert stages["build.train_epoch"]["calls"] == 3
    assert stages["build.label_mining"]["calls"] == 1
    assert profiler.total_s >= 0.0


def test_profiler_reset_clears_recorded_stages() -> None:
    profiler = StageProfiler(enabled=True)
    with profiler.stage("s"):
        pass
    with profiler.stage("t"):
        pass
    assert profiler.to_dict()["s"]["calls"] == 1

    profiler.reset()
    assert profiler.to_dict() == {}


def test_stage_reconciliation_flags_unattributed_time() -> None:
    profiler = StageProfiler(enabled=True)
    # Sleep so the stage is well clear of the microsecond rounding in to_dict;
    # sub-microsecond stages are not what this reconciliation is for.
    with profiler.stage("s"):
        time.sleep(0.005)

    measured = profiler.total_s
    assert measured >= 0.005

    within = stage_reconciliation(profiler, wall_clock_s=measured * 1.05)
    assert within["within_tolerance"] is True

    outside = stage_reconciliation(profiler, wall_clock_s=measured * 10)
    assert outside["within_tolerance"] is False
    assert outside["unattributed_s"] > 0
    assert outside["stage_total_s"] == pytest.approx(measured, abs=1e-6)


def test_stage_reconciliation_handles_zero_wall_clock() -> None:
    result = stage_reconciliation(StageProfiler(enabled=True), wall_clock_s=0.0)
    assert result["unattributed_fraction"] == 0.0


def test_device_metadata_reports_threading_and_precision() -> None:
    metadata = device_metadata("cpu")

    assert metadata["device"] == "cpu"
    assert metadata["torch"] == torch.__version__
    assert "tf32_matmul" in metadata
    assert isinstance(metadata["cpu_count"], int)
    assert isinstance(metadata["torch_num_threads"], int)


def test_memory_metadata_reports_host_peak() -> None:
    peaks = memory_metadata("cpu")

    assert peaks["peak_host_rss_bytes"] == pytest.approx(peak_host_rss_bytes(), rel=0.5)
    assert "peak_cuda_allocated_bytes" not in peaks


def test_repetition_summary_reports_median_and_every_value() -> None:
    assert repetition_summary([]) == {"repetitions": 0, "median_s": 0.0, "times_s": []}

    odd = repetition_summary([3.0, 1.0, 2.0])
    assert odd["median_s"] == 2.0
    assert odd["min_s"] == 1.0
    assert odd["max_s"] == 3.0
    # Individual values survive so variance stays visible, in original order.
    assert odd["times_s"] == [3.0, 1.0, 2.0]

    even = repetition_summary([1.0, 2.0, 3.0, 4.0])
    assert even["median_s"] == 2.5


def test_search_batch_records_stages_only_when_profiling(monkeypatch: pytest.MonkeyPatch) -> None:
    from batl.search import search_batch
    from batl.tree import BATLTree
    from tests.batl.test_search import PrefixRoutingModel

    model = PrefixRoutingModel(K=2)
    tree = BATLTree(K=2, H=1, alpha=1.0, N=3, paths=np.array([[0], [0], [1]], dtype=np.uint16))
    database = np.array([[0.0], [10.0], [1.0]], dtype=np.float32)
    queries = np.array([[0.2]], dtype=np.float32)

    off = StageProfiler(enabled=False)
    baseline = search_batch([model], [tree], database, queries, beam_size=1, top_k=2, profiler=off)
    assert off.to_dict() == {}

    on = StageProfiler(enabled=True)
    profiled = search_batch([model], [tree], database, queries, beam_size=1, top_k=2, profiler=on)

    stages = on.to_dict()
    assert "search.beam_decode" in stages
    assert "search.leaf_lookup" in stages
    assert "search.select_and_rerank_cpu" in stages
    # Acceptance 1: results are identical whether or not profiling is on.
    assert profiled.tolist() == baseline.tolist()
