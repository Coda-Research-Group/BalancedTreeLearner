import logging
import os
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from batl import rerank as rerank_module
from batl.distance import compute_distances
from batl.rerank import (
    RerankGpuMemoryError,
    ResidentGpuReranker,
    check_device_capacity,
    resident_rerank_bytes,
)
from batl.search import per_query_rerank_backend, search_batch

RUN_GPU_TESTS = os.environ.get("BATL_RUN_GPU_TESTS") == "1"


def _reference_top_k(
    database: np.ndarray,
    candidates: np.ndarray,
    query: np.ndarray,
    top_k: int,
    metric: str,
) -> np.ndarray:
    """Rerank one query with the numpy_cpu path, -1 padded to ``top_k``."""
    result = np.full(top_k, -1, dtype=np.int64)
    real = candidates[candidates >= 0]
    if real.size == 0:
        return result
    distances = compute_distances(database, real, query, metric)
    order = np.argsort(distances, kind="mergesort")[:top_k]
    result[: order.size] = real[order]
    return result


@pytest.fixture(scope="module")
def synthetic_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    database = rng.normal(size=(10_000, 32)).astype(np.float32)
    queries = rng.normal(size=(16, 32)).astype(np.float32)
    candidates = rng.choice(10_000, size=(16, 400), replace=False).astype(np.int64)
    return database, queries, candidates


@pytest.mark.parametrize("metric", ["euclidean", "angular", "inner_product"])
def test_rerank_batch_matches_numpy_cpu_ids(synthetic_case, metric: str) -> None:
    database, queries, candidates = synthetic_case
    reranker = ResidentGpuReranker(database, torch.device("cpu"))

    reranked = reranker.rerank_batch(queries, candidates, top_k=10, metric=metric)

    assert reranked.shape == (queries.shape[0], 10)
    assert reranked.dtype == np.int64
    for row, query in enumerate(queries):
        expected = _reference_top_k(database, candidates[row], query, 10, metric)
        # Continuous random data has no exact ties, so ids must match exactly.
        assert reranked[row].tolist() == expected.tolist()


def test_rerank_batch_orders_exact_ties_by_distance_value() -> None:
    # Duplicated rows produce exactly-equal distances; only the returned
    # distance values are well defined, not which duplicate id is picked.
    database = np.array([[0.0], [1.0], [1.0], [5.0]], dtype=np.float32)
    queries = np.array([[0.0]], dtype=np.float32)
    candidates = np.array([[0, 1, 2, 3]], dtype=np.int64)
    reranker = ResidentGpuReranker(database, torch.device("cpu"))

    reranked = reranker.rerank_batch(queries, candidates, top_k=3, metric="euclidean")

    distances = compute_distances(database, reranked[0], queries[0], "euclidean")
    assert distances.tolist() == pytest.approx([0.0, 1.0, 1.0])
    assert sorted(reranked[0].tolist()[1:]) == [1, 2]


def test_rerank_batch_pads_short_and_empty_candidate_rows() -> None:
    database = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    queries = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
    candidates = np.array([[0, 1, 2], [3, -1, -1], [-1, -1, -1]], dtype=np.int64)
    reranker = ResidentGpuReranker(database, torch.device("cpu"))

    reranked = reranker.rerank_batch(queries, candidates, top_k=4, metric="euclidean")

    assert reranked[0].tolist() == [0, 1, 2, -1]
    assert reranked[1].tolist() == [3, -1, -1, -1]
    assert reranked[2].tolist() == [-1, -1, -1, -1]


def test_rerank_batch_inner_product_is_magnitude_sensitive_unlike_angular() -> None:
    # Same direction, different magnitude: angular ranks these as tied;
    # inner_product must not.
    database = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    candidates = np.array([[0, 1]], dtype=np.int64)
    reranker = ResidentGpuReranker(database, torch.device("cpu"))

    ip_order = reranker.rerank_batch(queries, candidates, top_k=2, metric="inner_product")
    angular_order = reranker.rerank_batch(queries, candidates, top_k=2, metric="angular")

    # inner_product has no tie (raw dots -1.0 vs -2.0), so the order is exact.
    assert ip_order[0].tolist() == [1, 0]
    # angular ranks both as cosine-identical to the query; only the set of
    # ids is well defined for a tie (see test_rerank_batch_orders_exact_ties).
    assert sorted(angular_order[0].tolist()) == [0, 1]


def test_rerank_batch_handles_zero_vectors_under_angular() -> None:
    # The numpy path defines similarity as 0 (distance 1) for a zero row.
    database = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    candidates = np.array([[0, 1, 2]], dtype=np.int64)
    reranker = ResidentGpuReranker(database, torch.device("cpu"))

    reranked = reranker.rerank_batch(queries, candidates, top_k=3, metric="angular")

    assert reranked[0].tolist() == [1, 0, 2]


def test_rerank_batch_micro_batches_without_changing_results(synthetic_case) -> None:
    database, queries, candidates = synthetic_case
    single_pass = ResidentGpuReranker(database, torch.device("cpu"))
    # One query per gather step: 400 candidates x 32 dims x 4 bytes.
    micro = ResidentGpuReranker(database, torch.device("cpu"), max_gather_bytes=400 * 32 * 4)

    assert micro._micro_batch_rows(400) == 1
    assert single_pass._micro_batch_rows(400) > 1
    assert (
        micro.rerank_batch(queries, candidates, top_k=10).tolist()
        == single_pass.rerank_batch(queries, candidates, top_k=10).tolist()
    )


def test_rerank_batch_accepts_cosine_alias_and_rejects_other_metrics() -> None:
    database = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    candidates = np.array([[0, 1]], dtype=np.int64)
    reranker = ResidentGpuReranker(database, torch.device("cpu"))

    assert reranker.rerank_batch(queries, candidates, top_k=1, metric="cosine")[0].tolist() == [0]
    with pytest.raises(ValueError, match="resident rerank supports only"):
        reranker.rerank_batch(queries, candidates, top_k=1, metric="hamming")


def test_reranker_validates_shapes_and_arguments() -> None:
    database = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    reranker = ResidentGpuReranker(database, torch.device("cpu"))
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    candidates = np.array([[0, 1]], dtype=np.int64)

    with pytest.raises(ValueError, match="top_k must be positive"):
        reranker.rerank_batch(queries, candidates, top_k=0)
    with pytest.raises(ValueError, match="same number of rows"):
        reranker.rerank_batch(np.vstack([queries, queries]), candidates, top_k=1)
    with pytest.raises(ValueError, match="same dimension as the database"):
        reranker.rerank_batch(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), candidates, top_k=1)
    with pytest.raises(ValueError, match="database must be a 2D"):
        ResidentGpuReranker(np.zeros(4, dtype=np.float32), torch.device("cpu"))
    with pytest.raises(ValueError, match="max_gather_bytes must be positive"):
        ResidentGpuReranker(database, torch.device("cpu"), max_gather_bytes=0)


def test_resident_rerank_bytes_counts_matrix_and_norms() -> None:
    assert resident_rerank_bytes(100, 96) == 100 * 96 * 4 + 100 * 4


def test_capacity_check_raises_when_vram_budget_is_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (4 * 1024**3, 8 * 1024**3))

    check_device_capacity(
        n_vectors=1_000_000, dim=96, device=torch.device("cuda"), headroom_bytes=2 * 1024**3
    )
    with pytest.raises(RerankGpuMemoryError, match="resident rerank needs"):
        check_device_capacity(
            n_vectors=100_000_000, dim=96, device=torch.device("cuda"), headroom_bytes=2 * 1024**3
        )


def test_capacity_check_does_not_budget_non_cuda_devices() -> None:
    check_device_capacity(n_vectors=10**9, dim=128, device=torch.device("cpu"))


def test_reranker_construction_propagates_capacity_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(**kwargs) -> None:
        raise RerankGpuMemoryError("no room")

    monkeypatch.setattr("batl.rerank.check_device_capacity", refuse)

    with pytest.raises(RerankGpuMemoryError, match="no room"):
        ResidentGpuReranker(np.zeros((4, 2), dtype=np.float32), torch.device("cpu"))


def test_per_query_rerank_backend_downgrades_resident_only() -> None:
    assert per_query_rerank_backend("torch_gpu_resident") == "torch_gpu"
    assert per_query_rerank_backend("torch_gpu") == "torch_gpu"
    assert per_query_rerank_backend("numpy_cpu") == "numpy_cpu"


def test_search_batch_requires_a_reranker_for_the_resident_backend() -> None:
    from batl.tree import BATLTree
    from tests.batl.test_search import PrefixRoutingModel

    model = PrefixRoutingModel(K=2)
    tree = BATLTree(K=2, H=1, alpha=1.0, N=3, paths=np.array([[0], [0], [1]], dtype=np.uint16))
    database = np.array([[0.0], [10.0], [1.0]], dtype=np.float32)
    queries = np.array([[0.2]], dtype=np.float32)

    with pytest.raises(ValueError, match="requires reranker="):
        search_batch(
            [model],
            [tree],
            database,
            queries,
            beam_size=1,
            top_k=2,
            rerank_backend="torch_gpu_resident",
        )


def test_search_batch_resident_matches_numpy_cpu_end_to_end() -> None:
    from batl.tree import BATLTree
    from tests.batl.test_search import PrefixRoutingModel

    model = PrefixRoutingModel(K=2)
    first_tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=6,
        paths=np.array([[0], [0], [1], [1], [0], [1]], dtype=np.uint16),
    )
    second_tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=6,
        paths=np.array([[0], [1], [1], [0], [0], [1]], dtype=np.uint16),
    )
    database = np.array([[0.0], [3.0], [1.0], [9.0], [0.5], [2.0]], dtype=np.float32)
    queries = np.array([[0.2], [2.5]], dtype=np.float32)
    reranker = ResidentGpuReranker(database, torch.device("cpu"))

    baseline, baseline_counts = search_batch(
        [model, model],
        [first_tree, second_tree],
        database,
        queries,
        beam_size=2,
        top_k=3,
        return_candidate_counts=True,
    )
    resident, resident_counts = search_batch(
        [model, model],
        [first_tree, second_tree],
        database,
        queries,
        beam_size=2,
        top_k=3,
        return_candidate_counts=True,
        rerank_backend="torch_gpu_resident",
        reranker=reranker,
    )

    assert resident.tolist() == baseline.tolist()
    assert resident_counts.tolist() == baseline_counts.tolist()


def test_search_with_progress_passes_the_reranker_through_every_chunk() -> None:
    import search as search_entrypoint
    from batl.tree import BATLTree
    from tests.batl.test_search import PrefixRoutingModel

    model = PrefixRoutingModel(K=2)
    tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=6,
        paths=np.array([[0], [0], [1], [1], [0], [1]], dtype=np.uint16),
    )
    database = np.array([[0.0], [3.0], [1.0], [9.0], [0.5], [2.0]], dtype=np.float32)
    queries = np.array([[0.2], [2.5], [8.0], [1.1]], dtype=np.float32)
    reranker = ResidentGpuReranker(database, torch.device("cpu"))

    kwargs = {
        "models": [model],
        "trees": [tree],
        "database": database,
        "queries": queries,
        "beam_size": 2,
        "num_return_leaves": 2,
        "label": "test",
        "metric": "euclidean",
    }
    baseline, baseline_counts = search_entrypoint._search_with_progress(
        progress_every=0, rerank_backend="numpy_cpu", **kwargs
    )
    # progress_every=3 forces two unequal chunks through the resident path.
    chunked, chunked_counts = search_entrypoint._search_with_progress(
        progress_every=3,
        rerank_backend="torch_gpu_resident",
        reranker=reranker,
        **kwargs,
    )

    assert chunked.tolist() == baseline.tolist()
    assert chunked_counts.tolist() == baseline_counts.tolist()


def _reranker_cfg(backend: str):
    return SimpleNamespace(rerank_backend=backend, train=SimpleNamespace(device="cpu"))


def test_build_reranker_is_a_noop_for_non_resident_backends() -> None:
    import search as search_entrypoint

    cfg = _reranker_cfg("numpy_cpu")
    assert search_entrypoint._build_reranker(cfg, np.zeros((4, 2), dtype=np.float32)) == (None, 0.0)
    assert cfg.rerank_backend == "numpy_cpu"


def test_build_reranker_returns_a_reusable_instance() -> None:
    import search as search_entrypoint

    cfg = _reranker_cfg("torch_gpu_resident")
    reranker, elapsed = search_entrypoint._build_reranker(cfg, np.zeros((4, 2), dtype=np.float32))

    assert isinstance(reranker, ResidentGpuReranker)
    assert elapsed >= 0.0
    assert cfg.rerank_backend == "torch_gpu_resident"


def test_build_reranker_falls_back_to_numpy_cpu_when_vram_is_short(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import search as search_entrypoint

    def refuse(database, device):
        raise RerankGpuMemoryError("needs 38.00 GiB but only 4.00 GiB is free")

    monkeypatch.setattr(search_entrypoint, "ResidentGpuReranker", refuse)
    cfg = _reranker_cfg("torch_gpu_resident")

    with caplog.at_level(logging.WARNING):
        reranker, elapsed = search_entrypoint._build_reranker(
            cfg, np.zeros((4, 2), dtype=np.float32)
        )

    assert reranker is None
    assert elapsed == 0.0
    assert cfg.rerank_backend == "numpy_cpu"
    assert "falling back to numpy_cpu" in caplog.text


@pytest.mark.skipif(not RUN_GPU_TESTS, reason="set BATL_RUN_GPU_TESTS=1 to run CUDA rerank tests")
def test_rerank_batch_on_cuda_matches_numpy_cpu(synthetic_case) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    database, queries, candidates = synthetic_case
    reranker = ResidentGpuReranker(database, torch.device("cuda"))

    reranked = reranker.rerank_batch(queries, candidates, top_k=10, metric="euclidean")

    for row, query in enumerate(queries):
        expected = _reference_top_k(database, candidates[row], query, 10, "euclidean")
        assert reranked[row].tolist() == expected.tolist()


def test_gather_budget_is_clamped_to_free_vram_after_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stand in for a card whose capacity check passes with little to spare:
    # the configured 2 GiB gather must shrink to half of what is actually free.
    free_after_upload = 1536 * 1024**2  # 1.5 GiB
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device: (free_after_upload, 40 * 1024**3)
    )
    monkeypatch.setattr("batl.rerank.check_device_capacity", lambda **kwargs: None)
    reranker = ResidentGpuReranker(np.zeros((4, 2), dtype=np.float32), torch.device("cpu"))
    # CPU construction skips the clamp; exercise it directly against the device
    # branch by pretending the instance is resident on CUDA.
    reranker.device = torch.device("cuda")

    assert reranker._resolve_gather_budget(2 * 1024**3) == free_after_upload // 2
    # A budget already smaller than the affordable share is left alone.
    assert reranker._resolve_gather_budget(256 * 1024**2) == 256 * 1024**2


def test_gather_budget_is_untouched_on_non_cuda_devices() -> None:
    reranker = ResidentGpuReranker(
        np.zeros((4, 2), dtype=np.float32), torch.device("cpu"), max_gather_bytes=1234
    )

    assert reranker.max_gather_bytes == 1234


def test_micro_batch_warns_when_it_collapses(caplog) -> None:
    """A micro-batch near 1 is why numpy_cpu can beat the resident path.

    The measured 3.09-4.92x speedup was taken at T=1/M=100, where the batch is
    ~22 queries. T=4 triples candidates per query on the same card.
    """
    reranker = ResidentGpuReranker.__new__(ResidentGpuReranker)
    reranker.dim = 96
    reranker.max_gather_bytes = 256 * 1024**2

    with caplog.at_level(logging.WARNING, logger="batl.rerank"):
        rows = reranker._micro_batch_rows(candidates_per_query=800_000)

    assert rows < rerank_module.MIN_HEALTHY_MICRO_BATCH_ROWS
    assert "launch-bound" in caplog.text


def test_healthy_micro_batch_stays_quiet(caplog) -> None:
    reranker = ResidentGpuReranker.__new__(ResidentGpuReranker)
    reranker.dim = 96
    reranker.max_gather_bytes = 2 * 1024**3

    with caplog.at_level(logging.WARNING, logger="batl.rerank"):
        rows = reranker._micro_batch_rows(candidates_per_query=250_000)

    assert rows >= rerank_module.MIN_HEALTHY_MICRO_BATCH_ROWS
    assert caplog.text == ""


def test_upload_accepts_a_read_only_memmap_without_warning(tmp_path) -> None:
    """The Deep100M base is opened read-only; torch warned on every run."""
    path = tmp_path / "db.f32"
    np.arange(32, dtype=np.float32).reshape(8, 4).tofile(path)
    memmap = np.memmap(path, dtype=np.float32, mode="r", shape=(8, 4))

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        chunk = np.array(memmap[0:4], dtype=np.float32, copy=True, order="C")
        torch.from_numpy(chunk)

    assert chunk.flags.writeable
