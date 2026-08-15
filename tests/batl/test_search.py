import math

import numpy as np
import pytest
import torch

from batl import search
from batl.constants import DEFAULT_ENSEMBLE_MIN_TREE_MATCHES, DEFAULT_RETRIEVAL_TOP_K
from batl.distance import Metric, compute_distances, metrics
from batl.model import BATLModel
from batl.search import _resolve_rerank_workers, search_batch
from batl.training import alternating_train
from batl.tree import BATLTree
from batl.utils.config import ModelConfig, TrainConfig
from batl.utils.metrics import recall_at_k


class PrefixRoutingDecoder:
    def __init__(self, K: int) -> None:
        self.K = K

    def __call__(self, path_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        logits = torch.empty(path_ids.shape[0], path_ids.shape[1], self.K, device=path_ids.device)
        for row, tokens in enumerate(path_ids):
            if path_ids.shape[1] == 1:
                probs = torch.tensor([0.6, 0.4], device=path_ids.device)
            elif int(tokens[1]) == 0:
                probs = torch.tensor([0.1, 0.9], device=path_ids.device)
            else:
                probs = torch.tensor([0.8, 0.2], device=path_ids.device)
            logits[row, :, :] = torch.log(probs)
        return logits


class PrefixRoutingModel(torch.nn.Module):
    def __init__(self, K: int = 2) -> None:
        super().__init__()
        self.K = K
        self.START_TOKEN = K
        self.decoder = PrefixRoutingDecoder(K)
        self.param = torch.nn.Parameter(torch.zeros(1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 3, device=x.device)


def test_search_batch_routes_queries_through_tensor_batched_beam_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = PrefixRoutingModel(K=2)
    tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=3,
        paths=np.array([[0], [0], [1]], dtype=np.uint16),
    )
    database = np.array([[0.0], [10.0], [1.0]], dtype=np.float32)
    queries = np.array([[0.2], [0.8]], dtype=np.float32)
    calls = []

    def fake_beam_search_batch_tensors(
        *,
        model,
        queries,
        H,
        beam_size,
        num_return_leaves=None,
    ):
        del model, H, beam_size, num_return_leaves
        paths = torch.zeros((queries.shape[0], 1, 1), dtype=torch.long)
        scores = torch.zeros((queries.shape[0], 1), dtype=torch.float32)
        calls.append(queries.shape[0])
        return scores, paths

    monkeypatch.setattr("batl.search._beam_search_batch_tensors", fake_beam_search_batch_tensors)

    retrieved = search_batch([model], [tree], database, queries, beam_size=1, top_k=1)

    assert calls == [2]
    assert retrieved.tolist() == [[0], [0]]


def test_search_batch_reranks_single_tree_candidates_by_l2_distance() -> None:
    model = PrefixRoutingModel(K=2)
    tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=3,
        paths=np.array([[0], [0], [1]], dtype=np.uint16),
    )
    database = np.array([[0.0], [10.0], [1.0]], dtype=np.float32)
    queries = np.array([[0.2]], dtype=np.float32)

    retrieved = search_batch([model], [tree], database, queries, beam_size=1, top_k=2)

    assert retrieved.dtype == np.int64
    assert retrieved.tolist() == [[0, 1]]


def test_search_batch_reranks_single_tree_candidates_by_inner_product() -> None:
    model = PrefixRoutingModel(K=2)
    tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=3,
        paths=np.array([[0], [0], [1]], dtype=np.uint16),
    )
    # Raw dot product with query [1.0]: candidate 1 (10.0) beats candidate 0
    # (0.0), the opposite ranking euclidean would give for this query.
    database = np.array([[0.0], [10.0], [1.0]], dtype=np.float32)
    queries = np.array([[1.0]], dtype=np.float32)

    retrieved = search_batch(
        [model], [tree], database, queries, beam_size=1, top_k=2, metric="inner_product"
    )

    assert retrieved.tolist() == [[1, 0]]


def test_search_batch_torch_gpu_rerank_accepts_inner_product() -> None:
    model = PrefixRoutingModel(K=2)
    tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=3,
        paths=np.array([[0], [0], [1]], dtype=np.uint16),
    )
    database = np.array([[0.0], [10.0], [1.0]], dtype=np.float32)
    queries = np.array([[1.0]], dtype=np.float32)

    retrieved = search_batch(
        [model],
        [tree],
        database,
        queries,
        beam_size=1,
        top_k=2,
        metric="inner_product",
        rerank_backend="torch_gpu",
    )

    assert retrieved.tolist() == [[1, 0]]


def test_search_batch_torch_rerank_rejects_unsupported_metrics() -> None:
    model = PrefixRoutingModel(K=2)
    tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=3,
        paths=np.array([[0], [0], [1]], dtype=np.uint16),
    )
    database = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float32)
    queries = np.array([[1, 0]], dtype=np.float32)

    with pytest.raises(ValueError, match="torch_gpu rerank supports only"):
        search_batch(
            [model],
            [tree],
            database,
            queries,
            beam_size=1,
            top_k=2,
            metric="hamming",
            rerank_backend="torch_gpu",
        )


def test_resolve_rerank_workers_parallelizes_numpy_only() -> None:
    assert 1 <= _resolve_rerank_workers(None, "numpy_cpu", n_queries=8) <= 8
    assert _resolve_rerank_workers(4, "numpy_cpu", n_queries=8) == 4
    assert _resolve_rerank_workers(8, "numpy_cpu", n_queries=2) == 2
    assert _resolve_rerank_workers(None, "torch_gpu", n_queries=8) == 1
    with pytest.raises(ValueError, match="rerank_workers"):
        _resolve_rerank_workers(0, "numpy_cpu", n_queries=8)


def test_search_batch_defaults_to_process_retrieval_top_k() -> None:
    model = PrefixRoutingModel(K=2)
    tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=3,
        paths=np.array([[0], [0], [1]], dtype=np.uint16),
    )
    database = np.array([[0.0], [10.0], [1.0]], dtype=np.float32)
    queries = np.array([[0.2]], dtype=np.float32)

    retrieved = search_batch([model], [tree], database, queries, beam_size=1)

    assert retrieved.shape == (1, DEFAULT_RETRIEVAL_TOP_K)
    assert retrieved[:, :2].tolist() == [[0, 1]]
    assert retrieved[:, 2:].tolist() == [[-1] * (DEFAULT_RETRIEVAL_TOP_K - 2)]


def test_jaccard_rejects_dense_float_vectors() -> None:
    with pytest.raises(ValueError, match="sparse integer id arrays"):
        compute_distances(
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
            np.array([1.0, 0.0], dtype=np.float32),
            "jaccard",
        )


def test_compute_distances_uses_ann_benchmarks_l2_norm_not_squared_l2() -> None:
    database = np.array([[3.0, 4.0], [0.0, 12.0]], dtype=np.float32)
    query = np.array([0.0, 0.0], dtype=np.float32)

    distances = compute_distances(database, np.array([0, 1], dtype=np.int64), query, "euclidean")

    assert distances.tolist() == pytest.approx([5.0, 12.0])


def test_compute_distances_uses_ann_benchmarks_angular_not_arccos() -> None:
    database = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)

    distances = compute_distances(database, np.array([0, 1], dtype=np.int64), query, "angular")

    assert distances.tolist() == pytest.approx([0.0, 1.0 - 1.0 / math.sqrt(2.0)])


def test_compute_distances_dispatches_through_metric_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    query = np.array([0.0], dtype=np.float32)
    candidates = np.array([0, 2], dtype=np.int64)

    def batch_distance(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
        assert vectors.tolist() == [[1.0], [3.0]]
        assert query.tolist() == [0.0]
        return np.array([10.0, 30.0], dtype=np.float32)

    monkeypatch.setitem(metrics, "test_metric", Metric(distances=batch_distance))

    distances = compute_distances(database, candidates, query, "test_metric")

    assert distances.tolist() == [10.0, 30.0]


def test_search_batch_applies_min_trees_frequency_filter() -> None:
    """With 2 trees, only candidates appearing in 2+ buckets are kept automatically."""
    model = PrefixRoutingModel(K=2)
    first_tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=4,
        paths=np.array([[0], [0], [0], [1]], dtype=np.uint16),
    )
    second_tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=4,
        paths=np.array([[0], [1], [0], [0]], dtype=np.uint16),
    )
    database = np.array([[0.0], [100.0], [0.5], [50.0]], dtype=np.float32)
    queries = np.array([[0.1]], dtype=np.float32)

    retrieved = search_batch(
        [model, model],
        [first_tree, second_tree],
        database,
        queries,
        beam_size=1,
        top_k=2,
    )

    assert retrieved.tolist() == [[0, 2]]


def test_search_batch_allows_explicit_frequency_threshold() -> None:
    model = PrefixRoutingModel(K=2)
    first_tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=4,
        paths=np.array([[0], [0], [0], [1]], dtype=np.uint16),
    )
    second_tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=4,
        paths=np.array([[0], [1], [0], [0]], dtype=np.uint16),
    )
    database = np.array([[0.0], [100.0], [0.5], [50.0]], dtype=np.float32)
    queries = np.array([[0.1]], dtype=np.float32)

    retrieved, n_candidates = search_batch(
        [model, model],
        [first_tree, second_tree],
        database,
        queries,
        beam_size=1,
        top_k=4,
        min_trees=1,
        return_candidate_counts=True,
    )

    assert n_candidates.tolist() == [4]
    assert sorted(retrieved[0].tolist()) == [0, 1, 2, 3]


def test_search_batch_rejects_frequency_threshold_above_tree_count() -> None:
    model = PrefixRoutingModel(K=2)
    tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=2,
        paths=np.array([[0], [1]], dtype=np.uint16),
    )

    with pytest.raises(ValueError, match="min_trees"):
        search_batch(
            [model],
            [tree],
            np.array([[0.0], [1.0]], dtype=np.float32),
            np.array([[0.1]], dtype=np.float32),
            beam_size=1,
            min_trees=2,
        )


def test_search_batch_diagnostics_report_floor_fallback_and_empty_candidates() -> None:
    """floor_fallback fires when T>1 and the filtered (count>=2) set is empty;
    falls back to the full union.  empty_candidates fires when even the union is empty."""
    model = PrefixRoutingModel(K=2)
    # Both trees route item 1 to different buckets — no item appears in 2+ buckets.
    # floor fallback kicks in and returns the full union (all 4 items).
    first_tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=4,
        paths=np.array([[0], [0], [0], [1]], dtype=np.uint16),
    )
    second_tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=4,
        paths=np.array([[1], [1], [1], [0]], dtype=np.uint16),
    )
    empty_tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=4,
        paths=np.array([[1], [1], [1], [1]], dtype=np.uint16),
    )
    database = np.array([[0.0], [100.0], [0.5], [50.0]], dtype=np.float32)
    queries = np.array([[0.1]], dtype=np.float32)

    retrieved = search_batch(
        [model, model],
        [first_tree, second_tree],
        database,
        queries,
        beam_size=1,
        top_k=4,
    )

    assert sorted(retrieved[0].tolist()) == [0, 1, 2, 3]

    retrieved = search_batch(
        [model],
        [empty_tree],
        database,
        queries,
        beam_size=1,
        top_k=4,
    )

    assert retrieved.tolist() == [[-1, -1, -1, -1]]


def test_search_batch_diagnostics_report_ground_truth_leaf_coverage() -> None:
    model = PrefixRoutingModel(K=2)
    first_tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=4,
        paths=np.array([[0], [0], [1], [1]], dtype=np.uint16),
    )
    second_tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=4,
        paths=np.array([[0], [1], [0], [1]], dtype=np.uint16),
    )
    database = np.array([[0.0], [10.0], [1.0], [11.0]], dtype=np.float32)
    queries = np.array([[0.1]], dtype=np.float32)

    retrieved = search_batch(
        [model, model],
        [first_tree, second_tree],
        database,
        queries,
        beam_size=1,
        top_k=2,
    )

    assert retrieved.tolist() == [[0, -1]]


def test_search_batch_can_return_candidate_counts() -> None:
    model = PrefixRoutingModel(K=2)
    first_tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=4,
        paths=np.array([[0], [0], [1], [1]], dtype=np.uint16),
    )
    second_tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=4,
        paths=np.array([[0], [0], [1], [1]], dtype=np.uint16),
    )
    database = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    queries = np.array([[0.1], [2.1]], dtype=np.float32)

    retrieved, n_candidates = search_batch(
        [model, model],
        [first_tree, second_tree],
        database,
        queries,
        beam_size=1,
        top_k=2,
        return_candidate_counts=True,
    )

    assert retrieved.tolist() == [[0, 1], [1, 0]]
    assert n_candidates.dtype == np.int64
    assert n_candidates.tolist() == [2, 2]


def test_search_batch_candidate_counts_match_single_query_search() -> None:
    model = PrefixRoutingModel(K=2)
    first_tree = BATLTree(
        K=2,
        H=2,
        alpha=1.0,
        N=6,
        paths=np.array([[0, 1], [0, 1], [1, 0], [1, 0], [1, 1], [0, 0]], dtype=np.uint16),
    )
    second_tree = BATLTree(
        K=2,
        H=2,
        alpha=1.0,
        N=6,
        paths=np.array([[0, 1], [1, 0], [1, 0], [1, 1], [1, 1], [0, 0]], dtype=np.uint16),
    )
    database = np.array([[0.0], [2.0], [1.0], [3.0], [4.0], [100.0]], dtype=np.float32)
    queries = np.array([[0.1], [3.1]], dtype=np.float32)

    batched, batched_counts = search_batch(
        [model, model],
        [first_tree, second_tree],
        database,
        queries,
        beam_size=3,
        top_k=3,
        return_candidate_counts=True,
    )
    singles = [
        search_batch(
            [model, model],
            [first_tree, second_tree],
            database,
            query.reshape(1, -1),
            beam_size=3,
            top_k=3,
            return_candidate_counts=True,
        )
        for query in queries
    ]

    assert batched.tolist() == [result[0][0].tolist() for result in singles]
    assert batched_counts.tolist() == [int(result[1][0]) for result in singles]


def test_multi_tree_candidate_filter_uses_default_min_tree_matches() -> None:
    assert DEFAULT_ENSEMBLE_MIN_TREE_MATCHES == 2

    model = PrefixRoutingModel(K=2)
    first_tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=4,
        paths=np.array([[0], [0], [1], [1]], dtype=np.uint16),
    )
    second_tree = BATLTree(
        K=2,
        H=1,
        alpha=1.0,
        N=4,
        paths=np.array([[0], [1], [1], [1]], dtype=np.uint16),
    )
    database = np.array([[0.0], [10.0], [1.0], [2.0]], dtype=np.float32)
    queries = np.array([[0.2]], dtype=np.float32)

    retrieved, n_candidates = search_batch(
        [model, model],
        [first_tree, second_tree],
        database,
        queries,
        beam_size=1,
        top_k=3,
        return_candidate_counts=True,
    )

    assert n_candidates.tolist() == [1]
    assert retrieved.tolist() == [[0, -1, -1]]


def test_search_batch_rerank_uses_partial_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    model = PrefixRoutingModel(K=2)
    tree = BATLTree(
        K=2,
        H=2,
        alpha=1.0,
        N=4,
        paths=np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.uint16),
    )
    database = np.array([[0.0], [10.0], [20.0], [30.0]], dtype=np.float32)
    queries = np.array([[9.0]], dtype=np.float32)
    calls = 0
    original_argpartition = np.argpartition

    def counting_argpartition(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_argpartition(*args, **kwargs)

    monkeypatch.setattr(np, "argpartition", counting_argpartition)

    retrieved = search_batch([model], [tree], database, queries, beam_size=4, top_k=2)

    assert calls == 1
    assert retrieved.tolist() == [[1, 0]]


def test_search_batch_can_return_fewer_leaves_than_beam_width() -> None:
    model = PrefixRoutingModel(K=2)
    tree = BATLTree(
        K=2,
        H=2,
        alpha=1.0,
        N=4,
        paths=np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.uint16),
    )
    database = np.array([[0.0], [10.0], [20.0], [30.0]], dtype=np.float32)
    queries = np.array([[0.1]], dtype=np.float32)

    retrieved, n_candidates = search_batch(
        [model],
        [tree],
        database,
        queries,
        beam_size=4,
        num_return_leaves=2,
        top_k=4,
        return_candidate_counts=True,
    )

    assert n_candidates.tolist() == [2]
    assert retrieved.tolist() == [[0, 1, -1, -1]]


def test_train_and_search_end_to_end_returns_valid_recall() -> None:
    """End-to-end: alternating_train → search_batch → recall_at_k on tiny CPU data."""
    model_cfg = ModelConfig(
        branching_factor=2,
        tree_height=2,
        embedding_dim=3,
        encoder_hidden=8,
        embed_dim=8,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
        alpha=1.0,
    )
    train_cfg = TrainConfig(
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_epochs=1,
        alternating_interval=1,
        device="cpu",
        top_k_neighbors=1,
        neighbor_search_subset=8,
    )
    rng = np.random.default_rng(0)
    database = rng.normal(size=(8, 3)).astype(np.float32)
    queries = database[:4] + rng.normal(scale=1e-3, size=(4, 3)).astype(np.float32)
    ground_truth = np.arange(4, dtype=np.int64).reshape(-1, 1)

    model = BATLModel(model_cfg)
    trained, tree = alternating_train(model, database, train_cfg, model_cfg, seed=0)

    retrieved = search_batch(
        [trained],
        [tree],
        database,
        queries,
        beam_size=4,
        top_k=1,
    )

    assert retrieved.shape == (4, 1)
    assert retrieved.dtype == np.int64
    # Search must return valid database indices (no padding sentinels for a
    # full-coverage beam over 8 vectors).
    assert (retrieved >= 0).all()
    assert (retrieved < database.shape[0]).all()

    per_query_recall = recall_at_k(retrieved, ground_truth, k=1)
    assert per_query_recall.shape == (4,)
    assert ((per_query_recall >= 0.0) & (per_query_recall <= 1.0)).all()


# --- A3: M > beam_size must fail, not silently cap ---


def test_search_batch_refuses_more_leaves_than_beam_size() -> None:
    """Silent capping made benchmark rows report an M the search never used."""
    model = PrefixRoutingModel(K=2)
    tree = BATLTree(
        K=2, H=2, alpha=1.0, N=4, paths=np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.uint16)
    )
    database = np.array([[0.0], [10.0], [20.0], [30.0]], dtype=np.float32)
    queries = np.array([[0.1]], dtype=np.float32)

    with pytest.raises(ValueError, match=r"must be <= beam_size \(b\); got M=4, b=2"):
        search_batch([model], [tree], database, queries, beam_size=2, num_return_leaves=4)

    with pytest.raises(ValueError, match="num_return_leaves \\(M\\) must be positive"):
        search_batch([model], [tree], database, queries, beam_size=2, num_return_leaves=0)


def test_search_batch_allows_leaves_equal_to_beam_size() -> None:
    model = PrefixRoutingModel(K=2)
    tree = BATLTree(
        K=2, H=2, alpha=1.0, N=4, paths=np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.uint16)
    )
    database = np.array([[0.0], [10.0], [20.0], [30.0]], dtype=np.float32)
    queries = np.array([[0.1]], dtype=np.float32)

    retrieved = search_batch(
        [model], [tree], database, queries, beam_size=4, num_return_leaves=4, top_k=2
    )

    assert retrieved.shape == (1, 2)


def test_config_rejects_num_leaves_above_beam_size() -> None:
    from batl.utils.config import ExperimentConfig

    def build(num_leaves: list[int], beam_size: int = 100) -> ExperimentConfig:
        return ExperimentConfig(
            name="t",
            seed=0,
            output_dir="out",
            dataset_name="d",
            dataset_path="p",
            split="train",
            subset_size=None,
            recall_at=[10],
            num_queries=10,
            beam_size=beam_size,
            num_leaves=num_leaves,
        )

    # Valid sweep loads.
    assert build([10, 100]).num_leaves == [10, 100]

    with pytest.raises(ValueError, match=r"must be <= evaluation.beam_size \(100\); got \[200\]"):
        build([10, 200])
    with pytest.raises(ValueError, match="num_leaves values must be positive"):
        build([0, 10])


def _skewed_candidate_reranker(recorded_widths):
    """A stand-in reranker that records the padded width it is handed."""

    class _Recorder:
        def rerank_batch(self, queries, candidates, top_k, metric):
            recorded_widths.append(candidates.shape[1])
            out = np.full((candidates.shape[0], top_k), -1, dtype=np.int64)
            for row in range(candidates.shape[0]):
                real = candidates[row][candidates[row] >= 0]
                out[row, : min(top_k, real.size)] = real[:top_k]
            return out

    return _Recorder()


def test_resident_rerank_groups_queries_by_candidate_count() -> None:
    """The resident path pays for padding; numpy_cpu pays per real candidate.

    At T=4 the frequency filter makes counts vary 17-35%, and padding every
    query to the global max made the GPU 2.2-3.7x SLOWER than the CPU on job
    22848765 while being 1.8-5.4x faster on a T=1 index of the same data.
    """
    widths: list[int] = []
    reranker = _skewed_candidate_reranker(widths)
    n_queries = search.RERANK_GROUP_QUERIES * 2
    # One pathological query with 50x the candidates of every other.
    parts = [[np.arange(10, dtype=np.int64)] for _ in range(n_queries)]
    parts[0] = [np.arange(500, dtype=np.int64)]

    _results, n_candidates = search._rerank_resident(
        candidate_parts_by_query=parts,
        n_queries=n_queries,
        queries=np.zeros((n_queries, 4), dtype=np.float32),
        reranker=reranker,
        top_k=5,
        trees_count=1,
        min_trees=1,
        metric="euclidean",
        profiler=None,
    )

    assert int(n_candidates.max()) == 500
    # The wide query lands in its own group; the rest stay narrow.
    assert min(widths) == 10, widths
    assert max(widths) == 500, widths


def test_resident_rerank_returns_rows_in_query_order() -> None:
    """Grouping reorders internally; results must land back where they belong."""
    widths: list[int] = []
    reranker = _skewed_candidate_reranker(widths)
    parts = [[np.array([7 * i, 7 * i + 1], dtype=np.int64)] for i in range(5)]
    parts[2] = [np.arange(100, 140, dtype=np.int64)]

    results, n_candidates = search._rerank_resident(
        candidate_parts_by_query=parts,
        n_queries=5,
        queries=np.zeros((5, 4), dtype=np.float32),
        reranker=reranker,
        top_k=2,
        trees_count=1,
        min_trees=1,
        metric="euclidean",
        profiler=None,
    )

    assert list(n_candidates) == [2, 2, 40, 2, 2]
    for i in (0, 1, 3, 4):
        assert list(results[i]) == [7 * i, 7 * i + 1], (i, results[i])
    assert list(results[2]) == [100, 101]
