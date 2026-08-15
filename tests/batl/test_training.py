import math
import os
import resource
import sys

import numpy as np
import pytest
import torch

import batl.neighbor_search as neighbor_search_module
from batl.constants import DEFAULT_TRAINING_QUERY_FRACTION
from batl.model import BATLModel
from batl.neighbor_search import find_approximate_neighbors
from batl.training import (
    _num_training_queries,
    _relative_loss_improvement,
    _train_epoch,
    _validate_neighbor_ids,
    _validate_target_paths,
    alternating_train,
)
from batl.tree import BATLTree
from batl.tree_update import (
    TreeUpdateDiagnostics,
    _assign_balanced_vectorized,
    _assignment_margins,
    _assignment_order,
    _branch_order_dtype,
    _cuda_attention_batch_guard,
    _decode_group_topr,
    _first_available_branch,
    _resolve_tree_update_batch_size,
    _resolve_tree_update_cache_embeddings,
    _sort_branches_by_probability,
    update_tree,
)
from batl.utils.config import ModelConfig, TrainConfig


class FixedRoutingModel(torch.nn.Module):
    def __init__(self, K: int, H: int) -> None:
        super().__init__()
        self.K = K
        self.H = H
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def decode_node_probs(
        self, vectors: torch.Tensor, path_prefix: tuple[int, ...]
    ) -> torch.Tensor:
        self.calls.append((path_prefix, vectors.shape[0]))
        probs = torch.full((vectors.shape[0], self.K), 0.01, dtype=torch.float32)
        probs[:, 0] = 0.99
        return probs / probs.sum(dim=1, keepdim=True)


class ConfidenceRoutingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.K = 2
        self.H = 1

    def decode_node_probs(
        self, vectors: torch.Tensor, path_prefix: tuple[int, ...]
    ) -> torch.Tensor:
        del path_prefix
        p0 = vectors[:, 0].to(dtype=torch.float32)
        return torch.stack([p0, 1.0 - p0], dim=1)


def test_find_approximate_neighbors_returns_original_database_indices() -> None:
    database = np.array([[0.0], [1.0], [10.0], [11.0]], dtype=np.float32)
    queries = np.array([[0.2], [10.2]], dtype=np.float32)

    neighbors = find_approximate_neighbors(
        queries=queries,
        database=database,
        top_k=2,
        subset_size=4,
        seed=123,
    )

    assert neighbors.dtype == np.int64
    assert neighbors.tolist() == [[0, 1], [2, 3]]


def test_find_approximate_neighbors_samples_seeded_subset() -> None:
    database = np.arange(8, dtype=np.float32).reshape(-1, 1)
    queries = np.array([[0.1], [7.1]], dtype=np.float32)
    subset_size = 4
    seed = 7

    neighbors = find_approximate_neighbors(
        queries, database, top_k=1, subset_size=subset_size, seed=seed
    )

    subset = np.random.default_rng(seed).choice(len(database), size=subset_size, replace=False)
    expected = []
    for query in queries:
        distances = np.sum((database[subset] - query) ** 2, axis=1)
        expected.append([int(subset[np.argsort(distances)[0]])])
    assert neighbors.tolist() == expected


def test_find_approximate_neighbors_can_build_subset_sequentially() -> None:
    database = np.array([[0.0], [1.0], [10.0], [11.0], [20.0], [21.0]], dtype=np.float32)
    queries = np.array([[0.2], [20.2]], dtype=np.float32)

    neighbors = find_approximate_neighbors(
        queries=queries,
        database=database,
        top_k=2,
        subset_size=6,
        seed=123,
        mode="sequential_chunked",
        chunk_size=2,
    )

    assert neighbors.tolist() == [[0, 1], [4, 5]]


def test_faiss_gpu_neighbor_search_batches_large_query_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeIndex:
        ntotal = 2048

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
            self.batch_sizes.append(queries.shape[0])
            distances = np.zeros((queries.shape[0], top_k), dtype=np.float32)
            labels = np.zeros((queries.shape[0], top_k), dtype=np.int64)
            return distances, labels

    monkeypatch.setattr(
        neighbor_search_module,
        "_DEFAULT_GPU_SEARCH_BATCH_SIZE",
        2,
    )
    monkeypatch.setattr(
        neighbor_search_module,
        "_resolve_faiss_backend",
        lambda backend: "faiss_gpu",
    )
    index = FakeIndex()
    queries = np.zeros((5, 1), dtype=np.float32)

    distances, labels = neighbor_search_module._faiss_search(
        index, queries, top_k=3, backend="faiss_gpu"
    )

    assert index.batch_sizes == [2, 2, 1]
    assert distances.shape == (5, 3)
    assert labels.shape == (5, 3)


def test_update_tree_preserves_count_and_respects_level_capacity() -> None:
    vectors = np.arange(10, dtype=np.float32).reshape(5, 2)
    current_tree = BATLTree.random_init(N=5, K=2, H=2, alpha=1.0, seed=0)
    model = FixedRoutingModel(K=2, H=2)

    updated = update_tree(
        model=model,
        vectors=vectors,
        current_tree=current_tree,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert updated.paths.shape == (5, 2)
    assert updated.paths.dtype == np.uint16
    assert sum(len(updated.get_leaf_indices(tuple(path))) for path in updated._leaf_buckets) == 5

    root_capacity = math.ceil(current_tree.alpha * current_tree.N / current_tree.K)
    level1_capacity = math.ceil(current_tree.alpha * current_tree.N / (current_tree.K**2))
    assert np.bincount(updated.paths[:, 0], minlength=2).max() <= root_capacity
    for branch in range(current_tree.K):
        child_rows = updated.paths[updated.paths[:, 0] == branch]
        if child_rows.size:
            assert np.bincount(child_rows[:, 1], minlength=2).max() <= level1_capacity

    assert ((), 2) in model.calls


def test_update_tree_can_return_assignment_diagnostics_when_top_branch_fills() -> None:
    vectors = np.array([[0.51], [0.52], [0.99], [0.98], [0.97]], dtype=np.float32)
    current_tree = BATLTree.random_init(N=5, K=2, H=1, alpha=1.0, seed=0)
    model = ConfidenceRoutingModel()

    updated, diagnostics = update_tree(
        model=model,
        vectors=vectors,
        current_tree=current_tree,
        batch_size=5,
        device=torch.device("cpu"),
        return_diagnostics=True,
    )

    assert isinstance(diagnostics, TreeUpdateDiagnostics)
    assert updated.paths.shape == (5, 1)
    assert diagnostics.assignment_order == "input"
    assert diagnostics.levels[0]["num_vectors"] == 5
    assert diagnostics.levels[0]["second_choice_fraction"] == pytest.approx(2 / 5)
    assert diagnostics.levels[0]["denied_top_fraction"] == pytest.approx(2 / 5)
    assert diagnostics.levels[0]["fallback_fraction"] == 0.0
    assert diagnostics.levels[0]["mean_chosen_rank"] == pytest.approx(0.4)
    # 3 vectors took rank 0, 2 took rank 1; K=2 so no higher rank exists.
    assert diagnostics.levels[0]["rank_hist_rank_0"] == 3
    assert diagnostics.levels[0]["rank_hist_rank_1"] == 2
    assert diagnostics.levels[0]["max_chosen_rank"] == 1
    assert diagnostics.levels[0]["min_top_r_covering_999"] == 2


def test_update_tree_confidence_order_preserves_capacity_and_prioritizes_confident_vectors() -> (
    None
):
    vectors = np.array([[0.51], [0.52], [0.99], [0.98], [0.97]], dtype=np.float32)
    current_tree = BATLTree.random_init(N=5, K=2, H=1, alpha=1.0, seed=0)
    model = ConfidenceRoutingModel()

    input_tree = update_tree(
        model=model,
        vectors=vectors,
        current_tree=current_tree,
        batch_size=5,
        device=torch.device("cpu"),
    )
    confidence_tree, diagnostics = update_tree(
        model=model,
        vectors=vectors,
        current_tree=current_tree,
        batch_size=5,
        device=torch.device("cpu"),
        assignment_order="confidence",
        return_diagnostics=True,
    )

    assert np.bincount(confidence_tree.paths[:, 0], minlength=2).max() <= 3
    assert confidence_tree.paths.shape == input_tree.paths.shape
    assert sorted(confidence_tree.paths[:, 0].tolist()) == sorted(input_tree.paths[:, 0].tolist())
    assert set(np.flatnonzero(confidence_tree.paths[:, 0] == 0)) == {2, 3, 4}
    assert diagnostics.assignment_order == "confidence"
    assert diagnostics.levels[0]["mean_assignment_confidence"] > 0.78


def test_update_tree_cached_embeddings_match_uncached_reference() -> None:
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
    vectors = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.1, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.1],
        ],
        dtype=np.float32,
    )
    current_tree = BATLTree.random_init(N=len(vectors), K=2, H=2, alpha=1.0, seed=3)
    torch.manual_seed(5)
    uncached_model = BATLModel(model_cfg)
    cached_model = BATLModel(model_cfg)
    cached_model.load_state_dict(uncached_model.state_dict())
    encoder_calls = {"count": 0}
    original_forward = cached_model.encoder.forward

    def counted_forward(x: torch.Tensor) -> torch.Tensor:
        encoder_calls["count"] += 1
        return original_forward(x)

    cached_model.encoder.forward = counted_forward  # type: ignore[method-assign]

    uncached = update_tree(
        model=uncached_model,
        vectors=vectors,
        current_tree=current_tree,
        batch_size=2,
        device=torch.device("cpu"),
        assignment_order="confidence",
        cache_embeddings=False,
    )
    cached = update_tree(
        model=cached_model,
        vectors=vectors,
        current_tree=current_tree,
        batch_size=2,
        device=torch.device("cpu"),
        assignment_order="confidence",
        cache_embeddings=True,
    )

    assert np.array_equal(cached.paths, uncached.paths)
    assert encoder_calls["count"] == math.ceil(len(vectors) / 2)


def test_decode_group_topr_streams_batches_into_preallocated_numpy_arrays() -> None:
    vectors = np.arange(12, dtype=np.float32).reshape(6, 2)
    model = FixedRoutingModel(K=3, H=2)

    top_probs, top_branches = _decode_group_topr(
        model=model,  # type: ignore[arg-type]
        vectors=vectors,
        vec_indices=np.arange(6, dtype=np.int64),
        path_prefix=(),
        batch_size=2,
        device=torch.device("cpu"),
        top_r=2,
    )

    assert top_probs.shape == (6, 2)
    assert top_probs.dtype == np.float32
    assert top_branches.shape == (6, 2)
    assert model.calls == [((), 2), ((), 2), ((), 2)]
    # FixedRoutingModel puts all mass on branch 0; the remaining branches tie,
    # so the stable rule picks the lower id.
    assert top_branches[:, 0].tolist() == [0] * 6
    assert top_branches[:, 1].tolist() == [1] * 6
    assert (np.diff(top_probs, axis=1) <= 0).all()


def test_resolve_tree_update_cache_embeddings_explicit_and_auto() -> None:
    assert _resolve_tree_update_cache_embeddings(True) is True
    assert _resolve_tree_update_cache_embeddings(False) is False
    # "auto" defaults to False: the cache is n_vectors x embed_dim x float32
    # (~10 GB at Deep10M / embed_dim=256), so leaving it off avoids surprise
    # GPU memory pressure on commodity single-GPU setups.
    assert _resolve_tree_update_cache_embeddings("auto") is False
    with pytest.raises(ValueError, match="must be True, False, or 'auto'"):
        _resolve_tree_update_cache_embeddings("bogus")


def test_alternating_train_runs_tiny_cpu_cycle() -> None:
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
    vectors = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.1, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.1],
        ],
        dtype=np.float32,
    )
    model = BATLModel(model_cfg)

    trained, tree = alternating_train(model, vectors, train_cfg, model_cfg, seed=11)

    assert trained is model
    assert tree.paths.shape == (len(vectors), model_cfg.tree_height)
    assert tree.paths.dtype == np.uint16


def test_alternating_train_early_stops_on_loss_plateau(monkeypatch) -> None:
    from batl import training as training_module

    epoch_loss_sequence = iter(
        [
            1.0,
            1.0,
            0.999,
            0.999,
            0.999,
            0.999,
            0.999,
            0.999,
        ]
    )
    train_epoch_calls = {"count": 0}

    def fake_train_epoch(
        *, model, vectors, query_idx, neighbor_ids, tree_paths, batch_size, device, optimizer, rng
    ):
        train_epoch_calls["count"] += 1
        return next(epoch_loss_sequence)

    monkeypatch.setattr(training_module, "_train_epoch", fake_train_epoch)

    model_cfg = ModelConfig(
        branching_factor=2,
        tree_height=2,
        embedding_dim=3,
        encoder_hidden=4,
        embed_dim=4,
        num_heads=2,
        ff_dim=8,
        dropout=0.0,
        alpha=1.0,
    )
    train_cfg = TrainConfig(
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_epochs=None,
        alternating_interval=2,
        device="cpu",
        top_k_neighbors=1,
        neighbor_search_subset=4,
        convergence_patience=2,
        convergence_min_delta=0.01,
    )
    vectors = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
        dtype=np.float32,
    )
    model = BATLModel(model_cfg)

    alternating_train(model, vectors, train_cfg, model_cfg, seed=7)

    # Cycle 1 sets baseline (loss=1.0). Cycles 2 and 3 plateau at 0.999 (improvement <1%).
    # Stale counter reaches patience=2 after cycle 3 -> break. Total cycles run = 3.
    # Each cycle calls _train_epoch alternating_interval=2 times, so 6 local epochs total.
    assert train_epoch_calls["count"] == 6


def test_alternating_train_runs_full_budget_when_loss_keeps_improving(monkeypatch) -> None:
    from batl import training as training_module

    epoch_loss_sequence = iter([1.0 / (i + 1) for i in range(20)])
    train_epoch_calls = {"count": 0}

    def fake_train_epoch(
        *, model, vectors, query_idx, neighbor_ids, tree_paths, batch_size, device, optimizer, rng
    ):
        train_epoch_calls["count"] += 1
        return next(epoch_loss_sequence)

    monkeypatch.setattr(training_module, "_train_epoch", fake_train_epoch)

    model_cfg = ModelConfig(
        branching_factor=2,
        tree_height=2,
        embedding_dim=3,
        encoder_hidden=4,
        embed_dim=4,
        num_heads=2,
        ff_dim=8,
        dropout=0.0,
        alpha=1.0,
    )
    train_cfg = TrainConfig(
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_epochs=None,
        alternating_interval=2,
        max_alternating_cycles=4,
        device="cpu",
        top_k_neighbors=1,
        neighbor_search_subset=4,
        convergence_patience=2,
        convergence_min_delta=0.01,
    )
    vectors = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
        dtype=np.float32,
    )
    model = BATLModel(model_cfg)

    alternating_train(model, vectors, train_cfg, model_cfg, seed=7)

    # Loss strictly improves each epoch -> never plateaus -> full 4 cycles -> 8 epoch calls.
    assert train_epoch_calls["count"] == 8


def test_alternating_train_rejects_unbounded_run_without_convergence() -> None:
    model_cfg = ModelConfig(
        branching_factor=2,
        tree_height=2,
        embedding_dim=3,
        encoder_hidden=4,
        embed_dim=4,
        num_heads=2,
        ff_dim=8,
        dropout=0.0,
        alpha=1.0,
    )
    train_cfg = TrainConfig(
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_epochs=None,
        alternating_interval=2,
        max_alternating_cycles=None,
        device="cpu",
        top_k_neighbors=1,
        neighbor_search_subset=4,
        convergence_patience=0,
    )
    vectors = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
        dtype=np.float32,
    )
    model = BATLModel(model_cfg)

    with pytest.raises(ValueError, match="stopping condition"):
        alternating_train(model, vectors, train_cfg, model_cfg, seed=7)


def test_alternating_train_advances_neighbor_seed_by_cycle(monkeypatch) -> None:
    from batl import training as training_module

    neighbor_seeds: list[int] = []

    def fake_find_approximate_neighbors(
        queries,
        database,
        top_k,
        subset_size,
        seed,
        mode,
        chunk_size,
        metric="euclidean",
        backend="faiss_cpu",
    ):
        neighbor_seeds.append(seed)
        assert mode == "random_subset"
        assert chunk_size == 1_000_000
        assert backend == "auto"
        return np.zeros((queries.shape[0], top_k), dtype=np.int64)

    def fake_train_epoch(
        *, model, vectors, query_idx, neighbor_ids, tree_paths, batch_size, device, optimizer, rng
    ):
        return 1.0

    monkeypatch.setattr(
        training_module, "find_approximate_neighbors", fake_find_approximate_neighbors
    )
    monkeypatch.setattr(training_module, "_train_epoch", fake_train_epoch)

    model_cfg = ModelConfig(
        branching_factor=2,
        tree_height=2,
        embedding_dim=3,
        encoder_hidden=4,
        embed_dim=4,
        num_heads=2,
        ff_dim=8,
        dropout=0.0,
        alpha=1.0,
    )
    train_cfg = TrainConfig(
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_epochs=None,
        alternating_interval=1,
        max_alternating_cycles=3,
        device="cpu",
        top_k_neighbors=1,
        neighbor_search_subset=4,
        convergence_patience=0,
    )
    vectors = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
        dtype=np.float32,
    )
    model = BATLModel(model_cfg)

    alternating_train(model, vectors, train_cfg, model_cfg, seed=7)

    assert neighbor_seeds == [8, 9, 10]


def test_alternating_train_mines_once_and_refreshes_only_target_paths(monkeypatch, caplog) -> None:
    from batl import training as training_module

    caplog.set_level("INFO", logger="batl.training")
    neighbor_seeds: list[int] = []
    query_ids_seen: list[np.ndarray] = []
    neighbor_ids_seen: list[np.ndarray] = []
    target_paths_seen: list[np.ndarray] = []
    update_count = 0

    def fake_find_approximate_neighbors(
        queries,
        database,
        top_k,
        subset_size,
        seed,
        mode,
        chunk_size,
        metric="euclidean",
        backend="faiss_cpu",
    ):
        del database, subset_size, mode, chunk_size, metric, backend
        neighbor_seeds.append(seed)
        return np.zeros((queries.shape[0], top_k), dtype=np.int64)

    def fake_train_epoch(
        *, model, vectors, query_idx, neighbor_ids, tree_paths, batch_size, device, optimizer, rng
    ):
        del model, vectors, batch_size, device, optimizer, rng
        query_ids_seen.append(query_idx.copy())
        neighbor_ids_seen.append(neighbor_ids.copy())
        target_paths_seen.append(tree_paths[neighbor_ids].copy())
        return 1.0

    def fake_update_tree(*, current_tree, assignment_mode, assignment_order, **kwargs):
        nonlocal update_count
        del kwargs
        update_count += 1
        paths = np.full_like(current_tree.paths, update_count % current_tree.K)
        tree = BATLTree(
            K=current_tree.K,
            H=current_tree.H,
            alpha=current_tree.alpha,
            N=current_tree.N,
            paths=paths,
        )
        diagnostics = TreeUpdateDiagnostics(
            assignment_mode=assignment_mode,
            assignment_order=assignment_order,
            levels=[],
        )
        return tree, diagnostics

    monkeypatch.setattr(
        training_module, "find_approximate_neighbors", fake_find_approximate_neighbors
    )
    monkeypatch.setattr(training_module, "_train_epoch", fake_train_epoch)
    monkeypatch.setattr(training_module, "update_tree", fake_update_tree)

    model_cfg = ModelConfig(
        branching_factor=2,
        tree_height=2,
        embedding_dim=3,
        encoder_hidden=4,
        embed_dim=4,
        num_heads=2,
        ff_dim=8,
        dropout=0.0,
        alpha=1.0,
    )
    train_cfg = TrainConfig(
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        alternating_interval=1,
        max_alternating_cycles=3,
        device="cpu",
        top_k_neighbors=1,
        neighbor_search_subset=4,
        convergence_patience=0,
        label_refresh="once",
    )
    vectors = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
        dtype=np.float32,
    )

    alternating_train(BATLModel(model_cfg), vectors, train_cfg, model_cfg, seed=7)

    assert neighbor_seeds == [8]
    assert len(query_ids_seen) == 3
    assert all(np.array_equal(value, query_ids_seen[0]) for value in query_ids_seen[1:])
    assert all(np.array_equal(value, neighbor_ids_seen[0]) for value in neighbor_ids_seen[1:])
    assert not np.array_equal(target_paths_seen[0], target_paths_seen[1])
    assert not np.array_equal(target_paths_seen[1], target_paths_seen[2])
    assert "label_refresh=once" in caplog.text
    assert "mining_seed=8" in caplog.text


def test_train_epoch_runs_one_optimizer_step_per_batch() -> None:
    model_cfg = ModelConfig(
        branching_factor=2,
        tree_height=1,
        embedding_dim=3,
        encoder_hidden=8,
        embed_dim=8,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
        alpha=1.0,
    )
    model = BATLModel(model_cfg)
    vectors = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.1, 0.0],
        ],
        dtype=np.float32,
    )
    query_idx = np.array([0, 2], dtype=np.int64)
    neighbor_ids = np.array([[1, 3, 5], [0, 4, 5]], dtype=np.int64)
    tree_paths = np.array([[0], [1], [0], [1], [0], [1]], dtype=np.uint16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    step_calls = {"count": 0}
    original_step = optimizer.step

    def counting_step(*args, **kwargs):
        step_calls["count"] += 1
        return original_step(*args, **kwargs)

    optimizer.step = counting_step  # type: ignore[method-assign]

    mean_loss = _train_epoch(
        model=model,
        vectors=vectors,
        query_idx=query_idx,
        neighbor_ids=neighbor_ids,
        tree_paths=tree_paths,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        rng=np.random.default_rng(0),
    )

    # 2 queries x 3 neighbors = 6 pairs, batch_size=2 → 3 optimizer steps.
    assert step_calls["count"] == 3
    assert math.isfinite(mean_loss)


def test_training_target_validation_rejects_tree_paths_outside_model_k() -> None:
    tree_paths = np.array([[0, 1], [127, 128]], dtype=np.uint16)

    with pytest.raises(ValueError, match=r"tree paths.*K=128.*max=128"):
        _validate_target_paths(tree_paths, branch_count=128, context="initial tree")


def test_training_neighbor_validation_rejects_out_of_range_database_indices() -> None:
    neighbor_ids = np.array([[0, 1], [2, 6]], dtype=np.int64)

    with pytest.raises(ValueError, match=r"neighbor ids.*N=6.*max=6"):
        _validate_neighbor_ids(neighbor_ids, n_vectors=6)


def test_train_epoch_uses_memory_bounded_blocks_when_pairs_exceed_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from batl import training as training_module

    model_cfg = ModelConfig(
        branching_factor=2,
        tree_height=1,
        embedding_dim=3,
        encoder_hidden=8,
        embed_dim=8,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
        alpha=1.0,
    )
    model = BATLModel(model_cfg)
    vectors = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.1, 0.0],
        ],
        dtype=np.float32,
    )
    query_idx = np.array([0, 2], dtype=np.int64)
    neighbor_ids = np.array([[1, 3, 5], [0, 4, 5]], dtype=np.int64)
    tree_paths = np.array([[0], [1], [0], [1], [0], [1]], dtype=np.uint16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    repeat_calls: list[int] = []
    original_repeat = np.repeat

    def counted_repeat(a, repeats, axis=None):
        repeat_calls.append(int(a.shape[0]))
        return original_repeat(a, repeats, axis=axis)

    monkeypatch.setattr(training_module, "_MAX_MATERIALIZED_EPOCH_BYTES", 48)
    monkeypatch.setattr(np, "repeat", counted_repeat)

    mean_loss = _train_epoch(
        model=model,
        vectors=vectors,
        query_idx=query_idx,
        neighbor_ids=neighbor_ids,
        tree_paths=tree_paths,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        rng=np.random.default_rng(0),
    )

    assert math.isfinite(mean_loss)
    assert repeat_calls
    assert max(repeat_calls) == 1


def test_num_training_queries_uses_named_one_percent_policy() -> None:
    assert DEFAULT_TRAINING_QUERY_FRACTION == 0.01
    assert _num_training_queries(50) == 1
    assert _num_training_queries(10_000) == 100


def test_relative_loss_improvement_has_no_zero_baseline_fallback() -> None:
    assert _relative_loss_improvement(2.0, 1.0) == 0.5
    assert _relative_loss_improvement(0.0, 0.0) is None


def test_resolve_tree_update_batch_size_auto_cpu_defaults_to_4096() -> None:
    device = torch.device("cpu")
    assert _resolve_tree_update_batch_size(None, num_heads=8, device=device) == 4096
    assert _resolve_tree_update_batch_size("auto", num_heads=8, device=device) == 4096


def test_resolve_tree_update_batch_size_auto_cuda_returns_attention_guard() -> None:
    device = torch.device("cuda")
    # On CUDA, "auto" returns the documented per-launch ceiling (no memory
    # introspection): the CUDA attention guard, 65535 // num_heads.
    assert _resolve_tree_update_batch_size("auto", num_heads=8, device=device) == 65535 // 8
    assert _resolve_tree_update_batch_size(None, num_heads=16, device=device) == 65535 // 16


def test_resolve_tree_update_batch_size_int_passthrough_on_cpu() -> None:
    device = torch.device("cpu")
    assert _resolve_tree_update_batch_size(32768, num_heads=8, device=device) == 32768


def test_resolve_tree_update_batch_size_caps_explicit_cuda_batch_by_heads(caplog) -> None:
    with caplog.at_level("WARNING", logger="batl.tree_update"):
        result = _resolve_tree_update_batch_size(32768, num_heads=8, device=torch.device("cuda"))
    assert result == 65535 // 8
    assert "capping explicit CUDA tree_update_batch_size" in caplog.text


def test_resolve_tree_update_batch_size_rejects_invalid_policy() -> None:
    with pytest.raises(ValueError, match="must be 'auto', None, or an integer"):
        _resolve_tree_update_batch_size(
            "bogus",  # type: ignore[arg-type]
            num_heads=8,
            device=torch.device("cpu"),
        )


def test_cuda_attention_batch_guard_cpu_has_no_limit() -> None:
    assert _cuda_attention_batch_guard(torch.device("cpu"), num_heads=8) == 2**31 - 1


def test_cuda_attention_batch_guard_caps_effective_cuda_attention_work() -> None:
    # PyTorch owns SDPA backend selection. This guard only prevents requesting
    # CUDA attention work larger than a known per-kernel launch dimension.
    # This test exercises the constant path without needing a real CUDA device.
    assert _cuda_attention_batch_guard(torch.device("cuda"), num_heads=8) == 65535 // 8
    assert _cuda_attention_batch_guard(torch.device("cuda"), num_heads=16) == 65535 // 16


# --- _assign_balanced_vectorized ---


def _sequential_assign(
    probs: np.ndarray, K: int, capacity: int, assignment_order: str
) -> np.ndarray:
    """Reference sequential implementation for equivalence testing."""
    N = probs.shape[0]
    sorted_branches = np.argsort(-probs, axis=1)
    local_order = _assignment_order(probs, assignment_order)  # type: ignore[arg-type]
    counts = np.zeros(K, dtype=np.int64)
    branches = np.empty(N, dtype=np.int32)
    for local_pos in local_order:
        b, _, _, _ = _first_available_branch(sorted_branches[local_pos], counts, capacity)
        branches[local_pos] = b
        counts[b] += 1
    return branches


def test_vectorized_assign_respects_capacity() -> None:
    rng = np.random.default_rng(0)
    N, K, capacity = 200, 8, 30
    probs = rng.dirichlet(np.ones(K), size=N).astype(np.float32)
    branches, _, _ = _assign_balanced_vectorized(probs, K, capacity, "confidence")
    counts = np.bincount(branches, minlength=K)
    assert counts.sum() == N
    assert (counts <= capacity).all()


def test_vectorized_assign_all_vectors_placed() -> None:
    rng = np.random.default_rng(1)
    for order in ("confidence", "margin", "input"):
        N, K, capacity = 150, 6, 30
        probs = rng.dirichlet(np.ones(K), size=N).astype(np.float32)
        branches, ranks, fallbacks = _assign_balanced_vectorized(probs, K, capacity, order)  # type: ignore[arg-type]
        assert branches.shape == (N,)
        assert (branches >= 0).all() and (branches < K).all()
        assert ranks.shape == (N,)
        assert fallbacks.shape == (N,)


def test_assignment_sorting_and_margin_helpers_chunk_without_changing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(13)
    probs = rng.random((17, 7), dtype=np.float32)
    monkeypatch.setattr("batl.tree_update.ASSIGNMENT_ARGSORT_CHUNK_ROWS", 4)

    sorted_branches = _sort_branches_by_probability(probs, K=7)
    margins = _assignment_margins(probs)

    assert np.array_equal(sorted_branches, np.argsort(-probs, axis=1).astype(np.int32))
    sorted_probs = np.sort(probs, axis=1)
    assert margins == pytest.approx(sorted_probs[:, -1] - sorted_probs[:, -2])


def test_branch_order_dtype_keeps_deep100m_root_assignment_compact() -> None:
    assert _branch_order_dtype(256) == np.dtype(np.uint8)
    assert _branch_order_dtype(257) == np.dtype(np.uint16)
    assert _branch_order_dtype(65_537) == np.dtype(np.int32)


@pytest.mark.skipif(
    os.environ.get("BATL_RUN_MEMORY_TESTS") != "1",
    reason="set BATL_RUN_MEMORY_TESTS=1 to run large RSS regression checks",
)
def test_deep100m_root_assignment_rss_delta_stays_bounded() -> None:
    rng = np.random.default_rng(17)
    probs = rng.random((1_000_000, 256), dtype=np.float32)
    before = _ru_maxrss_bytes()

    sorted_branches = _sort_branches_by_probability(probs, K=256)

    after = _ru_maxrss_bytes()
    assert sorted_branches.dtype == np.uint8
    assert after - before < 3 * 1024**3


def _ru_maxrss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value
    return value * 1024


def test_vectorized_assign_equivalent_balance_to_sequential() -> None:
    """Vectorized and sequential produce same per-branch count distribution."""
    rng = np.random.default_rng(42)
    N, K = 500, 10
    capacity = int(np.ceil(N / K)) + 5  # slightly loose so no fallbacks
    probs = rng.dirichlet(np.ones(K), size=N).astype(np.float32)

    seq_branches = _sequential_assign(probs, K, capacity, "confidence")
    vec_branches, _, _ = _assign_balanced_vectorized(probs, K, capacity, "confidence")

    seq_counts = np.bincount(seq_branches, minlength=K)
    vec_counts = np.bincount(vec_branches, minlength=K)

    assert seq_counts.sum() == vec_counts.sum() == N
    assert (seq_counts <= capacity).all()
    assert (vec_counts <= capacity).all()
    # Both should have the same number of vectors getting their top choice
    # (within a small tolerance, since tie resolution differs).
    seq_top = int((seq_branches == np.argmax(probs, axis=1)).sum())
    vec_top = int((vec_branches == np.argmax(probs, axis=1)).sum())
    assert abs(seq_top - vec_top) <= N * 0.05  # within 5%


def test_vectorized_assign_fallback_triggered_when_all_branches_full() -> None:
    """With tight capacity, fallback is triggered and all vectors still placed."""
    rng = np.random.default_rng(7)
    N, K = 20, 4
    capacity = 4  # 4 x 4 = 16 < 20, so fallback must fire
    probs = rng.dirichlet(np.ones(K), size=N).astype(np.float32)
    branches, _, used_fallback = _assign_balanced_vectorized(probs, K, capacity, "input")
    assert len(branches) == N
    assert (branches >= 0).all()
    assert used_fallback.any()


# --- C2b: top-R assignment equivalence ---


def _topr_inputs(probs: np.ndarray, K: int, top_r: int):
    """Derive top-R inputs and a resolve_full callback from a full probs matrix."""
    from batl.tree_update import _sort_branches_by_probability

    order = _sort_branches_by_probability(probs, K)
    top_branches = order[:, :top_r]
    top_probs = np.take_along_axis(probs, top_branches.astype(np.int64), axis=1)
    calls: list[int] = []

    def resolve_full(local_positions: np.ndarray) -> np.ndarray:
        calls.append(int(local_positions.size))
        return order[local_positions]

    return top_probs, top_branches, resolve_full, calls


@pytest.mark.parametrize("K", [8, 64, 256])
@pytest.mark.parametrize("order_mode", ["input", "confidence", "margin"])
@pytest.mark.parametrize("top_r", [2, 4, 16])
def test_topr_assignment_matches_full_sort_reference(K: int, order_mode: str, top_r: int) -> None:
    from batl.tree_update import _assign_balanced_topr, _assign_balanced_vectorized

    rng = np.random.default_rng(K * 100 + top_r)
    n = 2_000
    logits = rng.normal(size=(n, K)).astype(np.float32)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    # Capacity below n/K forces heavy contention, so many vectors are pushed
    # past their top choices and the straggler path gets exercised.
    capacity = max(1, n // K // 2)

    expected_branches, expected_ranks, expected_fallback = _assign_balanced_vectorized(
        probs, K, capacity, order_mode
    )
    top_probs, top_branches, resolve_full, _calls = _topr_inputs(probs, K, min(top_r, K))
    branches, ranks, fallback, _probs, _stragglers = _assign_balanced_topr(
        top_probs, top_branches, K, capacity, order_mode, resolve_full
    )

    assert branches.tolist() == expected_branches.tolist()
    assert ranks.tolist() == expected_ranks.tolist()
    assert fallback.tolist() == expected_fallback.tolist()


def test_topr_assignment_matches_reference_on_exactly_tied_probabilities() -> None:
    from batl.tree_update import _assign_balanced_topr, _assign_balanced_vectorized

    K, n = 4, 40
    # Every branch equally likely: the tie rule alone decides the ordering.
    probs = np.full((n, K), 0.25, dtype=np.float32)
    capacity = n // K

    expected = _assign_balanced_vectorized(probs, K, capacity, "input")
    top_probs, top_branches, resolve_full, _calls = _topr_inputs(probs, K, 2)
    branches, ranks, fallback, _probs, _stragglers = _assign_balanced_topr(
        top_probs, top_branches, K, capacity, "input", resolve_full
    )

    assert branches.tolist() == expected[0].tolist()
    assert ranks.tolist() == expected[1].tolist()
    assert fallback.tolist() == expected[2].tolist()


def test_sequential_assignment_differs_from_round_input_when_retry_branch_fills() -> None:
    from batl.tree_update import _assign_balanced_sequential, _assign_balanced_topr

    probs = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    branch_orders = np.argsort(-probs, axis=1, kind="stable").astype(np.uint8)
    ranked_probs = np.take_along_axis(probs, branch_orders, axis=1)

    def resolve_full(local_positions: np.ndarray) -> np.ndarray:
        return branch_orders[local_positions]

    round_result = _assign_balanced_topr(
        ranked_probs,
        branch_orders,
        K=2,
        capacity=1,
        assignment_order="input",
        resolve_full=resolve_full,
    )
    sequential_result = _assign_balanced_sequential(
        ranked_probs,
        branch_orders,
        K=2,
        capacity=1,
        assignment_order="input",
    )

    assert round_result[0].tolist() == [0, 0, 1]
    assert sequential_result[0].tolist() == [0, 1, 0]
    assert sequential_result[1].tolist() == [0, 1, 2]
    assert sequential_result[2].tolist() == [False, False, True]
    assert sequential_result[3][:2].tolist() == pytest.approx([0.9, 0.2])
    assert np.isnan(sequential_result[3][2])
    assert sequential_result[4] == 0


def test_sequential_assignment_confidence_order_is_deterministic() -> None:
    from batl.tree_update import _assign_balanced_sequential

    probs = np.array(
        [
            [0.6, 0.4],
            [0.9, 0.1],
            [0.4, 0.6],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    branch_orders = np.argsort(-probs, axis=1, kind="stable").astype(np.uint8)
    ranked_probs = np.take_along_axis(probs, branch_orders, axis=1)

    first = _assign_balanced_sequential(
        ranked_probs,
        branch_orders,
        K=2,
        capacity=2,
        assignment_order="confidence",
    )
    second = _assign_balanced_sequential(
        ranked_probs,
        branch_orders,
        K=2,
        capacity=2,
        assignment_order="confidence",
    )

    assert first[0].tolist() == [0, 0, 1, 1]
    for left, right in zip(first[:4], second[:4], strict=True):
        assert np.array_equal(left, right, equal_nan=True)


def test_update_tree_supports_literal_sequential_assignment() -> None:
    vectors = np.arange(16, dtype=np.float32).reshape(8, 2)
    current_tree = BATLTree.random_init(N=8, K=2, H=2, alpha=1.0, seed=4)
    model = FixedRoutingModel(K=2, H=2)

    updated, diagnostics = update_tree(
        model=model,  # type: ignore[arg-type]
        vectors=vectors,
        current_tree=current_tree,
        batch_size=4,
        device=torch.device("cpu"),
        assignment_mode="sequential",
        assignment_order="input",
        return_diagnostics=True,
    )

    assert updated.paths.shape == (8, 2)
    assert updated.leaf_size_stats()["max"] <= 2
    assert diagnostics.assignment_mode == "sequential"
    assert diagnostics.assignment_order == "input"


def test_update_tree_rejects_unknown_assignment_mode() -> None:
    vectors = np.arange(8, dtype=np.float32).reshape(4, 2)
    current_tree = BATLTree.random_init(N=4, K=2, H=1, alpha=1.0, seed=4)

    with pytest.raises(ValueError, match="assignment_mode"):
        update_tree(
            model=FixedRoutingModel(K=2, H=1),  # type: ignore[arg-type]
            vectors=vectors,
            current_tree=current_tree,
            batch_size=4,
            device=torch.device("cpu"),
            assignment_mode="unknown",  # type: ignore[arg-type]
        )


def test_update_tree_rejects_truncated_top_r_for_sequential_assignment() -> None:
    vectors = np.arange(16, dtype=np.float32).reshape(8, 2)
    current_tree = BATLTree.random_init(N=8, K=4, H=1, alpha=1.0, seed=4)

    with pytest.raises(ValueError, match="full-K"):
        update_tree(
            model=FixedRoutingModel(K=4, H=1),  # type: ignore[arg-type]
            vectors=vectors,
            current_tree=current_tree,
            batch_size=4,
            device=torch.device("cpu"),
            assignment_mode="sequential",
            top_r=2,
        )


def test_topr_assignment_resolves_only_stragglers() -> None:
    from batl.tree_update import _assign_balanced_topr

    K, n = 16, 400
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(n, K)).astype(np.float32)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    capacity = n // K  # exactly tight: every branch fills, forcing deep ranks

    top_probs, top_branches, resolve_full, calls = _topr_inputs(probs, K, 4)
    _branches, ranks, fallback, _probs, stragglers = _assign_balanced_topr(
        top_probs, top_branches, K, capacity, "confidence", resolve_full
    )

    assert stragglers == sum(calls)
    assert stragglers < n  # only the tail needs a full-K ordering
    # Every vector that ended up at rank >= 4 must have been resolved.
    assert stragglers >= int(((ranks >= 4) & ~fallback).sum())


def test_topr_assignment_never_resolves_when_r_equals_k() -> None:
    from batl.tree_update import _assign_balanced_topr

    K, n = 8, 200
    rng = np.random.default_rng(11)
    logits = rng.normal(size=(n, K)).astype(np.float32)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    top_probs, top_branches, resolve_full, calls = _topr_inputs(probs, K, K)
    _b, _r, _f, _p, stragglers = _assign_balanced_topr(
        top_probs, top_branches, K, max(1, n // K), "margin", resolve_full
    )

    assert calls == []
    assert stragglers == 0


def test_topr_assignment_rejects_degenerate_r() -> None:
    from batl.tree_update import _assign_balanced_topr

    probs = np.full((4, 1), 1.0, dtype=np.float32)
    branches = np.zeros((4, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="R >= 2"):
        _assign_balanced_topr(probs, branches, 4, 2, "input", lambda pos: np.zeros((pos.size, 4)))

    with pytest.raises(ValueError, match="R cannot exceed K"):
        _assign_balanced_topr(
            np.zeros((4, 8), dtype=np.float32),
            np.zeros((4, 8), dtype=np.uint8),
            4,
            2,
            "input",
            lambda pos: np.zeros((pos.size, 4)),
        )


def test_update_tree_top_r_produces_the_same_tree_as_full_k() -> None:
    from batl.tree_update import update_tree

    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(600, 8)).astype(np.float32)
    torch.manual_seed(3)
    model = BATLModel(
        ModelConfig(
            branching_factor=8,
            tree_height=2,
            embedding_dim=8,
            encoder_hidden=32,
            embed_dim=16,
            num_heads=2,
            ff_dim=32,
            dropout=0.0,
            alpha=1.0,
        )
    )
    tree = BATLTree.random_init(N=600, K=8, H=2, alpha=1.0, seed=3)

    truncated, _diag = update_tree(
        model=model,
        vectors=vectors,
        current_tree=tree,
        batch_size=128,
        device=torch.device("cpu"),
        assignment_order="confidence",
        return_diagnostics=True,
        top_r=2,
    )
    full, _full_diag = update_tree(
        model=model,
        vectors=vectors,
        current_tree=tree,
        batch_size=128,
        device=torch.device("cpu"),
        assignment_order="confidence",
        return_diagnostics=True,
        top_r=8,
    )

    assert truncated.paths.tolist() == full.paths.tolist()


def test_topr_assignment_allocates_far_less_than_full_k_ordering() -> None:
    """The point of top-R: host arrays scale with R, not K.

    Sizes are checked directly rather than through RSS so this runs in normal
    CI; the opt-in RSS test above covers the full-K path it replaces.
    """
    from batl.tree_update import _branch_order_dtype

    n, K, R = 100_000, 256, 16
    branch_bytes = _branch_order_dtype(K).itemsize
    # The replaced path held both the (n, K) probs matrix and its (n, K)
    # ordering on the host at once.
    full_bytes = n * K * np.dtype(np.float32).itemsize + n * K * branch_bytes
    topr_bytes = n * R * (np.dtype(np.float32).itemsize + branch_bytes)

    assert topr_bytes * 16 <= full_bytes

    rng = np.random.default_rng(19)
    logits = rng.normal(size=(2_000, K)).astype(np.float32)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    top_probs, top_branches, _resolve_full, _calls = _topr_inputs(probs, K, R)

    assert top_probs.nbytes + top_branches.nbytes < probs.nbytes // 4
    assert top_branches.dtype == np.uint8


# --- C4: exact labels when the mining subset covers the whole database ---


def test_full_database_subset_returns_exact_neighbors() -> None:
    """subset_size >= N must give true nearest neighbors, not a sample."""
    rng = np.random.default_rng(4)
    database = rng.normal(size=(5_000, 8)).astype(np.float32)
    queries = rng.normal(size=(25, 8)).astype(np.float32)

    neighbors = find_approximate_neighbors(
        queries=queries,
        database=database,
        top_k=10,
        subset_size=database.shape[0],
        seed=0,
        backend="faiss_cpu",
    )

    distances = np.linalg.norm(queries[:, None, :] - database[None, :, :], axis=2)
    expected = np.argsort(distances, axis=1)[:, :10]
    assert neighbors.tolist() == expected.tolist()


def test_subset_size_above_database_size_is_clamped_to_exact() -> None:
    rng = np.random.default_rng(5)
    database = rng.normal(size=(500, 4)).astype(np.float32)
    queries = rng.normal(size=(5, 4)).astype(np.float32)

    exact = find_approximate_neighbors(
        queries=queries, database=database, top_k=5, subset_size=500, seed=0, backend="faiss_cpu"
    )
    oversized = find_approximate_neighbors(
        queries=queries, database=database, top_k=5, subset_size=10**6, seed=0, backend="faiss_cpu"
    )

    assert exact.tolist() == oversized.tolist()


def test_full_database_mining_uses_the_chunked_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact-label path must not fancy-index a second full-size copy.

    Checked by which builder runs: the chunked one searches the database in
    `chunk_size` slices, while the random_subset path materializes
    `database[subset_indices]` in one allocation.
    """
    import batl.neighbor_search as ns

    rng = np.random.default_rng(6)
    database = rng.normal(size=(400, 4)).astype(np.float32)
    queries = rng.normal(size=(3, 4)).astype(np.float32)
    original = ns._search_subset_in_chunks
    calls: list[int] = []

    def spy(**kwargs):
        calls.append(int(kwargs["subset_indices"].size))
        return original(**kwargs)

    monkeypatch.setattr(ns, "_search_subset_in_chunks", spy)

    exact = ns.find_approximate_neighbors(
        queries=queries,
        database=database,
        top_k=5,
        subset_size=400,
        seed=0,
        chunk_size=100,
        mode="random_subset",
        backend="faiss_cpu",
    )
    assert calls == [400]

    calls.clear()
    ns.find_approximate_neighbors(
        queries=queries,
        database=database,
        top_k=5,
        subset_size=200,
        seed=0,
        chunk_size=100,
        mode="random_subset",
        backend="faiss_cpu",
    )
    # A genuine subset still takes the fancy-index path, so the assertion above
    # is measuring the routing change and not a no-op.
    assert calls == []

    # Routing changed; results must not.
    distances = np.linalg.norm(queries[:, None, :] - database[None, :, :], axis=2)
    assert exact.tolist() == np.argsort(distances, axis=1)[:, :5].tolist()


@pytest.mark.parametrize("chunk_size", [7, 64, 399, 400, 10**6])
def test_chunked_search_is_invariant_to_chunk_size(chunk_size: int) -> None:
    """Chunking bounds device memory; it must not change which neighbors win.

    The chunked path exists because one flat index over the whole subset does
    not fit on a single card at Deep100M. That is only a valid substitution if
    the merge of per-chunk top-k equals the global top-k, including when a
    chunk holds fewer than top_k vectors.
    """
    rng = np.random.default_rng(7)
    database = rng.normal(size=(400, 4)).astype(np.float32)
    queries = rng.normal(size=(6, 4)).astype(np.float32)

    neighbors = find_approximate_neighbors(
        queries=queries,
        database=database,
        top_k=10,
        subset_size=database.shape[0],
        seed=0,
        chunk_size=chunk_size,
        backend="faiss_cpu",
    )

    distances = np.linalg.norm(queries[:, None, :] - database[None, :, :], axis=2)
    assert neighbors.tolist() == np.argsort(distances, axis=1)[:, :10].tolist()


def test_chunked_search_handles_a_strict_subset() -> None:
    """`sequential_chunked` mode takes the chunked path with a partial subset.

    Chunks then hold only the sampled rows, and the returned ids must still be
    database indices rather than positions within a chunk.
    """
    rng = np.random.default_rng(8)
    database = rng.normal(size=(300, 5)).astype(np.float32)
    queries = rng.normal(size=(4, 5)).astype(np.float32)

    neighbors = find_approximate_neighbors(
        queries=queries,
        database=database,
        top_k=5,
        subset_size=120,
        seed=3,
        chunk_size=37,
        mode="sequential_chunked",
        backend="faiss_cpu",
    )

    subset = np.sort(np.random.default_rng(3).choice(300, size=120, replace=False))
    distances = np.linalg.norm(queries[:, None, :] - database[None, subset, :], axis=2)
    expected = subset[np.argsort(distances, axis=1)[:, :5]]
    assert neighbors.tolist() == expected.tolist()
