"""Beam-search retrieval for BATL indexes."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, get_args

import numpy as np
import torch
import torch.nn.functional as F

from batl.constants import (
    DEFAULT_ENSEMBLE_MIN_TREE_MATCHES,
    DEFAULT_RETRIEVAL_TOP_K,
    RERANK_GROUP_QUERIES,
)
from batl.distance import compute_distances, compute_distances_torch, metrics
from batl.model import BATLModel
from batl.profiling import StageProfiler
from batl.rerank import ResidentGpuReranker
from batl.tree import BATLTree
from batl.utils.data import as_float32_matrix

RerankBackend = Literal["numpy_cpu", "torch_gpu", "torch_gpu_resident"]
_GPU_RERANK_BACKENDS = {"torch_gpu", "torch_gpu_resident"}


def search_batch(
    models: list[BATLModel],
    trees: list[BATLTree],
    database: np.ndarray,
    queries: np.ndarray,
    beam_size: int,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    return_candidate_counts: bool = False,
    num_return_leaves: int | None = None,
    metric: str = "euclidean",
    rerank_backend: RerankBackend = "numpy_cpu",
    rerank_workers: int | None = None,
    reranker: ResidentGpuReranker | None = None,
    profiler: StageProfiler | None = None,
    min_trees: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Search a batch of queries and rerank merged leaf candidates exactly.

    ``min_trees`` controls candidate-frequency filtering. When omitted, a
    single tree returns the union and an ensemble keeps candidates appearing
    in at least two trees. Any empty filtered set falls back to the full union.

    ``metric`` controls exact reranking. Definitions follow ANN-Benchmarks:
    ``"euclidean"`` is the L2 norm, ``"angular"`` is ``1 - cosine_similarity``,
    ``"hamming"`` is mean boolean xor, and ``"jaccard"`` is set Jaccard distance.
    ``"cosine"`` is kept as a compatibility alias for ANN-Benchmarks angular.
    ``"inner_product"`` is the raw (unnormalized) dot product, negated so that
    smaller is still closer like every other metric here; unlike angular, it
    does not divide out vector magnitude.

    ``rerank_backend="torch_gpu_resident"`` reranks the whole query batch
    against a database already resident on the GPU and requires ``reranker``;
    build one per process (see ``batl.rerank.ResidentGpuReranker``) and reuse it
    across a sweep.
    """
    if len(models) != len(trees):
        raise ValueError("models and trees must have the same length.")
    resolved_min_trees = _resolve_min_trees(min_trees, len(trees))
    if metric not in {*metrics.keys(), "cosine"}:
        raise ValueError(
            "metric must be 'euclidean', 'angular', 'hamming', 'jaccard', 'inner_product', "
            "or 'cosine'."
        )
    _validate_rerank_backend(rerank_backend, metric=metric, reranker=reranker)

    database = as_float32_matrix(database, "database")
    if database.shape[1] != queries.shape[1]:
        raise ValueError("database and queries must have the same vector dimension.")

    stages = profiler or StageProfiler(enabled=False)
    n_queries = queries.shape[0]
    candidate_parts_by_query: list[list[np.ndarray]] = [[] for _ in range(n_queries)]
    selected_leaf_ids_by_tree: list[np.ndarray] = []
    for model, tree in zip(models, trees, strict=True):
        with stages.stage("search.beam_decode"):
            _, beam_paths = _beam_search_batch_tensors(
                model=model,
                queries=queries,
                H=tree.H,
                beam_size=beam_size,
                num_return_leaves=num_return_leaves,
            )
            leaf_ids = _paths_to_leaf_ids(beam_paths, tree.K).detach().cpu().numpy()
        selected_leaf_ids_by_tree.append(leaf_ids)
        with stages.stage("search.leaf_lookup"):
            for query_parts, query_leaf_ids in zip(candidate_parts_by_query, leaf_ids, strict=True):
                query_parts.extend(
                    tree._get_leaf_indices_by_id(int(leaf_id)) for leaf_id in query_leaf_ids
                )

    return search_batch_candidates(
        trees_count=len(trees),
        database=database,
        queries=queries,
        top_k=top_k,
        return_candidate_counts=return_candidate_counts,
        metric=metric,
        rerank_backend=rerank_backend,
        rerank_device=_model_device(models[0]),
        rerank_workers=rerank_workers,
        n_queries=n_queries,
        candidate_parts_by_query=candidate_parts_by_query,
        min_trees=resolved_min_trees,
        reranker=reranker,
        profiler=stages,
    )


def search_batch_candidates(
    trees_count: int,
    database: np.ndarray,
    queries: np.ndarray,
    top_k: int,
    n_queries: int,
    candidate_parts_by_query: list[list[np.ndarray]],
    return_candidate_counts: bool = False,
    metric: str = "euclidean",
    rerank_backend: RerankBackend = "numpy_cpu",
    rerank_device: torch.device | None = None,
    rerank_workers: int | None = None,
    reranker: ResidentGpuReranker | None = None,
    profiler: StageProfiler | None = None,
    min_trees: int | None = None,
):
    stages = profiler or StageProfiler(enabled=False)
    results = np.full((n_queries, top_k), -1, dtype=np.int64)
    queries = as_float32_matrix(queries, "queries")
    n_candidates = np.zeros(n_queries, dtype=np.int64)
    resolved_min_trees = _resolve_min_trees(min_trees, trees_count)

    if rerank_backend == "torch_gpu_resident":
        _validate_rerank_backend(rerank_backend, metric=metric, reranker=reranker)
        assert reranker is not None  # narrowed by _validate_rerank_backend
        results, n_candidates = _rerank_resident(
            reranker=reranker,
            queries=queries,
            candidate_parts_by_query=candidate_parts_by_query,
            trees_count=trees_count,
            min_trees=resolved_min_trees,
            top_k=top_k,
            metric=metric,
            n_queries=n_queries,
            profiler=stages,
        )
        if return_candidate_counts:
            return results, n_candidates
        return results

    worker_count = _resolve_rerank_workers(rerank_workers, rerank_backend, n_queries)

    items = list(enumerate(queries))
    # The CPU path fuses candidate selection and distance computation per query,
    # so it reports one combined stage rather than splitting merge from rerank.
    with stages.stage("search.select_and_rerank_cpu"):
        rows = _rerank_all_queries(
            items=items,
            worker_count=worker_count,
            candidate_parts_by_query=candidate_parts_by_query,
            trees_count=trees_count,
            min_trees=resolved_min_trees,
            database=database,
            top_k=top_k,
            metric=metric,
            rerank_backend=rerank_backend,
            rerank_device=rerank_device or torch.device("cpu"),
        )

    for query_idx, candidates_size, reranked in rows:
        n_candidates[query_idx] = candidates_size
        results[query_idx, : reranked.size] = reranked

    if return_candidate_counts:
        return results, n_candidates
    return results


def _rerank_all_queries(
    *,
    items: list[tuple[int, np.ndarray]],
    worker_count: int,
    candidate_parts_by_query: list[list[np.ndarray]],
    trees_count: int,
    min_trees: int,
    database: np.ndarray,
    top_k: int,
    metric: str,
    rerank_backend: RerankBackend,
    rerank_device: torch.device,
) -> list[tuple[int, int, np.ndarray]]:
    if worker_count == 1:
        return [
            _rerank_query(
                item=item,
                candidate_parts_by_query=candidate_parts_by_query,
                trees_count=trees_count,
                min_trees=min_trees,
                database=database,
                top_k=top_k,
                metric=metric,
                rerank_backend=rerank_backend,
                rerank_device=rerank_device,
            )
            for item in items
        ]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(
            executor.map(
                lambda item: _rerank_query(
                    item=item,
                    candidate_parts_by_query=candidate_parts_by_query,
                    trees_count=trees_count,
                    min_trees=min_trees,
                    database=database,
                    top_k=top_k,
                    metric=metric,
                    rerank_backend=rerank_backend,
                    rerank_device=rerank_device,
                ),
                items,
            )
        )


def _rerank_query(
    *,
    item: tuple[int, np.ndarray],
    candidate_parts_by_query: list[list[np.ndarray]],
    trees_count: int,
    min_trees: int,
    database: np.ndarray,
    top_k: int,
    metric: str,
    rerank_backend: RerankBackend,
    rerank_device: torch.device,
) -> tuple[int, int, np.ndarray]:
    query_idx, query = item
    candidates, _filtered_count = _select_candidates(
        candidate_parts_by_query[query_idx],
        n_trees=trees_count,
        min_trees=min_trees,
    )
    if candidates.size == 0:
        return query_idx, 0, np.empty(0, dtype=np.int64)
    dists = _compute_rerank_distances(
        database=database,
        candidates=candidates,
        query=query,
        metric=metric,
        rerank_backend=rerank_backend,
        device=rerank_device,
    )
    if candidates.size > top_k:
        selected = np.argpartition(dists, top_k - 1)[:top_k]
        selected = selected[np.argsort(dists[selected], kind="mergesort")]
    else:
        selected = np.argsort(dists, kind="mergesort")
    return query_idx, int(candidates.size), candidates[selected]


def per_query_rerank_backend(rerank_backend: RerankBackend) -> RerankBackend:
    """Return the closest backend usable without a resident reranker.

    Cycle diagnostics rerank inside the training loop, where a second full copy
    of the database cannot share VRAM with the model and optimizer, so the
    resident backend degrades to the per-query GPU path there.
    """
    return "torch_gpu" if rerank_backend == "torch_gpu_resident" else rerank_backend


def _validate_rerank_backend(
    rerank_backend: str,
    *,
    metric: str,
    reranker: ResidentGpuReranker | None,
) -> None:
    valid = get_args(RerankBackend)
    if rerank_backend not in valid:
        raise ValueError(f"rerank_backend must be one of {valid}.")
    if rerank_backend in _GPU_RERANK_BACKENDS and _canonical_metric(metric) not in {
        "euclidean",
        "angular",
        "inner_product",
    }:
        raise ValueError(
            f"{rerank_backend} rerank supports only euclidean, angular, cosine, "
            "and inner_product metrics."
        )
    if rerank_backend == "torch_gpu_resident" and reranker is None:
        raise ValueError(
            "rerank_backend='torch_gpu_resident' requires reranker=ResidentGpuReranker(...); "
            "build it once per process and reuse it across the sweep."
        )


def _rerank_resident(
    *,
    reranker: ResidentGpuReranker,
    queries: np.ndarray,
    candidate_parts_by_query: list[list[np.ndarray]],
    trees_count: int,
    min_trees: int,
    top_k: int,
    metric: str,
    n_queries: int,
    profiler: StageProfiler | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rerank a whole query chunk against the GPU-resident database."""
    stages = profiler or StageProfiler(enabled=False)
    with stages.stage("search.candidate_merge"):
        candidate_sets = [
            _select_candidates(
                candidate_parts_by_query[query_idx],
                n_trees=trees_count,
                min_trees=min_trees,
            )[0]
            for query_idx in range(n_queries)
        ]
        n_candidates = np.array([candidates.size for candidates in candidate_sets], dtype=np.int64)
        max_candidates = int(n_candidates.max()) if n_queries else 0
    if max_candidates == 0:
        return np.full((n_queries, top_k), -1, dtype=np.int64), n_candidates

    # Group queries of similar candidate count together, because the resident
    # reranker pays for the padded matrix width while numpy_cpu pays per actual
    # candidate. Padding every query to the global maximum is nearly free at
    # T=1, where the frequency filter is inactive and counts vary by ~2%, and
    # ruinous at T=4, where they vary by 17-35%: job 22848765 measured the
    # resident path at 0.27-0.45x of numpy_cpu on a T=4 index, against 1.76-5.37x
    # on a T=1 index of the same dataset and card. Sorting first makes each
    # group's padding tight.
    order = np.argsort(n_candidates, kind="stable")
    results = np.full((n_queries, top_k), -1, dtype=np.int64)
    with stages.stage("search.rerank_resident"):
        for start in range(0, n_queries, RERANK_GROUP_QUERIES):
            rows = order[start : start + RERANK_GROUP_QUERIES]
            width = int(n_candidates[rows].max())
            if width == 0:
                continue
            padded = np.full((rows.size, width), -1, dtype=np.int64)
            for slot, row in enumerate(rows):
                candidates = candidate_sets[row]
                padded[slot, : candidates.size] = candidates
            results[rows] = reranker.rerank_batch(
                queries[rows], padded, top_k, _canonical_metric(metric)
            )
    return results, n_candidates


def _resolve_rerank_workers(
    requested: int | None,
    rerank_backend: RerankBackend,
    n_queries: int,
) -> int:
    if requested is not None and requested <= 0:
        raise ValueError("rerank_workers must be positive when provided.")
    if n_queries <= 1 or rerank_backend != "numpy_cpu":
        return 1
    if requested is not None:
        return min(requested, n_queries)
    return min(n_queries, os.cpu_count() or 1, 32)


def _canonical_metric(metric: str) -> str:
    return "angular" if metric == "cosine" else metric


def _compute_rerank_distances(
    *,
    database: np.ndarray,
    candidates: np.ndarray,
    query: np.ndarray,
    metric: str,
    rerank_backend: RerankBackend,
    device: torch.device,
) -> np.ndarray:
    """Compute exact rerank distances using the requested backend."""
    metric = _canonical_metric(metric)
    if rerank_backend == "numpy_cpu":
        return compute_distances(database, candidates, query, metric)
    if rerank_backend != "torch_gpu":
        raise ValueError("per-query rerank must use 'numpy_cpu' or 'torch_gpu'.")
    if metric not in {"euclidean", "angular", "inner_product"}:
        raise ValueError(
            "torch_gpu rerank supports only euclidean, angular, cosine, and inner_product metrics."
        )
    return compute_distances_torch(database, candidates, query, metric, device)


def _beam_search_batch_tensors(
    model: BATLModel,
    queries: np.ndarray,
    H: int,
    beam_size: int,
    num_return_leaves: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return batched beam scores and paths as tensors."""
    queries = _validate_beam_inputs(queries)
    _validate_return_leaves(num_return_leaves, beam_size)
    K = model.K
    max_leaves = K**H
    beam_limit = min(beam_size, max_leaves)
    # Only the K**H cap remains, and it is a completeness bound: there are no
    # further leaves to return. M > beam_size was rejected above.
    return_limit = min(num_return_leaves or beam_size, beam_limit)
    device = _model_device(model)
    was_training = model.training
    model.eval()

    try:
        query_tensor = torch.as_tensor(queries, dtype=torch.float32, device=device)
        if query_tensor.shape[0] == 0:
            return (
                torch.empty((0, return_limit), dtype=torch.float32, device=device),
                torch.empty((0, return_limit, H), dtype=torch.long, device=device),
            )
        with torch.inference_mode():
            encoded_queries = model.encode(query_tensor)
            n_queries = encoded_queries.shape[0]
            beam_scores = torch.zeros((n_queries, 1), dtype=torch.float32, device=device)
            beam_paths = torch.empty((n_queries, 1, 0), dtype=torch.long, device=device)

            for depth in range(H):
                current_beam = beam_scores.shape[1]
                flat_paths = beam_paths.reshape(n_queries * current_beam, depth)
                start_tokens = torch.full(
                    (flat_paths.shape[0], 1),
                    model.START_TOKEN,
                    dtype=torch.long,
                    device=device,
                )
                prefix_ids = torch.cat([start_tokens, flat_paths], dim=1)
                memory = encoded_queries.repeat_interleave(current_beam, dim=0)
                logits = model.decoder(prefix_ids, memory)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1).reshape(
                    n_queries,
                    current_beam,
                    K,
                )
                candidate_scores = (log_probs + beam_scores[:, :, None]).flatten(start_dim=1)
                keep = min(beam_limit, candidate_scores.shape[1])
                beam_scores, top_indices = torch.topk(candidate_scores, k=keep, dim=1)
                parent_indices = torch.div(top_indices, K, rounding_mode="floor")
                branches = top_indices.remainder(K)

                if depth:
                    parent_paths = torch.gather(
                        beam_paths,
                        dim=1,
                        index=parent_indices[:, :, None].expand(-1, -1, depth),
                    )
                else:
                    parent_paths = torch.empty(
                        (n_queries, keep, 0), dtype=torch.long, device=device
                    )
                beam_paths = torch.cat([parent_paths, branches[:, :, None]], dim=2)

        return beam_scores[:, :return_limit], beam_paths[:, :return_limit]
    finally:
        if was_training:
            model.train()


def _validate_return_leaves(num_return_leaves: int | None, beam_size: int) -> None:
    """Reject M > beam_size instead of silently returning beam_size leaves.

    Beam search only ever holds ``beam_size`` prefixes, so a larger M cannot
    produce more leaves. Capping it silently made benchmark rows report the
    requested M while the search used ``min(M, b)`` — a Deep10M sweep with
    ``beam_size=100`` emitted M=150 and M=200 rows that were byte-identical
    copies of M=100, with `estimated_candidate_set_size` still scaled to the
    request. Failing loudly is the only way those rows cannot be produced.
    """
    if num_return_leaves is None:
        return
    if num_return_leaves <= 0:
        raise ValueError(f"num_return_leaves (M) must be positive; got M={num_return_leaves}.")
    if num_return_leaves > beam_size:
        raise ValueError(
            "num_return_leaves (M) must be <= beam_size (b); "
            f"got M={num_return_leaves}, b={beam_size}."
        )


def _validate_beam_inputs(
    queries: np.ndarray,
) -> np.ndarray:
    query_matrix = np.ascontiguousarray(as_float32_matrix(queries, "queries"), dtype=np.float32)
    if not query_matrix.flags.writeable:
        query_matrix = query_matrix.copy()
    return query_matrix


def _paths_to_leaf_ids(paths: torch.Tensor, K: int) -> torch.Tensor:
    H = paths.shape[-1]
    powers = K ** torch.arange(H - 1, -1, -1, dtype=torch.long, device=paths.device)
    return (paths.to(dtype=torch.long) * powers).sum(dim=-1)


def _select_candidates(
    candidate_parts: list[np.ndarray],
    n_trees: int,
    min_trees: int | None = None,
) -> tuple[np.ndarray, int]:
    resolved_min_trees = _resolve_min_trees(min_trees, n_trees)
    non_empty = [part for part in candidate_parts if part.size]
    if not non_empty:
        return np.empty(0, dtype=np.int64), 0

    merged = np.concatenate(non_empty).astype(np.int64, copy=False)
    unique, counts = np.unique(merged, return_counts=True)

    if resolved_min_trees == 1:
        return unique, int(unique.size)

    filtered = unique[counts >= resolved_min_trees]
    if filtered.size > 0:
        return filtered, int(filtered.size)
    return unique, int(unique.size)


def _resolve_min_trees(requested: int | None, n_trees: int) -> int:
    """Resolve the frequency threshold while preserving historical defaults."""
    if n_trees <= 0:
        raise ValueError("search requires at least one tree.")
    resolved = (
        requested
        if requested is not None
        else (1 if n_trees == 1 else DEFAULT_ENSEMBLE_MIN_TREE_MATCHES)
    )
    if not 1 <= resolved <= n_trees:
        raise ValueError(f"min_trees must be in [1, {n_trees}] for this search; got {resolved}.")
    return resolved


def _model_device(model: torch.nn.Module) -> torch.device:
    param = next(model.parameters(), None)
    if param is None:
        return torch.device("cpu")
    return param.device
