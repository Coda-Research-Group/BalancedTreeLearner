"""Alternating optimisation loop for BATL (model training + tree update)."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from batl.constants import DEFAULT_TRAINING_QUERY_FRACTION
from batl.model import BATLModel, batl_loss
from batl.neighbor_search import find_approximate_neighbors as find_approximate_neighbors
from batl.profiling import StageProfiler
from batl.tree import BATLTree
from batl.tree_update import (
    AssignmentMode as AssignmentMode,
)
from batl.tree_update import (
    AssignmentOrder as AssignmentOrder,
)
from batl.tree_update import (
    TreeUpdateDiagnostics as TreeUpdateDiagnostics,
)
from batl.tree_update import (
    _resolve_tree_update_batch_size,
    _resolve_tree_update_cache_embeddings,
)
from batl.tree_update import (
    update_tree as update_tree,
)
from batl.utils.config import ModelConfig, TrainConfig
from batl.utils.data import as_float32_matrix
from batl.utils.reproducibility import set_seed

LOGGER = logging.getLogger(__name__)
_MAX_MATERIALIZED_EPOCH_BYTES = 1 * 1024**3


@dataclass(frozen=True)
class TrainingDiagnosticsConfig:
    """Opt-in per-cycle diagnostics for expensive benchmark builds."""

    queries: np.ndarray
    ground_truth: np.ndarray
    output_path: str | Path
    beam_size: int
    recall_k: int = 10
    num_return_leaves: int = 1
    max_queries: int | None = None
    max_loss_pairs: int = 100_000
    rerank_backend: Literal["numpy_cpu", "torch_gpu"] = "numpy_cpu"
    metric: str = "euclidean"


def _sample_training_labels(
    *,
    vectors: np.ndarray,
    config: TrainConfig,
    rng: np.random.Generator,
    seed: int,
    metric: str,
    log_label: str,
    profiler: StageProfiler,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample training queries and mine their neighbour labels once."""
    num_queries = _num_training_queries(vectors.shape[0])
    query_idx = rng.choice(vectors.shape[0], size=num_queries, replace=False)
    LOGGER.info(
        "%s: sampled %d training queries; neighbor subset=%d, top_k=%d, mining_seed=%d",
        log_label,
        num_queries,
        config.neighbor_search_subset,
        config.top_k_neighbors,
        seed,
    )
    neighbor_start = time.perf_counter()
    with profiler.stage("build.label_mining"):
        neighbor_ids = find_approximate_neighbors(
            queries=vectors[query_idx],
            database=vectors,
            top_k=config.top_k_neighbors,
            subset_size=config.neighbor_search_subset,
            seed=seed,
            mode=config.neighbor_search_mode,
            chunk_size=config.neighbor_search_chunk_size,
            metric=metric,
            backend=config.neighbor_search_backend,
        )
    LOGGER.info(
        "%s: neighbor labels ready in %.1fs; training pairs=%d",
        log_label,
        time.perf_counter() - neighbor_start,
        neighbor_ids.size,
    )
    _validate_neighbor_ids(neighbor_ids, n_vectors=vectors.shape[0])
    return query_idx, neighbor_ids


def alternating_train(
    model: BATLModel,
    vectors: np.ndarray,
    config: TrainConfig,
    model_cfg: ModelConfig,
    seed: int,
    assignment_mode: AssignmentMode = "round",
    assignment_order: AssignmentOrder = "input",
    metric: str = "euclidean",
    diagnostics: TrainingDiagnosticsConfig | None = None,
    profiler: StageProfiler | None = None,
) -> tuple[BATLModel, BATLTree]:
    """Train until convergence while alternating route training and tree updates."""
    stages = profiler or StageProfiler(enabled=False)
    set_seed(seed)
    vectors = as_float32_matrix(vectors, "vectors")
    if config.alternating_interval <= 0:
        raise ValueError("alternating_interval must be positive.")
    if config.num_epochs is not None and config.num_epochs <= 0:
        raise ValueError("num_epochs must be positive when provided.")
    if config.max_alternating_cycles is not None and config.max_alternating_cycles <= 0:
        raise ValueError("max_alternating_cycles must be positive when provided.")
    if config.convergence_patience < 0:
        raise ValueError("convergence_patience must be non-negative.")
    if config.convergence_min_delta < 0.0:
        raise ValueError("convergence_min_delta must be non-negative.")
    if config.neighbor_search_mode not in {"random_subset", "sequential_chunked"}:
        raise ValueError("neighbor_search_mode must be 'random_subset' or 'sequential_chunked'.")
    if config.neighbor_search_backend not in {"auto", "faiss_cpu", "faiss_gpu"}:
        raise ValueError("neighbor_search_backend must be 'auto', 'faiss_cpu', or 'faiss_gpu'.")
    if config.neighbor_search_chunk_size <= 0:
        raise ValueError("neighbor_search_chunk_size must be positive.")
    if assignment_mode not in {"round", "sequential"}:
        raise ValueError("assignment_mode must be 'round' or 'sequential'.")
    if assignment_order not in {"input", "confidence", "margin"}:
        raise ValueError("assignment_order must be 'input', 'confidence', or 'margin'.")

    device = torch.device(config.device)
    model.to(device)
    tree = BATLTree.random_init(
        N=vectors.shape[0],
        K=model_cfg.branching_factor,
        H=model_cfg.tree_height,
        alpha=model_cfg.alpha,
        seed=seed,
    )
    _validate_target_paths(tree.paths, branch_count=model.K, context="initial tree")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(seed)
    convergence_enabled = config.convergence_patience > 0
    max_cycles = _resolve_max_cycles(config)
    if max_cycles is None and not convergence_enabled:
        raise ValueError(
            "alternating_train would run without a stopping condition; "
            "set convergence_patience > 0 or provide max_alternating_cycles/num_epochs."
        )
    best_cycle_loss: float | None = None
    stale_cycles = 0
    LOGGER.info(
        "BATL training start: vectors=%d, dim=%d, K=%d, H=%d, "
        "alternating_interval=%d, max_cycles=%s, batch_size=%d, device=%s, convergence=%s, "
        "label_refresh=%s, torch_threads=%d",
        vectors.shape[0],
        vectors.shape[1],
        model_cfg.branching_factor,
        model_cfg.tree_height,
        config.alternating_interval,
        str(max_cycles) if max_cycles is not None else "until convergence",
        config.batch_size,
        device,
        f"patience={config.convergence_patience}, min_delta={config.convergence_min_delta}"
        if convergence_enabled
        else "disabled",
        config.label_refresh,
        torch.get_num_threads(),
    )

    fixed_labels: tuple[np.ndarray, np.ndarray] | None = None
    if config.label_refresh == "once":
        fixed_labels = _sample_training_labels(
            vectors=vectors,
            config=config,
            rng=rng,
            seed=seed + 1,
            metric=metric,
            log_label="fixed labels",
            profiler=stages,
        )

    cycle = 0
    while max_cycles is None or cycle < max_cycles:
        cycle += 1
        cycle_label = _cycle_label(cycle, max_cycles)
        cycle_start = time.perf_counter()
        if fixed_labels is None:
            query_idx, neighbor_ids = _sample_training_labels(
                vectors=vectors,
                config=config,
                rng=rng,
                seed=seed + cycle,
                metric=metric,
                log_label=cycle_label,
                profiler=stages,
            )
        else:
            query_idx, neighbor_ids = fixed_labels
        epoch_losses: list[float] = []
        for local_epoch in range(config.alternating_interval):
            epoch_start = time.perf_counter()
            with stages.stage("build.train_epoch"):
                mean_loss = _train_epoch(
                    model=model,
                    vectors=vectors,
                    query_idx=query_idx,
                    neighbor_ids=neighbor_ids,
                    tree_paths=tree.paths,
                    batch_size=config.batch_size,
                    device=device,
                    optimizer=optimizer,
                    rng=rng,
                )
            epoch_losses.append(mean_loss)
            if diagnostics is not None:
                _write_training_diagnostic(
                    diagnostics.output_path,
                    {
                        "cycle": cycle,
                        "stage": "model_train_epoch",
                        "local_epoch": local_epoch + 1,
                        "loss": mean_loss,
                    },
                )
            LOGGER.info(
                "%s model-train epoch %d/%d: loss=%.6f, elapsed=%.1fs",
                cycle_label,
                local_epoch + 1,
                config.alternating_interval,
                mean_loss,
                time.perf_counter() - epoch_start,
            )

        # Convergence is judged on the final epoch of the cycle (paper Fig. 4 shows
        # loss drops sharply right after each tree update; averaging across the
        # interval would smear that step and hide the converged state).
        cycle_final_loss = epoch_losses[-1] if epoch_losses else float("nan")
        if diagnostics is not None:
            _record_cycle_diagnostics(
                diagnostics=diagnostics,
                cycle=cycle,
                stage="after_model_training",
                model=model,
                tree=tree,
                vectors=vectors,
                query_idx=query_idx,
                neighbor_ids=neighbor_ids,
                batch_size=config.batch_size,
                device=device,
            )
        update_start = time.perf_counter()
        LOGGER.info("%s: updating balanced tree", cycle_label)
        use_cache = _resolve_tree_update_cache_embeddings(config.tree_update_cache_embeddings)
        tree_update_batch = _resolve_tree_update_batch_size(
            policy=config.tree_update_batch_size,
            num_heads=model_cfg.num_heads,
            device=device,
        )
        # Diagnostics are accumulated during the update either way; asking for
        # them only changes the return shape, and the chosen-rank histogram is
        # what sizes top-R assignment (SPEC_performance C2).
        tree, tree_diagnostics = update_tree(
            model=model,
            vectors=vectors,
            current_tree=tree,
            batch_size=tree_update_batch,
            device=device,
            assignment_mode=assignment_mode,
            assignment_order=assignment_order,
            cache_embeddings=use_cache,
            return_diagnostics=True,
            top_r=config.tree_update_top_r,
            profiler=stages,
        )
        _validate_target_paths(
            tree.paths, branch_count=model.K, context=f"{cycle_label} updated tree"
        )
        if diagnostics is not None:
            _record_cycle_diagnostics(
                diagnostics=diagnostics,
                cycle=cycle,
                stage="after_tree_update",
                model=model,
                tree=tree,
                vectors=vectors,
                query_idx=query_idx,
                neighbor_ids=neighbor_ids,
                batch_size=config.batch_size,
                device=device,
                tree_diagnostics=tree_diagnostics,
            )
        LOGGER.info(
            "%s done: tree_update=%.1fs, cycle_elapsed=%.1fs, leaf_gini=%.4f, cycle_loss=%.6f",
            cycle_label,
            time.perf_counter() - update_start,
            time.perf_counter() - cycle_start,
            tree.leaf_size_stats()["gini"],
            cycle_final_loss,
        )

        if not np.isfinite(cycle_final_loss):
            raise RuntimeError(f"{cycle_label} produced a non-finite training loss.")

        if convergence_enabled:
            if best_cycle_loss is None:
                best_cycle_loss = cycle_final_loss
                stale_cycles = 0
            else:
                relative_improvement = _relative_loss_improvement(best_cycle_loss, cycle_final_loss)
                if (
                    relative_improvement is not None
                    and relative_improvement >= config.convergence_min_delta
                ):
                    best_cycle_loss = cycle_final_loss
                    stale_cycles = 0
                else:
                    stale_cycles += 1
            LOGGER.info(
                "%s convergence: best_loss=%.6f, stale=%d/%d",
                cycle_label,
                best_cycle_loss if best_cycle_loss is not None else float("nan"),
                stale_cycles,
                config.convergence_patience,
            )
            if stale_cycles >= config.convergence_patience:
                LOGGER.info(
                    "Convergence reached after %s (loss plateaued).",
                    cycle_label,
                )
                break

    return model, tree


def _record_cycle_diagnostics(
    *,
    diagnostics: TrainingDiagnosticsConfig,
    cycle: int,
    stage: str,
    model: BATLModel,
    tree: BATLTree,
    vectors: np.ndarray,
    query_idx: np.ndarray,
    neighbor_ids: np.ndarray,
    batch_size: int,
    device: torch.device,
    tree_diagnostics: TreeUpdateDiagnostics | None = None,
) -> None:
    start = time.perf_counter()
    eval_loss = _evaluate_training_loss_sample(
        model=model,
        vectors=vectors,
        query_idx=query_idx,
        neighbor_ids=neighbor_ids,
        tree_paths=tree.paths,
        batch_size=batch_size,
        device=device,
        max_pairs=diagnostics.max_loss_pairs,
    )
    recall, mean_candidates = _top_bucket_recall(
        diagnostics=diagnostics,
        model=model,
        tree=tree,
        database=vectors,
    )
    row = {
        "cycle": cycle,
        "stage": stage,
        "eval_loss": eval_loss,
        f"recall@{diagnostics.recall_k}_top{diagnostics.num_return_leaves}_bucket": recall,
        "mean_top_bucket_candidates": mean_candidates,
        "loss_pairs": min(
            diagnostics.max_loss_pairs,
            int(query_idx.shape[0] * neighbor_ids.shape[1]),
        ),
        "diagnostic_queries": _diagnostic_query_count(diagnostics),
        "leaf_gini": tree.leaf_size_stats()["gini"],
        "elapsed_s": time.perf_counter() - start,
    }
    if tree_diagnostics is not None:
        row["assignment_order"] = tree_diagnostics.assignment_order
        row["tree_update_levels"] = tree_diagnostics.levels
    LOGGER.info(
        "cycle %d %s diagnostics: eval_loss=%.6f, recall=%.6f, elapsed=%.1fs",
        cycle,
        stage,
        eval_loss,
        recall,
        row["elapsed_s"],
    )
    _write_training_diagnostic(diagnostics.output_path, row)


def _top_bucket_recall(
    *,
    diagnostics: TrainingDiagnosticsConfig,
    model: BATLModel,
    tree: BATLTree,
    database: np.ndarray,
) -> tuple[float, float]:
    from batl.search import search_batch
    from batl.utils.metrics import recall_at_k

    query_count = _diagnostic_query_count(diagnostics)
    queries = diagnostics.queries[:query_count]
    ground_truth = diagnostics.ground_truth[:query_count]
    retrieved, n_candidates = search_batch(
        models=[model],
        trees=[tree],
        database=database,
        queries=queries,
        beam_size=diagnostics.beam_size,
        top_k=diagnostics.recall_k,
        return_candidate_counts=True,
        num_return_leaves=diagnostics.num_return_leaves,
        metric=diagnostics.metric,
        rerank_backend=diagnostics.rerank_backend,
    )
    per_query = recall_at_k(retrieved, ground_truth, diagnostics.recall_k)
    return float(np.mean(per_query)), float(np.mean(n_candidates))


def _diagnostic_query_count(diagnostics: TrainingDiagnosticsConfig) -> int:
    available = min(diagnostics.queries.shape[0], diagnostics.ground_truth.shape[0])
    if diagnostics.max_queries is None:
        return available
    return min(available, diagnostics.max_queries)


def _write_training_diagnostic(path: str | Path, row: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _evaluate_training_loss_sample(
    *,
    model: BATLModel,
    vectors: np.ndarray,
    query_idx: np.ndarray,
    neighbor_ids: np.ndarray,
    tree_paths: np.ndarray,
    batch_size: int,
    device: torch.device,
    max_pairs: int,
) -> float:
    """Evaluate CE loss on a deterministic sample of current training pairs."""
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive.")
    top_k = neighbor_ids.shape[1]
    n_pairs = query_idx.shape[0] * top_k
    if n_pairs == 0:
        return 0.0
    sample_size = min(max_pairs, n_pairs)
    if sample_size == n_pairs:
        pair_positions = np.arange(n_pairs, dtype=np.int64)
    else:
        pair_positions = np.linspace(0, n_pairs - 1, num=sample_size, dtype=np.int64)
    query_positions = pair_positions // top_k
    neighbor_positions = pair_positions % top_k

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_examples = 0
    try:
        with torch.inference_mode():
            for start in range(0, sample_size, batch_size):
                end = min(start + batch_size, sample_size)
                q_pos = query_positions[start:end]
                n_pos = neighbor_positions[start:end]
                x_np = np.ascontiguousarray(vectors[query_idx[q_pos]], dtype=np.float32)
                target_np = tree_paths[neighbor_ids[q_pos, n_pos]]
                _validate_target_paths(
                    target_np,
                    branch_count=model.K,
                    context="diagnostic training targets",
                )
                x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
                targets = torch.as_tensor(target_np, dtype=torch.long, device=device)
                loss = batl_loss(model(x, targets), targets)
                batch_count = x.shape[0]
                total_loss += float(loss.detach().cpu()) * batch_count
                total_examples += batch_count
    finally:
        if was_training:
            model.train()
    return total_loss / total_examples if total_examples else 0.0


def _train_epoch(
    model: BATLModel,
    vectors: np.ndarray,  # (N, d) — full database (read-only)
    query_idx: np.ndarray,  # (Q,)   — indices into vectors for this cycle's queries
    neighbor_ids: np.ndarray,  # (Q, top_k) — indices into vectors for each query's neighbours
    tree_paths: np.ndarray,  # (N, H) — current tree paths (uint16)
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
) -> float:
    """Training epoch over the Q*top_k (query, target-path) pairs.

    Shuffles the pair indices once on CPU, materializes the flattened (and
    pre-shuffled) epoch tensors directly on device, then iterates by
    contiguous slicing. Sequential slices are views — no per-batch gather
    kernel, no per-batch H2D copy. With BLISS-style 1% query sampling,
    top_k=100, d<=256 and H<=12, the pair tensors stay well under 2 GB
    through Deep10M; larger runs should narrow the query sample or top_k
    upstream.
    """
    model.train()
    top_k = neighbor_ids.shape[1]
    query_vectors = np.ascontiguousarray(vectors[query_idx], dtype=np.float32)
    n_pairs = query_vectors.shape[0] * top_k
    bytes_per_pair = (
        query_vectors.shape[1] * np.dtype(np.float32).itemsize
        + tree_paths.shape[1] * np.dtype(np.int64).itemsize
    )
    materialized_bytes = n_pairs * bytes_per_pair
    total_loss = 0.0
    total_examples = 0
    if materialized_bytes <= _MAX_MATERIALIZED_EPOCH_BYTES:
        total_loss, total_examples = _train_materialized_pair_block(
            model=model,
            query_vectors=query_vectors,
            neighbor_ids=neighbor_ids,
            tree_paths=tree_paths,
            pair_indices=rng.permutation(n_pairs),
            batch_size=batch_size,
            device=device,
            optimizer=optimizer,
        )
    else:
        query_order = rng.permutation(query_vectors.shape[0])
        queries_per_block = _queries_per_training_block(
            top_k=top_k,
            vector_dim=query_vectors.shape[1],
            path_height=tree_paths.shape[1],
        )
        for block_start in range(0, query_order.size, queries_per_block):
            block_query_indices = query_order[block_start : block_start + queries_per_block]
            block_pairs = block_query_indices.size * top_k
            block_loss, block_examples = _train_materialized_pair_block(
                model=model,
                query_vectors=query_vectors[block_query_indices],
                neighbor_ids=neighbor_ids[block_query_indices],
                tree_paths=tree_paths,
                pair_indices=rng.permutation(block_pairs),
                batch_size=batch_size,
                device=device,
                optimizer=optimizer,
            )
            total_loss += block_loss
            total_examples += block_examples
    return total_loss / total_examples if total_examples else 0.0


def _queries_per_training_block(top_k: int, vector_dim: int, path_height: int) -> int:
    bytes_per_query = top_k * (
        vector_dim * np.dtype(np.float32).itemsize + path_height * np.dtype(np.int64).itemsize
    )
    return max(1, _MAX_MATERIALIZED_EPOCH_BYTES // max(1, bytes_per_query))


def _train_materialized_pair_block(
    *,
    model: BATLModel,
    query_vectors: np.ndarray,
    neighbor_ids: np.ndarray,
    tree_paths: np.ndarray,
    pair_indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, int]:
    top_k = neighbor_ids.shape[1]
    epoch_x_np = np.repeat(query_vectors, top_k, axis=0)[pair_indices]
    epoch_targets_np = tree_paths[neighbor_ids.reshape(-1)][pair_indices]
    _validate_target_paths(
        epoch_targets_np,
        branch_count=model.K,
        context="materialized training targets",
    )
    epoch_x = torch.as_tensor(epoch_x_np, dtype=torch.float32, device=device)
    epoch_targets = torch.as_tensor(epoch_targets_np, dtype=torch.long, device=device)

    total_loss = 0.0
    total_examples = 0
    for start in range(0, epoch_x.shape[0], batch_size):
        end = start + batch_size
        x = epoch_x[start:end]
        targets = epoch_targets[start:end]
        optimizer.zero_grad(set_to_none=True)
        loss = batl_loss(model(x, targets), targets)
        loss.backward()
        optimizer.step()
        batch_count = x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_count
        total_examples += batch_count
    return total_loss, total_examples


def _validate_neighbor_ids(neighbor_ids: np.ndarray, *, n_vectors: int) -> None:
    if neighbor_ids.size == 0:
        return
    min_id = int(np.min(neighbor_ids))
    max_id = int(np.max(neighbor_ids))
    if min_id < 0 or max_id >= n_vectors:
        raise ValueError(
            "neighbor ids must contain database indices in [0, N); "
            f"N={n_vectors}, got min={min_id}, max={max_id}."
        )


def _validate_target_paths(paths: np.ndarray, *, branch_count: int, context: str) -> None:
    if paths.size == 0:
        return
    min_path = int(np.min(paths))
    max_path = int(np.max(paths))
    if min_path < 0 or max_path >= branch_count:
        invalid_entries = int(np.count_nonzero((paths < 0) | (paths >= branch_count)))
        raise ValueError(
            f"{context} paths must contain branch IDs in [0, K); "
            f"K={branch_count}, got min={min_path}, max={max_path}, "
            f"invalid_entries={invalid_entries}."
        )


def _resolve_max_cycles(config: TrainConfig) -> int | None:
    if config.max_alternating_cycles is not None:
        return config.max_alternating_cycles
    if config.num_epochs is None:
        return None
    return math.ceil(config.num_epochs / config.alternating_interval)


def _cycle_label(cycle: int, max_cycles: int | None) -> str:
    if max_cycles is None:
        return f"cycle {cycle}"
    return f"cycle {cycle}/{max_cycles}"


def _num_training_queries(n_vectors: int) -> int:
    """Return the BLISS-style 1% training-query sample size."""
    if n_vectors <= 0:
        raise ValueError("n_vectors must be positive.")
    return max(1, int(n_vectors * DEFAULT_TRAINING_QUERY_FRACTION))


def _relative_loss_improvement(best_loss: float, current_loss: float) -> float | None:
    """Return relative loss improvement, or None when a zero baseline cannot improve."""
    if best_loss == 0.0:
        return None
    return (best_loss - current_loss) / abs(best_loss)


# All names imported at the top of this file remain accessible as attributes of
# batl.training for backward compatibility (build.py, search.py, index_parsing.py,
# and tests that import AssignmentOrder or private helpers from here).
