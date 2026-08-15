"""Algorithm 1: balanced tree update, decoder infrastructure, and assignment kernel."""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
import torch
from torch import Tensor

from batl.constants import (
    ASSIGNMENT_ARGSORT_CHUNK_ROWS,
    RANK_HISTOGRAM_EDGES,
    STRAGGLER_COVERAGE_TARGET,
)
from batl.model import BATLModel
from batl.profiling import StageProfiler
from batl.tree import BATLTree
from batl.utils.config import TreeUpdateBatchSize
from batl.utils.data import as_float32_matrix

LOGGER = logging.getLogger(__name__)

AssignmentOrder = Literal["input", "confidence", "margin"]
AssignmentMode = Literal["round", "sequential"]


def _rank_histogram_labels() -> tuple[str, ...]:
    labels = []
    for position, low in enumerate(RANK_HISTOGRAM_EDGES):
        if position + 1 == len(RANK_HISTOGRAM_EDGES):
            labels.append(f"rank_{low}_plus")
            continue
        high = RANK_HISTOGRAM_EDGES[position + 1] - 1
        labels.append(f"rank_{low}" if high == low else f"rank_{low}_{high}")
    return tuple(labels)


_RANK_HISTOGRAM_LABELS = _rank_histogram_labels()


def _rank_histogram_counts(ranks: np.ndarray) -> np.ndarray:
    """Bucket non-negative chosen ranks into ``RANK_HISTOGRAM_EDGES``."""
    edges = np.asarray(RANK_HISTOGRAM_EDGES, dtype=np.int64)
    buckets = np.searchsorted(edges, ranks, side="right") - 1
    return np.bincount(buckets, minlength=edges.size).astype(np.int64)


def _min_top_r_covering(histogram: np.ndarray, num_vectors: int) -> int:
    """Return the smallest bucket edge covering ``STRAGGLER_COVERAGE_TARGET``.

    This is the R a top-R assignment would need for stragglers — vectors whose
    top-R branches are all full — to stay under 0.1%. Resolution is limited to
    the bucket edges, which is the granularity R would be set at anyway. Returns
    the last edge when even that does not reach the target.
    """
    if num_vectors <= 0:
        return 0
    covered = 0
    for position, count in enumerate(histogram):
        covered += int(count)
        if covered / num_vectors >= STRAGGLER_COVERAGE_TARGET:
            # Ranks in this bucket are < the next edge, so R must reach it.
            if position + 1 < len(RANK_HISTOGRAM_EDGES):
                return RANK_HISTOGRAM_EDGES[position + 1]
            break
    return RANK_HISTOGRAM_EDGES[-1]


@dataclass(frozen=True)
class TreeUpdateDiagnostics:
    """Per-level diagnostics for one balanced tree update."""

    assignment_mode: AssignmentMode
    assignment_order: AssignmentOrder
    levels: list[dict[str, float | int]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@overload
def update_tree(
    model: BATLModel,
    vectors: np.ndarray,
    current_tree: BATLTree,
    batch_size: int,
    device: torch.device,
) -> BATLTree: ...


@overload
def update_tree(
    model: BATLModel,
    vectors: np.ndarray,
    current_tree: BATLTree,
    batch_size: int,
    device: torch.device,
    *,
    assignment_mode: AssignmentMode = "round",
    assignment_order: AssignmentOrder = "input",
    cache_embeddings: bool = False,
    top_r: int | None = None,
    profiler: StageProfiler | None = None,
) -> BATLTree: ...


@overload
def update_tree(
    model: BATLModel,
    vectors: np.ndarray,
    current_tree: BATLTree,
    batch_size: int,
    device: torch.device,
    *,
    assignment_mode: AssignmentMode = "round",
    assignment_order: AssignmentOrder = "input",
    cache_embeddings: bool = False,
    return_diagnostics: Literal[True],
    top_r: int | None = None,
    profiler: StageProfiler | None = None,
) -> tuple[BATLTree, TreeUpdateDiagnostics]: ...


def update_tree(
    model: BATLModel,
    vectors: np.ndarray,
    current_tree: BATLTree,
    batch_size: int,
    device: torch.device,
    *,
    assignment_mode: AssignmentMode = "round",
    assignment_order: AssignmentOrder = "input",
    cache_embeddings: bool = False,
    return_diagnostics: bool = False,
    top_r: int | None = None,
    profiler: StageProfiler | None = None,
) -> BATLTree | tuple[BATLTree, TreeUpdateDiagnostics]:
    """Reassign all database vectors with BATL Algorithm 1."""
    stages = profiler or StageProfiler(enabled=False)
    vectors = as_float32_matrix(vectors, "vectors")
    if vectors.shape[0] != current_tree.N:
        raise ValueError("vectors row count must match current_tree.N.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if assignment_mode not in {"round", "sequential"}:
        raise ValueError("assignment_mode must be 'round' or 'sequential'.")
    if assignment_order not in {"input", "confidence", "margin"}:
        raise ValueError("assignment_order must be 'input', 'confidence', or 'margin'.")

    K = current_tree.K
    H = current_tree.H
    N = current_tree.N
    if assignment_mode == "sequential" and top_r is not None and top_r < K:
        raise ValueError("sequential assignment requires a full-K branch ordering.")

    device = torch.device(device)
    model.to(device)
    with stages.stage("build.tree_update.encode_all"):
        cached_embeddings = (
            _encode_all_vectors(model=model, vectors=vectors, batch_size=batch_size, device=device)
            if cache_embeddings
            else None
        )
    LOGGER.info(
        "tree update embedding cache: %s",
        "enabled" if cached_embeddings is not None else "disabled",
    )

    # None means full K: identical work to the pre-top-R path, zero stragglers.
    resolved_top_r = K if top_r is None else min(max(2, top_r), K)
    new_paths = np.zeros((N, H), dtype=np.uint16)
    current_groups: dict[tuple[int, ...], np.ndarray] = {
        (): np.arange(N, dtype=np.int64),
    }
    diagnostics = _TreeUpdateStats(
        assignment_order=assignment_order,
        assignment_mode=assignment_mode,
    )

    for h in range(H):
        level_start = time.perf_counter()
        capacity = max(1, math.ceil(current_tree.alpha * N / (K ** (h + 1))))
        next_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        LOGGER.info(
            "tree update level %d/%d: groups=%d, capacity=%d",
            h + 1,
            H,
            len(current_groups),
            capacity,
        )

        for path_prefix, vec_indices in current_groups.items():
            with stages.stage("build.tree_update.decode"):
                top_probs, top_branches = _decode_group_topr(
                    model=model,
                    vectors=vectors,
                    vec_indices=vec_indices,
                    path_prefix=path_prefix,
                    batch_size=batch_size,
                    device=device,
                    top_r=resolved_top_r,
                    cached_embeddings=cached_embeddings,
                )

            def resolve_full(
                local_positions: np.ndarray,
                _vec_indices: np.ndarray = vec_indices,
                _path_prefix: tuple[int, ...] = path_prefix,
            ) -> np.ndarray:
                """Full-K ordering for the few vectors that outran top-R.

                Chunked by the same batch size as the main decode: resolving
                every straggler as one (m, K) matrix would recreate the peak
                this change exists to remove.
                """
                return _decode_group_full_order(
                    model=model,
                    vectors=vectors,
                    vec_indices=_vec_indices[local_positions],
                    path_prefix=_path_prefix,
                    batch_size=batch_size,
                    device=device,
                    cached_embeddings=cached_embeddings,
                )

            with stages.stage("build.tree_update.assign"):
                (
                    local_branches,
                    local_ranks,
                    local_fallbacks,
                    local_probs,
                    local_stragglers,
                ) = (
                    _assign_balanced_sequential(
                        top_probs,
                        top_branches,
                        K,
                        capacity,
                        assignment_order,
                    )
                    if assignment_mode == "sequential"
                    else _assign_balanced_topr(
                        top_probs,
                        top_branches,
                        K,
                        capacity,
                        assignment_order,
                        resolve_full,
                    )
                )
            if local_stragglers:
                LOGGER.info(
                    "tree update level %d/%d: %d/%d vectors needed a full-K re-decode "
                    "(top_r=%d); raise training.tree_update_top_r if this is not rare",
                    h + 1,
                    H,
                    local_stragglers,
                    vec_indices.size,
                    resolved_top_r,
                )
            new_paths[vec_indices, h] = local_branches
            with stages.stage("build.tree_update.regroup"):
                # Group vec_indices by assigned branch using sort+split (O(N log N))
                # instead of one boolean mask per branch (O(N x K)).
                for b_int, grp in _regroup_by_branch(vec_indices, local_branches):
                    next_groups[(*path_prefix, b_int)].extend(grp.tolist())
            diagnostics.record_batch(
                level=h,
                chosen_ranks=local_ranks,
                used_fallback=local_fallbacks,
                assignment_probs=local_probs,
                straggler_count=local_stragglers,
            )

        current_groups = {
            prefix: np.asarray(indices, dtype=np.int64) for prefix, indices in next_groups.items()
        }
        LOGGER.info(
            "tree update level %d/%d done: next_groups=%d, elapsed=%.1fs",
            h + 1,
            H,
            len(current_groups),
            time.perf_counter() - level_start,
        )

    updated = BATLTree(K=K, H=H, alpha=current_tree.alpha, N=N, paths=new_paths)
    if return_diagnostics:
        return updated, diagnostics.to_diagnostics()
    return updated


# ---------------------------------------------------------------------------
# Tree-update knob resolution (explicit user knobs + one CUDA safety guard)
# ---------------------------------------------------------------------------


def _cuda_attention_batch_guard(device: torch.device, num_heads: int) -> int:
    """Return a conservative CUDA attention batch guard for tree updates.

    PyTorch owns scaled-dot-product-attention backend selection. This guard is
    only here to avoid feeding CUDA attention kernels an effective
    batch-by-heads work dimension beyond the CUDA grid limit seen by
    flash-style implementations.
    """
    if device.type != "cuda":
        return 2**31 - 1
    return max(1, 65535 // max(1, num_heads))


def _resolve_tree_update_batch_size(
    policy: TreeUpdateBatchSize | None,
    num_heads: int,
    device: torch.device,
) -> int:
    """Return the tree-update batch size, clamped by the CUDA attention guard.

    Explicit user knob. Pass an integer to set the batch directly. With ``None``
    or ``"auto"``, defaults to the CUDA attention guard on CUDA (the largest
    documented per-launch ceiling, ``65535 // num_heads``) and to 4096 on CPU.
    Explicit integers exceeding the guard on CUDA are clamped down with a WARN
    log so the user sees the change.
    """
    guard = _cuda_attention_batch_guard(device, num_heads)
    if policy is None or policy == "auto":
        return guard if device.type == "cuda" else 4096
    if isinstance(policy, int):
        if device.type == "cuda" and policy > guard:
            LOGGER.warning(
                "capping explicit CUDA tree_update_batch_size from %d to %d (num_heads=%d)",
                policy,
                guard,
                num_heads,
            )
            return guard
        return policy
    raise ValueError("tree_update_batch_size must be 'auto', None, or an integer.")


def _resolve_tree_update_cache_embeddings(policy: bool | str) -> bool:
    """Return whether to pre-encode database vectors per tree update.

    Explicit user knob — ``True``/``False`` is honoured directly. ``"auto"``
    defaults to ``False`` because the cache is a per-cycle buffer of
    ``n_vectors x embed_dim x float32`` bytes (e.g. ~10 GB at Deep10M /
    embed_dim=256, ~100 GB at Deep100M), which causes GPU memory pressure on
    any commodity single-GPU setup. Set ``True`` explicitly when free VRAM
    comfortably exceeds the cache footprint.
    """
    if isinstance(policy, bool):
        return policy
    if policy == "auto":
        return False
    raise ValueError("tree_update_cache_embeddings must be True, False, or 'auto'.")


# ---------------------------------------------------------------------------
# Decoder infrastructure (bridges model <-> tree update)
# ---------------------------------------------------------------------------


def _decode_group_topr(
    model: BATLModel,
    vectors: np.ndarray,
    vec_indices: np.ndarray,
    path_prefix: tuple[int, ...],
    batch_size: int,
    device: torch.device,
    top_r: int,
    cached_embeddings: Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a group and keep only each vector's ``top_r`` best branches.

    Ordering is descending probability, ties by ascending branch id, matching
    ``_sort_branches_by_probability``. A stable full sort on-device is used
    rather than ``torch.topk`` because topk gives no tie guarantee, and
    reproducing the reference ordering exactly is worth more here than the
    marginal on-device saving — the cost this removes is the ``(n, K)`` host
    transfer and CPU argsort, both of which go away either way.

    Returns ``(top_probs (n, R) float32, top_branches (n, R) branch-id dtype)``.
    """
    K = model.K
    top_r = min(max(2, top_r), K)
    branch_dtype = _branch_order_dtype(K)
    top_probs = np.empty((vec_indices.size, top_r), dtype=np.float32)
    top_branches = np.empty((vec_indices.size, top_r), dtype=branch_dtype)
    cursor = 0
    for start in range(0, vec_indices.size, batch_size):
        batch_indices = vec_indices[start : start + batch_size]
        if cached_embeddings is None:
            batch_vectors = torch.as_tensor(
                vectors[batch_indices],
                dtype=torch.float32,
                device=device,
            )
            probs = model.decode_node_probs(batch_vectors, path_prefix)
        else:
            index = torch.as_tensor(batch_indices, dtype=torch.long, device=device)
            probs = model.decode_node_probs_from_embeddings(
                cached_embeddings.index_select(0, index),
                path_prefix,
            )
        values, indices = torch.sort(probs, dim=1, descending=True, stable=True)
        batch_end = cursor + int(values.shape[0])
        top_probs[cursor:batch_end] = values[:, :top_r].detach().cpu().numpy()
        top_branches[cursor:batch_end] = indices[:, :top_r].detach().cpu().numpy()
        cursor = batch_end
    return top_probs, top_branches


def _decode_group_full_order(
    model: BATLModel,
    vectors: np.ndarray,
    vec_indices: np.ndarray,
    path_prefix: tuple[int, ...],
    batch_size: int,
    device: torch.device,
    cached_embeddings: Tensor | None = None,
) -> np.ndarray:
    """Full-K descending branch ordering, chunked, for straggler vectors."""
    K = model.K
    orders = np.empty((vec_indices.size, K), dtype=_branch_order_dtype(K))
    cursor = 0
    for start in range(0, vec_indices.size, batch_size):
        batch_indices = vec_indices[start : start + batch_size]
        if cached_embeddings is None:
            batch_vectors = torch.as_tensor(
                vectors[batch_indices],
                dtype=torch.float32,
                device=device,
            )
            probs = model.decode_node_probs(batch_vectors, path_prefix)
        else:
            index = torch.as_tensor(batch_indices, dtype=torch.long, device=device)
            probs = model.decode_node_probs_from_embeddings(
                cached_embeddings.index_select(0, index),
                path_prefix,
            )
        _values, indices = torch.sort(probs, dim=1, descending=True, stable=True)
        batch_end = cursor + int(indices.shape[0])
        orders[cursor:batch_end] = indices.detach().cpu().numpy()
        cursor = batch_end
    return orders


def _encode_all_vectors(
    model: BATLModel,
    vectors: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Encode all database vectors once for one tree update."""
    batches: list[Tensor] = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, vectors.shape[0], batch_size):
                batch = torch.as_tensor(
                    vectors[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                batches.append(model.encode(batch).detach())
    finally:
        if was_training:
            model.train()
    return torch.cat(batches, dim=0)


# ---------------------------------------------------------------------------
# Assignment kernel (Algorithm 1 inner loop) — CPU / NumPy
# ---------------------------------------------------------------------------


def _assign_balanced_vectorized(
    probs: np.ndarray,
    K: int,
    capacity: int,
    assignment_order: AssignmentOrder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Round-based vectorized balanced assignment for one node's vector group.

    Each round all unassigned vectors simultaneously try their current
    preferred branch. Within each branch, candidates are sorted by priority
    score and the top ``capacity`` accepted; the rest retry with their next
    preferred branch next round. Repeats until all vectors are placed.

    Not bit-for-bit identical to the sequential per-vector greedy (within-round
    tie resolution order differs), but preserves the capacity invariant and
    produces equivalent balance quality.

    Returns:
        branches     — int32 (N,): assigned branch per local position
        assigned_ranks — int32 (N,): rank of assigned branch (0 = top choice)
        used_fallback  — bool (N,): True if argmin fallback was triggered
    """
    N, K_from_probs = probs.shape
    if K_from_probs != K:
        raise ValueError("probs must have K columns.")
    if assignment_order == "confidence":
        scores = probs.max(axis=1)
    elif assignment_order == "margin":
        scores = _assignment_margins(probs)
    else:  # "input": earlier index has higher priority
        scores = -np.arange(N, dtype=np.float64)
    sorted_branches = _sort_branches_by_probability(probs, K)

    branches = np.empty(N, dtype=np.int32)
    counts = np.zeros(K, dtype=np.int64)
    rank_pointer = np.zeros(N, dtype=np.int32)
    assigned_ranks = np.zeros(N, dtype=np.int32)
    used_fallback = np.zeros(N, dtype=bool)
    unassigned = np.arange(N, dtype=np.int64)

    while unassigned.size > 0:
        current_prefs = sorted_branches[unassigned, rank_pointer[unassigned]]

        # Group by preferred branch, highest-priority candidates first within each group.
        sort_order = np.lexsort((-scores[unassigned], current_prefs))
        sorted_u = unassigned[sort_order]
        sorted_prefs = current_prefs[sort_order]

        changes = np.flatnonzero(np.diff(sorted_prefs)) + 1
        boundaries = np.r_[0, changes, len(sorted_u)]

        retry_parts: list[np.ndarray] = []
        for i in range(len(boundaries) - 1):
            s, e = int(boundaries[i]), int(boundaries[i + 1])
            b = int(sorted_prefs[s])
            candidates = sorted_u[s:e]
            available = int(capacity - counts[b])
            if available >= len(candidates):
                branches[candidates] = b
                assigned_ranks[candidates] = rank_pointer[candidates]
                counts[b] += len(candidates)
            elif available > 0:
                accepted, rejected = candidates[:available], candidates[available:]
                branches[accepted] = b
                assigned_ranks[accepted] = rank_pointer[accepted]
                counts[b] = capacity
                rank_pointer[rejected] += 1
                retry_parts.append(rejected)
            else:
                rank_pointer[candidates] += 1
                retry_parts.append(candidates)

        if not retry_parts:
            break
        retry = np.concatenate(retry_parts)

        # Fallback: all K branches exhausted — assign to least-filled branch.
        fb_mask = rank_pointer[retry] >= K
        if fb_mask.any():
            fb_idx = retry[fb_mask]
            least_full = int(np.argmin(counts))
            branches[fb_idx] = least_full
            assigned_ranks[fb_idx] = K  # sentinel: beyond all normal ranks
            used_fallback[fb_idx] = True
            counts[least_full] += int(fb_mask.sum())
            retry = retry[~fb_mask]

        unassigned = retry

    return branches, assigned_ranks, used_fallback


def _assign_balanced_topr(
    top_probs: np.ndarray,
    top_branches: np.ndarray,
    K: int,
    capacity: int,
    assignment_order: AssignmentOrder,
    resolve_full: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Balanced assignment reading only the best ``R`` branches per vector.

    Equivalent to ``_assign_balanced_vectorized`` given the same probabilities:
    the round loop consults ranks in strictly increasing order, so truncating
    the ordering at R is invisible until a vector actually needs rank R, and
    such stragglers get their full ordering from ``resolve_full``.

    ``resolve_full(local_positions) -> (m, K)`` returns the full descending
    branch ordering for those rows, ordered like
    ``_sort_branches_by_probability``.

    Returns ``(branches, assigned_ranks, used_fallback, assignment_probs,
    straggler_count)``.
    """
    N, R = top_probs.shape
    if top_branches.shape != top_probs.shape:
        raise ValueError("top_probs and top_branches must have the same shape.")
    if R < 2:
        raise ValueError("top-R assignment needs R >= 2 for margin scores.")
    if R > K:
        raise ValueError("R cannot exceed K.")

    if assignment_order == "confidence":
        scores = top_probs[:, 0].astype(np.float64, copy=False)
    elif assignment_order == "margin":
        scores = (top_probs[:, 0] - top_probs[:, 1]).astype(np.float64, copy=False)
    else:  # "input": earlier index has higher priority
        scores = -np.arange(N, dtype=np.float64)

    branches = np.empty(N, dtype=np.int32)
    counts = np.zeros(K, dtype=np.int64)
    rank_pointer = np.zeros(N, dtype=np.int32)
    assigned_ranks = np.zeros(N, dtype=np.int32)
    used_fallback = np.zeros(N, dtype=bool)
    unassigned = np.arange(N, dtype=np.int64)

    # Full orderings for stragglers only, packed into a dense (m, K) block so
    # preference lookup stays vectorized. -1 means "still within top-R".
    straggler_slot = np.full(N, -1, dtype=np.int64)
    straggler_orders = np.empty((0, K), dtype=top_branches.dtype)

    while unassigned.size > 0:
        pointer = rank_pointer[unassigned]
        current_prefs = np.empty(unassigned.size, dtype=np.int64)
        within_topr = pointer < R
        if within_topr.any():
            rows = unassigned[within_topr]
            current_prefs[within_topr] = top_branches[rows, pointer[within_topr]]
        if not within_topr.all():
            rows = unassigned[~within_topr]
            current_prefs[~within_topr] = straggler_orders[
                straggler_slot[rows], pointer[~within_topr]
            ]

        # Group by preferred branch, highest-priority candidates first within each group.
        sort_order = np.lexsort((-scores[unassigned], current_prefs))
        sorted_u = unassigned[sort_order]
        sorted_prefs = current_prefs[sort_order]

        changes = np.flatnonzero(np.diff(sorted_prefs)) + 1
        boundaries = np.r_[0, changes, len(sorted_u)]

        retry_parts: list[np.ndarray] = []
        for i in range(len(boundaries) - 1):
            s, e = int(boundaries[i]), int(boundaries[i + 1])
            b = int(sorted_prefs[s])
            candidates = sorted_u[s:e]
            available = int(capacity - counts[b])
            if available >= len(candidates):
                branches[candidates] = b
                assigned_ranks[candidates] = rank_pointer[candidates]
                counts[b] += len(candidates)
            elif available > 0:
                accepted, rejected = candidates[:available], candidates[available:]
                branches[accepted] = b
                assigned_ranks[accepted] = rank_pointer[accepted]
                counts[b] = capacity
                rank_pointer[rejected] += 1
                retry_parts.append(rejected)
            else:
                rank_pointer[candidates] += 1
                retry_parts.append(candidates)

        if not retry_parts:
            break
        retry = np.concatenate(retry_parts)

        # Vectors that just ran past the truncated ordering need the full one.
        needs_full = (rank_pointer[retry] >= R) & (straggler_slot[retry] < 0)
        if needs_full.any():
            new_stragglers = retry[needs_full]
            resolved = resolve_full(new_stragglers)
            if resolved.shape != (new_stragglers.size, K):
                raise ValueError("resolve_full must return one full-K ordering per position.")
            straggler_slot[new_stragglers] = straggler_orders.shape[0] + np.arange(
                new_stragglers.size, dtype=np.int64
            )
            straggler_orders = np.concatenate(
                [straggler_orders, resolved.astype(top_branches.dtype, copy=False)]
            )

        # Fallback: all K branches exhausted — assign to least-filled branch.
        fb_mask = rank_pointer[retry] >= K
        if fb_mask.any():
            fb_idx = retry[fb_mask]
            least_full = int(np.argmin(counts))
            branches[fb_idx] = least_full
            assigned_ranks[fb_idx] = K  # sentinel: beyond all normal ranks
            used_fallback[fb_idx] = True
            counts[least_full] += int(fb_mask.sum())
            retry = retry[~fb_mask]

        unassigned = retry

    assignment_probs = _assignment_probabilities(
        top_probs=top_probs,
        assigned_ranks=assigned_ranks,
        used_fallback=used_fallback,
    )
    return (
        branches,
        assigned_ranks,
        used_fallback,
        assignment_probs,
        int((straggler_slot >= 0).sum()),
    )


def _assign_balanced_sequential(
    ranked_probs: np.ndarray,
    branch_orders: np.ndarray,
    K: int,
    capacity: int,
    assignment_order: AssignmentOrder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Assign one vector completely before visiting the next (Algorithm 1)."""
    N, retained = ranked_probs.shape
    if branch_orders.shape != ranked_probs.shape:
        raise ValueError("ranked_probs and branch_orders must have the same shape.")
    if retained != K:
        raise ValueError("sequential assignment requires the full-K branch ordering.")

    visit_order = _assignment_order(ranked_probs, assignment_order)
    counts = np.zeros(K, dtype=np.int64)
    branches = np.empty(N, dtype=np.int32)
    assigned_ranks = np.empty(N, dtype=np.int32)
    used_fallback = np.zeros(N, dtype=bool)
    assignment_probs = np.full(N, np.nan, dtype=np.float64)

    for position in visit_order:
        branch, rank, _denied_top, fallback = _first_available_branch(
            branch_orders[position], counts, capacity
        )
        branches[position] = branch
        assigned_ranks[position] = K if fallback else rank
        used_fallback[position] = fallback
        if not fallback:
            assignment_probs[position] = ranked_probs[position, rank]
        counts[branch] += 1

    return branches, assigned_ranks, used_fallback, assignment_probs, 0


def _assignment_probabilities(
    *,
    top_probs: np.ndarray,
    assigned_ranks: np.ndarray,
    used_fallback: np.ndarray,
) -> np.ndarray:
    """Probability of each vector's assigned branch, for the confidence mean.

    Only ranks inside the retained top-R window have a probability on hand.
    Stragglers and fallbacks are excluded rather than guessed, and the caller
    reports how many rows contributed so the mean stays interpretable.
    """
    R = top_probs.shape[1]
    known = (~used_fallback) & (assigned_ranks < R)
    probs = np.zeros(assigned_ranks.size, dtype=np.float64)
    rows = np.flatnonzero(known)
    probs[rows] = top_probs[rows, assigned_ranks[rows]]
    return np.where(known, probs, np.nan)


def _assignment_margins(probs: np.ndarray) -> np.ndarray:
    margins = np.empty(probs.shape[0], dtype=probs.dtype)
    for start in range(0, probs.shape[0], ASSIGNMENT_ARGSORT_CHUNK_ROWS):
        end = min(start + ASSIGNMENT_ARGSORT_CHUNK_ROWS, probs.shape[0])
        chunk = probs[start:end]
        if chunk.shape[1] < 2:
            margins[start:end] = chunk[:, 0]
        else:
            top2 = np.partition(chunk, kth=chunk.shape[1] - 2, axis=1)[:, -2:]
            margins[start:end] = top2.max(axis=1) - top2.min(axis=1)
    return margins


def _sort_branches_by_probability(probs: np.ndarray, K: int) -> np.ndarray:
    """Order branches by descending probability, ties by ascending branch id.

    ``kind="stable"`` is what makes the tie rule statable at all. The default
    quicksort ordered exact ties arbitrarily, so the top-R path had nothing
    well-defined to reproduce; with a stable sort both paths agree on every
    input, including tied rows.
    """
    sorted_branches = np.empty((probs.shape[0], K), dtype=_branch_order_dtype(K))
    for start in range(0, probs.shape[0], ASSIGNMENT_ARGSORT_CHUNK_ROWS):
        end = min(start + ASSIGNMENT_ARGSORT_CHUNK_ROWS, probs.shape[0])
        sorted_branches[start:end] = np.argsort(-probs[start:end], axis=1, kind="stable")
    return sorted_branches


def _branch_order_dtype(K: int) -> np.dtype:
    if K <= np.iinfo(np.uint8).max + 1:
        return np.dtype(np.uint8)
    if K <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    return np.dtype(np.int32)


def _regroup_by_branch(
    vec_indices: np.ndarray,
    local_branches: np.ndarray,
) -> list[tuple[int, np.ndarray]]:
    """Group vectors by branch while preserving their relative input order."""
    if vec_indices.shape != local_branches.shape:
        raise ValueError("vec_indices and local_branches must have the same shape.")
    if vec_indices.size == 0:
        return []
    sort_idx = np.argsort(local_branches, kind="stable")
    sorted_branches = local_branches[sort_idx]
    sorted_indices = vec_indices[sort_idx]
    changes = np.flatnonzero(np.diff(sorted_branches)) + 1
    boundary_idx = np.r_[0, changes]
    return [
        (int(branch), group)
        for branch, group in zip(
            sorted_branches[boundary_idx],
            np.split(sorted_indices, changes),
            strict=True,
        )
    ]


# ---------------------------------------------------------------------------
# Reference implementations (kept for equivalence tests, not used in prod)
# ---------------------------------------------------------------------------


def _assignment_order(probs: np.ndarray, assignment_order: AssignmentOrder) -> np.ndarray:
    if assignment_order == "input":
        return np.arange(probs.shape[0], dtype=np.int64)
    if assignment_order == "confidence":
        scores = probs.max(axis=1)
    else:
        sorted_probs = np.sort(probs, axis=1)
        scores = sorted_probs[:, -1] - sorted_probs[:, -2]
    return np.argsort(-scores, kind="mergesort").astype(np.int64, copy=False)


def _first_available_branch(
    branch_order: np.ndarray,
    counts: np.ndarray,
    capacity: int,
) -> tuple[int, int, bool, bool]:
    top_branch = int(branch_order[0])
    for rank, branch in enumerate(branch_order):
        branch_id = int(branch)
        if counts[branch_id] < capacity:
            return branch_id, rank, rank > 0 and counts[top_branch] >= capacity, False
    branch_id = int(np.argmin(counts))
    fallback_rank = int(np.flatnonzero(branch_order == branch_id)[0])
    return branch_id, fallback_rank, branch_id != top_branch, True


# ---------------------------------------------------------------------------
# Diagnostics accumulator
# ---------------------------------------------------------------------------


class _TreeUpdateStats:
    def __init__(
        self,
        assignment_order: AssignmentOrder,
        assignment_mode: AssignmentMode = "round",
    ) -> None:
        self.assignment_mode: AssignmentMode = assignment_mode
        self.assignment_order: AssignmentOrder = assignment_order
        self._levels: dict[int, dict[str, float | int]] = defaultdict(
            lambda: {
                "num_vectors": 0,
                "rank_sum": 0.0,
                "second_choice_count": 0,
                "third_or_later_choice_count": 0,
                "denied_top_count": 0,
                "fallback_count": 0,
                "straggler_count": 0,
                "confidence_sum": 0.0,
                "confidence_count": 0,
            }
        )
        # Fallback vectors carry the sentinel rank K and are excluded here;
        # they are already counted by fallback_count and would otherwise look
        # like a genuine rank-K choice.
        self._rank_histograms: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(len(RANK_HISTOGRAM_EDGES), dtype=np.int64)
        )
        self._max_chosen_rank: dict[int, int] = defaultdict(int)

    def _record_ranks(self, level: int, ranks: np.ndarray) -> None:
        if ranks.size == 0:
            return
        self._rank_histograms[level] += _rank_histogram_counts(ranks)
        self._max_chosen_rank[level] = max(
            self._max_chosen_rank[level],
            int(ranks.max()),
        )

    def record_batch(
        self,
        *,
        level: int,
        chosen_ranks: np.ndarray,
        used_fallback: np.ndarray,
        assignment_probs: np.ndarray,
        straggler_count: int = 0,
    ) -> None:
        """Vectorized equivalent of record() for the full assignment result of one group.

        ``assignment_probs`` holds the probability of each vector's assigned
        branch, or NaN where the top-R window did not retain it. NaN rows are
        left out of the confidence mean instead of being counted as zero.
        """
        stats = self._levels[level]
        N = int(chosen_ranks.size)
        stats["num_vectors"] += N
        stats["rank_sum"] += float(chosen_ranks.sum())
        stats["second_choice_count"] += int((chosen_ranks == 1).sum())
        stats["third_or_later_choice_count"] += int((chosen_ranks >= 2).sum())
        stats["denied_top_count"] += int((chosen_ranks > 0).sum())
        stats["fallback_count"] += int(used_fallback.sum())
        stats["straggler_count"] += int(straggler_count)
        scored = ~np.isnan(assignment_probs)
        stats["confidence_sum"] += float(assignment_probs[scored].sum())
        stats["confidence_count"] += int(scored.sum())
        self._record_ranks(level, chosen_ranks[~used_fallback].astype(np.int64, copy=False))

    def to_diagnostics(self) -> TreeUpdateDiagnostics:
        levels = []
        for level in sorted(self._levels):
            raw = self._levels[level]
            num_vectors = int(raw["num_vectors"])
            denominator = num_vectors if num_vectors else 1
            histogram = self._rank_histograms.get(
                level, np.zeros(len(RANK_HISTOGRAM_EDGES), dtype=np.int64)
            )
            ranked_vectors = int(histogram.sum())
            rank_buckets: dict[str, float | int] = {
                f"rank_hist_{label}": int(count)
                for label, count in zip(_RANK_HISTOGRAM_LABELS, histogram, strict=True)
            }
            levels.append(
                {
                    "level": level,
                    **rank_buckets,
                    "max_chosen_rank": self._max_chosen_rank.get(level, 0),
                    "min_top_r_covering_999": _min_top_r_covering(histogram, ranked_vectors),
                    "num_vectors": num_vectors,
                    "straggler_count": int(raw["straggler_count"]),
                    "straggler_fraction": float(raw["straggler_count"] / denominator),
                    "mean_assignment_confidence": float(
                        raw["confidence_sum"] / max(1, int(raw["confidence_count"]))
                    ),
                    "mean_chosen_rank": float(raw["rank_sum"] / denominator),
                    "second_choice_fraction": float(raw["second_choice_count"] / denominator),
                    "third_or_later_choice_fraction": float(
                        raw["third_or_later_choice_count"] / denominator
                    ),
                    "denied_top_fraction": float(raw["denied_top_count"] / denominator),
                    "fallback_fraction": float(raw["fallback_count"] / denominator),
                    "fallback_count": int(raw["fallback_count"]),
                    "denied_top_count": int(raw["denied_top_count"]),
                }
            )
        return TreeUpdateDiagnostics(
            assignment_mode=self.assignment_mode,
            assignment_order=self.assignment_order,
            levels=levels,
        )
