"""Evaluation metrics and lightweight measurement helpers for BATL."""

from __future__ import annotations

import numpy as np

from batl.tree import BATLTree
from batl.utils.index_parsing import create_index_payload


def recall_at_k(
    retrieved: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> np.ndarray:
    """Compute per-query Recall@k.

    Returns an array of shape ``(n_queries,)`` so callers can inspect the
    distribution (zero-recall outliers, percentiles). Search padding sentinels
    of ``-1`` never count as hits.

    Memory: the vectorised path materialises a ``(n_queries, k, k)`` bool
    intermediate. This is fine for benchmark settings (``k`` is typically
    ``10``-``100``); switch to a chunked or set-based implementation if you
    need ``k`` in the thousands.
    """
    retrieved = np.asarray(retrieved)
    ground_truth = np.asarray(ground_truth)
    if retrieved.ndim != 2 or ground_truth.ndim != 2:
        raise ValueError("retrieved and ground_truth must be 2D arrays.")
    if retrieved.shape[0] != ground_truth.shape[0]:
        raise ValueError("retrieved and ground_truth must have the same number of queries.")
    if retrieved.shape[1] < k or ground_truth.shape[1] < k:
        raise ValueError("retrieved and ground_truth must both contain at least k columns.")

    retrieved_top = retrieved[:, :k]
    ground_truth_top = ground_truth[:, :k]
    valid = retrieved_top >= 0
    matches = (retrieved_top[:, :, None] == ground_truth_top[:, None, :]) & valid[:, :, None]
    per_ground_truth_hit = matches.any(axis=1)
    return per_ground_truth_hit.sum(axis=1, dtype=np.float64) / k


def index_size_mb(models: list, trees: list) -> float:
    """Return approximate portable checkpoint size in megabytes."""
    payload = create_index_payload(models, trees)
    import io

    buffer = io.BytesIO()
    import torch

    torch.save(payload, buffer)
    return len(buffer.getvalue()) / (1024 * 1024)


def estimate_candidate_set_size(trees: list[BATLTree], num_return_leaves: int) -> float:
    """Estimate the pre-frequency-filter candidate-count upper bound.

    Measured ``n_candidates`` can be lower for ensembles because multi-tree
    search keeps only candidates that appear in enough selected leaf buckets.
    """
    return float(sum(num_return_leaves * tree.leaf_size_stats()["mean"] for tree in trees))
