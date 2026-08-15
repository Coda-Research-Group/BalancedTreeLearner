"""Distance helpers following ANN-Benchmarks definitions."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import torch


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Return ANN-Benchmarks set Jaccard similarity for sparse-id arrays."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    intersect = len(set(a) & set(b))
    return intersect / float(len(a) + len(b) - intersect)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized float32 vectors, preserving zero rows."""
    dense = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0, norms, 1.0)
    return dense / safe_norms


def _euclidean_distances(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vectors - query, axis=1)


def _hamming_distances(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.mean(vectors.astype(np.bool_) ^ query.astype(np.bool_), axis=1)


def _jaccard_distances(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    if np.issubdtype(vectors.dtype, np.floating) or np.issubdtype(query.dtype, np.floating):
        raise ValueError("jaccard distance expects sparse integer id arrays, not dense floats.")
    return np.asarray([1.0 - jaccard(vector, query) for vector in vectors])


def _angular_distances(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    dot = vectors @ query
    vec_norms = np.linalg.norm(vectors, axis=1)
    query_norm = float(np.linalg.norm(query))
    denom = vec_norms * query_norm
    similarities = np.divide(dot, denom, out=np.zeros_like(dot, dtype=np.float32), where=denom > 0)
    return 1.0 - similarities


def _inner_product_distances(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    # Raw (unnormalized) inner product, unlike angular/cosine. Negated so that,
    # like every other metric here, smaller means closer — callers argsort
    # ascending regardless of metric (see search.py's candidate selection).
    dot = np.asarray(vectors @ query, dtype=np.float64)
    return -dot


def _writable_float32_array(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if not arr.flags.writeable:
        return arr.copy()
    return arr


class Metric(NamedTuple):
    distances: Callable[[np.ndarray, np.ndarray], np.ndarray]


# Distance definitions mirror ANN-Benchmarks:
# https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/distance.py
metrics = {
    "hamming": Metric(distances=_hamming_distances),
    "jaccard": Metric(distances=_jaccard_distances),
    "euclidean": Metric(distances=_euclidean_distances),
    "angular": Metric(distances=_angular_distances),
    "inner_product": Metric(distances=_inner_product_distances),
}


def compute_distances(
    database: np.ndarray,
    candidates: np.ndarray,
    query: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Compute candidate distances for exact reranking."""
    vectors = database[candidates]
    if metric == "cosine":
        metric = "angular"
    if metric not in metrics:
        raise KeyError(f"Unknown metric '{metric}'. Known metrics are {list(metrics.keys())}")
    return metrics[metric].distances(vectors, query)


def compute_distances_torch(
    database: np.ndarray,
    candidates: np.ndarray,
    query: np.ndarray,
    metric: str,
    device: torch.device,
) -> np.ndarray:
    """Compute ANN-style candidate distances with torch tensors on ``device``."""
    if metric == "cosine":
        metric = "angular"
    if metric not in {"euclidean", "angular", "inner_product"}:
        raise ValueError(
            "torch distance rerank supports only euclidean, angular, cosine, and inner_product."
        )

    vectors = torch.as_tensor(
        np.ascontiguousarray(database[candidates]),
        dtype=torch.float32,
        device=device,
    )
    query_array = _writable_float32_array(query)
    query_tensor = torch.as_tensor(query_array, dtype=torch.float32, device=device)

    if metric == "euclidean":
        distances = torch.linalg.norm(vectors - query_tensor, dim=1)
    elif metric == "inner_product":
        distances = -(vectors * query_tensor).sum(dim=1)
    else:
        dot = (vectors * query_tensor).sum(dim=1)
        vec_norms = torch.linalg.norm(vectors, dim=1)
        query_norm = torch.linalg.norm(query_tensor)
        denom = vec_norms * query_norm
        similarities = torch.zeros_like(dot)
        valid = denom > 0
        similarities[valid] = dot[valid] / denom[valid]
        distances = 1.0 - similarities

    return distances.detach().cpu().numpy()
