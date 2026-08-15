"""FAISS-based approximate nearest-neighbour search for BATL training labels."""

from __future__ import annotations

import logging

import faiss
import numpy as np

from batl.distance import l2_normalize
from batl.utils.config_parsing import is_faiss_gpu_available
from batl.utils.data import as_float32_matrix

LOGGER = logging.getLogger(__name__)
_DEFAULT_GPU_SEARCH_BATCH_SIZE = 4096


def find_approximate_neighbors(
    queries: np.ndarray,
    database: np.ndarray,
    top_k: int,
    subset_size: int,
    seed: int,
    mode: str = "random_subset",
    chunk_size: int = 1_000_000,
    metric: str = "euclidean",
    backend: str = "faiss_cpu",
) -> np.ndarray:
    """Find exact nearest neighbors within a seeded random database subset.

    For angular/cosine, vectors are L2-normalised before indexing and
    IndexFlatIP is used — maximum inner product on the unit sphere equals
    minimum angular distance (same logic as BLISS cosine neighbour search).

    For ``inner_product``, IndexFlatIP is used on the *raw* vectors: no
    normalization, so vector magnitude matters, unlike angular/cosine.
    """
    queries = as_float32_matrix(queries, "queries")
    database = as_float32_matrix(database, "database")
    if queries.shape[1] != database.shape[1]:
        raise ValueError("queries and database must have the same vector dimension.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if subset_size <= 0:
        raise ValueError("subset_size must be positive.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if metric not in {"euclidean", "angular", "cosine", "inner_product"}:
        raise ValueError("metric must be 'euclidean', 'angular', 'cosine', or 'inner_product'.")
    if backend not in {"auto", "faiss_cpu", "faiss_gpu"}:
        raise ValueError("backend must be 'auto', 'faiss_cpu', or 'faiss_gpu'.")

    canonical_metric = "angular" if metric == "cosine" else metric
    # Both angular/cosine and raw inner_product rank by maximum inner product
    # and use IndexFlatIP; only angular/cosine additionally normalizes first.
    use_ip_index = canonical_metric in ("angular", "inner_product")
    normalize = canonical_metric == "angular"
    sample_size = min(subset_size, database.shape[0])
    if top_k > sample_size:
        raise ValueError("top_k cannot exceed the sampled database size.")
    if mode not in {"random_subset", "sequential_chunked"}:
        raise ValueError("mode must be 'random_subset' or 'sequential_chunked'.")

    rng = np.random.default_rng(seed)
    use_full_database = sample_size == database.shape[0]
    if use_full_database:
        subset_indices = np.arange(database.shape[0], dtype=np.int64)
    else:
        subset_indices = rng.choice(database.shape[0], size=sample_size, replace=False).astype(
            np.int64
        )

    query_matrix = np.ascontiguousarray(queries, dtype=np.float32)
    if normalize:
        query_matrix = l2_normalize(query_matrix)
    # Mining against the whole database is the exact-label case. Fancy-indexing
    # it would materialize a second full float32 copy (37 GB at Deep100M), so
    # it always takes the chunked path regardless of the requested mode; the
    # result is identical because the "subset" is every row.
    if mode == "random_subset" and not use_full_database:
        subset = np.ascontiguousarray(database[subset_indices], dtype=np.float32)
        if normalize:
            subset = l2_normalize(subset)
        index, _gpu_resources = _make_faiss_flat_index(database.shape[1], use_ip_index, backend)
        index.add(subset)  # type: ignore[reportCallIssue]
        _, local_indices = _faiss_search(index, query_matrix, top_k, backend=backend)
        return subset_indices[local_indices].astype(np.int64, copy=False)

    return _search_subset_in_chunks(
        database=database,
        subset_indices=subset_indices,
        queries=query_matrix,
        top_k=top_k,
        chunk_size=chunk_size,
        use_ip_index=use_ip_index,
        normalize=normalize,
        backend=backend,
    )


def _faiss_search(
    index: faiss.IndexFlat,
    queries: np.ndarray,
    top_k: int,
    backend: str = "faiss_cpu",
) -> tuple[np.ndarray, np.ndarray]:
    if (
        _resolve_faiss_backend(backend) == "faiss_gpu"
        and queries.shape[0] > _DEFAULT_GPU_SEARCH_BATCH_SIZE
    ):
        return _faiss_search_batched(index, queries, top_k, _DEFAULT_GPU_SEARCH_BATCH_SIZE)

    if index.ntotal >= 1024:
        return index.search(queries, top_k)  # type: ignore[reportCallIssue]

    # Some local FAISS/OpenMP builds crash on tiny searches with many threads.
    previous_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        return index.search(queries, top_k)  # type: ignore[reportCallIssue]
    finally:
        faiss.omp_set_num_threads(previous_threads)


def _faiss_search_batched(
    index: faiss.IndexFlat,
    queries: np.ndarray,
    top_k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Search query batches to avoid FAISS-GPU temporary-memory overflow."""
    distances = np.empty((queries.shape[0], top_k), dtype=np.float32)
    labels = np.empty((queries.shape[0], top_k), dtype=np.int64)
    for start in range(0, queries.shape[0], batch_size):
        end = min(start + batch_size, queries.shape[0])
        batch_distances, batch_labels = index.search(queries[start:end], top_k)  # type: ignore[reportCallIssue]
        distances[start:end] = batch_distances
        labels[start:end] = batch_labels
    return distances, labels


def _search_subset_in_chunks(
    database: np.ndarray,
    subset_indices: np.ndarray,
    queries: np.ndarray,
    top_k: int,
    chunk_size: int,
    use_ip_index: bool = False,
    normalize: bool = False,
    backend: str = "faiss_cpu",
) -> np.ndarray:
    """Exact top-k over a database subset, searched one bounded chunk at a time.

    An earlier version chunked the *adds* but accumulated them into a single
    flat index, so the device still had to hold ``len(subset) * dim * 4``
    bytes — 35.76 GiB when Deep100M mines exact labels. FAISS grows its device
    buffer geometrically and copies, so the transient peak is higher again and
    a 44 GiB card fails partway through (observed: a 21.46 GiB request, the
    capacity for 60M vectors, on job 22710911).

    Exact top-k over a union is the merge of the exact top-k over each part, so
    this searches chunk by chunk and merges the running best. Device memory is
    then bounded by ``chunk_size`` rather than by the subset size, and the
    result is unchanged.

    Within the returned k, ranks are ordered by (distance, database index) so
    the output does not depend on where the chunk boundaries fall.
    """
    sorted_indices = np.sort(subset_indices)
    best: tuple[np.ndarray, np.ndarray] | None = None
    cursor = 0

    for chunk_start in range(0, database.shape[0], chunk_size):
        chunk_end = min(chunk_start + chunk_size, database.shape[0])
        cursor_start = cursor
        while cursor < sorted_indices.size and sorted_indices[cursor] < chunk_end:
            cursor += 1
        if cursor == cursor_start:
            continue

        chunk_indices = sorted_indices[cursor_start:cursor]
        local_indices = chunk_indices - chunk_start
        chunk = np.asarray(database[chunk_start:chunk_end], dtype=np.float32)
        if local_indices.size == chunk.shape[0]:
            # Whole chunk selected (the exact-label case): fancy-indexing it
            # would copy every row for nothing.
            vectors = np.ascontiguousarray(chunk, dtype=np.float32)
        else:
            vectors = np.ascontiguousarray(chunk[local_indices], dtype=np.float32)
        if normalize:
            vectors = l2_normalize(vectors)

        index, gpu_resources = _make_faiss_flat_index(database.shape[1], use_ip_index, backend)
        index.add(vectors)  # type: ignore[reportCallIssue]
        chunk_k = min(top_k, vectors.shape[0])
        scores, local_ranks = _faiss_search(index, queries, chunk_k, backend=backend)
        # Drop the index before the next chunk allocates, so at most one
        # chunk is resident on the device at a time.
        del index, gpu_resources, vectors, chunk

        best = _merge_top_k(
            best,
            scores=scores,
            ids=chunk_indices[local_ranks],
            top_k=top_k,
            larger_is_better=use_ip_index,
        )

    if best is None:
        raise RuntimeError("sequential subset search produced no vectors.")
    return best[1].astype(np.int64, copy=False)


def _merge_top_k(
    best: tuple[np.ndarray, np.ndarray] | None,
    scores: np.ndarray,
    ids: np.ndarray,
    top_k: int,
    larger_is_better: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold one chunk's top-k into the running best, keeping rank order.

    ``larger_is_better`` is True for the angular/cosine and inner_product
    cases, where the index is IndexFlatIP and maximum inner product wins.
    """
    if best is not None:
        scores = np.concatenate([best[0], scores], axis=1)
        ids = np.concatenate([best[1], ids], axis=1)

    rows = np.arange(scores.shape[0])[:, None]
    if scores.shape[1] > top_k:
        ranking = -scores if larger_is_better else scores
        kept = np.argpartition(ranking, top_k - 1, axis=1)[:, :top_k]
        scores = scores[rows, kept]
        ids = ids[rows, kept]

    ranking = -scores if larger_is_better else scores
    order = np.lexsort((ids, ranking), axis=-1)
    return scores[rows, order], ids[rows, order]


def _make_faiss_flat_index(
    dim: int,
    use_ip_index: bool,
    backend: str,
) -> tuple[faiss.IndexFlat, object | None]:
    index: faiss.IndexFlat = faiss.IndexFlatIP(dim) if use_ip_index else faiss.IndexFlatL2(dim)
    if _resolve_faiss_backend(backend) == "faiss_cpu":
        return index, None

    resources = faiss.StandardGpuResources()  # type: ignore[attr-defined]
    gpu_index = faiss.index_cpu_to_gpu(resources, 0, index)  # type: ignore[attr-defined]
    return gpu_index, resources  # type: ignore[return-value]


def _resolve_faiss_backend(backend: str) -> str:
    if backend == "faiss_gpu" and is_faiss_gpu_available():
        return "faiss_gpu"
    if backend == "auto" and is_faiss_gpu_available():
        return "faiss_gpu"
    return "faiss_cpu"
