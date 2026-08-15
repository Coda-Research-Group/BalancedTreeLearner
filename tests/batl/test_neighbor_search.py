import numpy as np
import pytest

from batl.neighbor_search import find_approximate_neighbors


def _brute_force_top_k(
    database: np.ndarray, query: np.ndarray, top_k: int, metric: str
) -> np.ndarray:
    if metric == "euclidean":
        scores = np.linalg.norm(database - query, axis=1)
        order = np.argsort(scores, kind="mergesort")
    elif metric == "inner_product":
        scores = database @ query
        order = np.argsort(-scores, kind="mergesort")
    else:
        raise ValueError(metric)
    return order[:top_k].astype(np.int64)


def test_find_approximate_neighbors_euclidean_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    database = rng.normal(size=(200, 8)).astype(np.float32)
    queries = rng.normal(size=(5, 8)).astype(np.float32)

    result = find_approximate_neighbors(
        queries, database, top_k=3, subset_size=200, seed=0, metric="euclidean"
    )

    for row, query in enumerate(queries):
        expected = _brute_force_top_k(database, query, 3, "euclidean")
        assert result[row].tolist() == expected.tolist()


def test_find_approximate_neighbors_inner_product_matches_brute_force_dot_product() -> None:
    rng = np.random.default_rng(1)
    database = rng.normal(size=(200, 8)).astype(np.float32)
    queries = rng.normal(size=(5, 8)).astype(np.float32)

    result = find_approximate_neighbors(
        queries, database, top_k=3, subset_size=200, seed=0, metric="inner_product"
    )

    for row, query in enumerate(queries):
        expected = _brute_force_top_k(database, query, 3, "inner_product")
        assert result[row].tolist() == expected.tolist()


def test_find_approximate_neighbors_inner_product_is_magnitude_sensitive_unlike_angular() -> None:
    # Same direction, different magnitude: angular treats these as tied
    # (cosine similarity 1.0 to the query for both), inner_product must not.
    database = np.array([[1.0, 0.0], [5.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query = np.array([[1.0, 0.0]], dtype=np.float32)

    ip_top1 = find_approximate_neighbors(
        query, database, top_k=1, subset_size=3, seed=0, metric="inner_product"
    )
    angular_top1 = find_approximate_neighbors(
        query, database, top_k=1, subset_size=3, seed=0, metric="angular"
    )

    # Raw dot product prefers the larger-magnitude same-direction vector.
    assert ip_top1[0, 0] == 1
    # Angular is scale-invariant, so either same-direction candidate is a
    # valid top-1 (both have cosine similarity 1.0 to the query).
    assert angular_top1[0, 0] in (0, 1)


def test_find_approximate_neighbors_inner_product_does_not_normalize_database() -> None:
    # A large-magnitude vector pointing away from the query must lose to a
    # smaller-magnitude vector pointing towards it under raw inner product,
    # unlike a metric that first projects everything onto the unit sphere.
    database = np.array([[1.0, 0.0], [-100.0, 0.0]], dtype=np.float32)
    query = np.array([[1.0, 0.0]], dtype=np.float32)

    result = find_approximate_neighbors(
        query, database, top_k=1, subset_size=2, seed=0, metric="inner_product"
    )

    assert result[0, 0] == 0


def test_find_approximate_neighbors_sequential_chunked_matches_random_subset_for_inner_product() -> (
    None
):
    rng = np.random.default_rng(2)
    database = rng.normal(size=(50, 4)).astype(np.float32)
    queries = rng.normal(size=(3, 4)).astype(np.float32)

    full_subset = find_approximate_neighbors(
        queries,
        database,
        top_k=5,
        subset_size=50,
        seed=0,
        metric="inner_product",
        mode="random_subset",
    )
    chunked = find_approximate_neighbors(
        queries,
        database,
        top_k=5,
        subset_size=50,
        seed=0,
        metric="inner_product",
        mode="sequential_chunked",
        chunk_size=17,
    )

    assert chunked.tolist() == full_subset.tolist()


def test_find_approximate_neighbors_rejects_unsupported_metric() -> None:
    database = np.zeros((4, 2), dtype=np.float32)
    queries = np.zeros((1, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="metric must be"):
        find_approximate_neighbors(
            queries, database, top_k=1, subset_size=4, seed=0, metric="hamming"
        )
