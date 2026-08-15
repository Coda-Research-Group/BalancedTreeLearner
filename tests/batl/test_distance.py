import warnings

import numpy as np
import pytest
import torch

from batl.distance import (
    compute_distances,
    compute_distances_torch,
    l2_normalize,
)


def test_l2_normalize_unit_length() -> None:
    vecs = np.array([[3.0, 4.0]], dtype=np.float32)
    out = l2_normalize(vecs)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_l2_normalize_preserves_zero_rows() -> None:
    vecs = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    out = l2_normalize(vecs)
    assert np.allclose(out[0], [0.0, 0.0])
    assert np.allclose(out[1], [1.0, 0.0])


def test_angular_distance_handles_zero_norm_vectors() -> None:
    database = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    candidates = np.array([0, 1], dtype=np.int64)
    query = np.array([0.0, 0.0], dtype=np.float32)

    distances = compute_distances(database, candidates, query, "angular")

    assert np.isfinite(distances).all()
    assert distances.tolist() == [1.0, 1.0]


def test_compute_distances_euclidean() -> None:
    database = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    candidates = np.array([0, 1], dtype=np.int64)
    query = np.array([0.0, 0.0], dtype=np.float32)

    distances = compute_distances(database, candidates, query, "euclidean")

    assert distances[0] == pytest.approx(0.0)
    assert distances[1] == pytest.approx(5.0)


def test_compute_distances_hamming() -> None:
    database = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.float32)
    candidates = np.array([0, 1], dtype=np.int64)
    query = np.array([1, 0, 0], dtype=np.float32)

    distances = compute_distances(database, candidates, query, "hamming")

    assert distances[0] == pytest.approx(0.0)
    assert distances[1] == pytest.approx(1 / 3)


def test_compute_distances_jaccard() -> None:
    database = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    candidates = np.array([0, 1], dtype=np.int64)
    query = np.array([1, 2, 7], dtype=np.int64)

    distances = compute_distances(database, candidates, query, "jaccard")

    assert distances[0] == pytest.approx(1 - 2 / 4)
    assert distances[1] == pytest.approx(1.0)


def test_compute_distances_jaccard_empty_candidate_is_maximally_distant() -> None:
    empty = np.empty((1, 0), dtype=np.int64)
    candidates = np.array([0], dtype=np.int64)
    query = np.array([1], dtype=np.int64)

    distances = compute_distances(empty, candidates, query, "jaccard")

    assert distances[0] == 1.0


def test_compute_distances_angular_orthogonal_vectors() -> None:
    database = np.array([[1.0, 0.0]], dtype=np.float32)
    candidates = np.array([0], dtype=np.int64)
    query = np.array([0.0, 1.0], dtype=np.float32)

    distances = compute_distances(database, candidates, query, "angular")

    assert distances[0] == pytest.approx(1.0)


def test_compute_distances_cosine_aliases_angular() -> None:
    database = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    candidates = np.array([0, 1], dtype=np.int64)
    query = np.array([1.0, 0.0], dtype=np.float32)

    dist_cosine = compute_distances(database, candidates, query, "cosine")
    dist_angular = compute_distances(database, candidates, query, "angular")

    assert np.allclose(dist_cosine, dist_angular)


def test_compute_distances_torch_matches_numpy_for_euclidean() -> None:
    database = np.array([[0.0, 0.0], [3.0, 4.0], [2.0, 0.0]], dtype=np.float32)
    candidates = np.array([0, 1, 2], dtype=np.int64)
    query = np.array([0.0, 0.0], dtype=np.float32)

    numpy_distances = compute_distances(database, candidates, query, "euclidean")
    torch_distances = compute_distances_torch(
        database,
        candidates,
        query,
        "euclidean",
        torch.device("cpu"),
    )

    assert torch_distances.tolist() == pytest.approx(numpy_distances.tolist())


def test_compute_distances_torch_matches_numpy_for_angular_with_zero_vectors() -> None:
    database = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    candidates = np.array([0, 1, 2], dtype=np.int64)
    query = np.array([0.0, 0.0], dtype=np.float32)

    numpy_distances = compute_distances(database, candidates, query, "angular")
    torch_distances = compute_distances_torch(
        database,
        candidates,
        query,
        "cosine",
        torch.device("cpu"),
    )

    assert np.isfinite(torch_distances).all()
    assert torch_distances.tolist() == pytest.approx(numpy_distances.tolist())


def test_compute_distances_torch_copies_read_only_query_without_warning() -> None:
    database = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    candidates = np.array([0, 1], dtype=np.int64)
    query = np.array([0.0, 0.0], dtype=np.float32)
    query.setflags(write=False)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        distances = compute_distances_torch(
            database,
            candidates,
            query,
            "euclidean",
            torch.device("cpu"),
        )

    assert distances.tolist() == pytest.approx([0.0, 5.0])
    assert not [
        warning for warning in captured if "NumPy array is not writable" in str(warning.message)
    ]


def test_compute_distances_inner_product_ranks_by_raw_dot_product() -> None:
    # Same direction, different magnitude: unlike angular, inner_product must
    # not treat these as equally close.
    database = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    candidates = np.array([0, 1, 2], dtype=np.int64)
    query = np.array([1.0, 0.0], dtype=np.float32)

    distances = compute_distances(database, candidates, query, "inner_product")

    assert distances.tolist() == pytest.approx([-1.0, -2.0, 0.0])
    # Larger raw dot product (row 1) must rank closer (smaller distance).
    assert distances[1] < distances[0] < distances[2]


def test_compute_distances_torch_matches_numpy_for_inner_product() -> None:
    database = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    candidates = np.array([0, 1, 2], dtype=np.int64)
    query = np.array([1.0, 0.0], dtype=np.float32)

    numpy_distances = compute_distances(database, candidates, query, "inner_product")
    torch_distances = compute_distances_torch(
        database,
        candidates,
        query,
        "inner_product",
        torch.device("cpu"),
    )

    assert torch_distances.tolist() == pytest.approx(numpy_distances.tolist())


def test_compute_distances_torch_rejects_unsupported_metric() -> None:
    database = np.array([[1, 0]], dtype=np.float32)
    candidates = np.array([0], dtype=np.int64)
    query = np.array([1, 0], dtype=np.float32)

    with pytest.raises(ValueError, match="supports only"):
        compute_distances_torch(database, candidates, query, "hamming", torch.device("cpu"))


def test_compute_distances_unknown_metric_raises() -> None:
    database = np.array([[1.0, 0.0]], dtype=np.float32)
    candidates = np.array([0], dtype=np.int64)
    query = np.array([1.0, 0.0], dtype=np.float32)

    with pytest.raises(KeyError, match="unknown_metric"):
        compute_distances(database, candidates, query, "unknown_metric")
