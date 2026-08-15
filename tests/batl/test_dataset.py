from pathlib import Path

import h5py
import numpy as np
import pytest

from batl.utils.data import (
    load_queries_and_gt,
    load_vectors,
    matrix_metadata,
    read_fbin,
    read_i8bin,
    read_i32bin,
    read_ibin,
    read_u8bin,
    write_fbin,
    write_ibin,
)


def _write_fvecs(path: Path, vectors: np.ndarray) -> None:
    with path.open("wb") as f:
        for row in vectors.astype(np.float32):
            np.asarray([row.size], dtype=np.int32).tofile(f)
            row.tofile(f)


def _write_ivecs(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as f:
        for row in values.astype(np.int32):
            np.asarray([row.size], dtype=np.int32).tofile(f)
            row.tofile(f)


def _write_bin(path: Path, values: np.ndarray, dtype: np.dtype) -> None:
    matrix = np.asarray(values, dtype=dtype)
    with path.open("wb") as f:
        np.asarray(matrix.shape, dtype=np.dtype("<u4")).tofile(f)
        matrix.tofile(f)


def test_read_fbin_reads_chunk_by_vector_index(tmp_path) -> None:
    vectors = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=np.float32,
    )
    path = tmp_path / "base.fbin"
    write_fbin(path, vectors)

    loaded = read_fbin(path, start_idx=1, chunk_size=1)

    assert loaded.dtype == np.float32
    assert loaded.shape == (1, 2)
    assert np.array_equal(loaded, vectors[1:2])


def test_read_u8bin_reads_bigann_vectors_by_chunk(tmp_path) -> None:
    vectors = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    path = tmp_path / "base.u8bin"
    _write_bin(path, vectors, np.dtype("u1"))

    loaded = read_u8bin(path, start_idx=1, chunk_size=1)

    assert loaded.dtype == np.uint8
    assert loaded.shape == (1, 3)
    assert np.array_equal(loaded, vectors[1:2])


def test_read_i8bin_reads_spacev_vectors_by_chunk(tmp_path) -> None:
    vectors = np.array([[-3, -2, -1], [4, 5, 6]], dtype=np.int8)
    path = tmp_path / "base.i8bin"
    _write_bin(path, vectors, np.dtype("i1"))

    loaded = read_i8bin(path, start_idx=0, chunk_size=1)

    assert loaded.dtype == np.int8
    assert loaded.shape == (1, 3)
    assert np.array_equal(loaded, vectors[:1])


def test_read_ibin_reads_chunk_by_vector_index(tmp_path) -> None:
    values = np.array([[7, 8, 9], [1, 2, 3]], dtype=np.int32)
    path = tmp_path / "groundtruth.ibin"
    write_ibin(path, values)

    loaded = read_ibin(path, start_idx=1, chunk_size=None)

    assert loaded.dtype == np.int32
    assert loaded.shape == (1, 3)
    assert np.array_equal(loaded, values[1:])


def test_read_i32bin_reads_bigann_ground_truth_by_chunk(tmp_path) -> None:
    values = np.array([[7, 8, 9], [1, 2, 3]], dtype=np.int32)
    path = tmp_path / "groundtruth.i32bin"
    _write_bin(path, values, np.dtype("<i4"))

    loaded = read_i32bin(path, start_idx=1, chunk_size=None)

    assert loaded.dtype == np.int32
    assert loaded.shape == (1, 3)
    assert np.array_equal(loaded, values[1:])


def test_write_fbin_replaces_existing_file_atomically(tmp_path) -> None:
    path = tmp_path / "base.fbin"
    original = np.array([[1.0, 2.0]], dtype=np.float32)
    replacement = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    write_fbin(path, original)

    write_fbin(path, replacement)

    assert np.array_equal(np.asarray(read_fbin(path)), replacement)
    assert not list(tmp_path.glob("*.tmp"))


def test_load_vectors_reads_fvecs_with_subset(tmp_path) -> None:
    vectors = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=np.float32,
    )
    _write_fvecs(tmp_path / "train.fvecs", vectors)

    loaded = load_vectors("deep1b", str(tmp_path), split="train", subset_size=2)

    assert loaded.dtype == np.float32
    assert loaded.shape == (2, 2)
    assert np.array_equal(np.asarray(loaded), vectors[:2])


def test_load_vectors_reads_fbin_with_subset(tmp_path) -> None:
    vectors = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=np.float32,
    )
    write_fbin(tmp_path / "base.fbin", vectors)

    loaded = load_vectors("deep1b", str(tmp_path), split="train", subset_size=2)

    assert loaded.dtype == np.float32
    assert loaded.shape == (2, 2)
    assert np.array_equal(np.asarray(loaded), vectors[:2])


def test_load_vectors_reads_u8bin_as_float32_with_subset(tmp_path) -> None:
    vectors = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint8)
    _write_bin(tmp_path / "base.u8bin", vectors, np.dtype("u1"))

    loaded = load_vectors("bigann", str(tmp_path), split="train", subset_size=2)

    assert loaded.dtype == np.float32
    assert loaded.shape == (2, 2)
    assert np.array_equal(loaded, vectors[:2].astype(np.float32))


def test_load_vectors_reads_i8bin_as_float32_with_subset(tmp_path) -> None:
    vectors = np.array([[-1, 2], [3, -4], [5, 6]], dtype=np.int8)
    _write_bin(tmp_path / "base.i8bin", vectors, np.dtype("i1"))

    loaded = load_vectors("spacev", str(tmp_path), split="train", subset_size=2)

    assert loaded.dtype == np.float32
    assert loaded.shape == (2, 2)
    assert np.array_equal(loaded, vectors[:2].astype(np.float32))


def test_load_vectors_raises_when_subset_size_exceeds_available_rows(tmp_path) -> None:
    vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    write_fbin(tmp_path / "base.fbin", vectors)

    with pytest.raises(ValueError, match="subset_size=3 exceeds available rows"):
        load_vectors("deep1b", str(tmp_path), split="train", subset_size=3)


def test_load_vectors_uses_explicit_base_path_manifest_override(tmp_path) -> None:
    data_dir = tmp_path / "data"
    manifest_dir = tmp_path / "manifest"
    vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    write_fbin(data_dir / "deep_base.fbin", vectors)
    manifest_dir.mkdir()

    loaded = load_vectors(
        "deep1b",
        str(manifest_dir),
        split="train",
        subset_size=None,
        base_path=str(data_dir / "deep_base.fbin"),
    )

    assert np.array_equal(np.asarray(loaded), vectors)


def test_load_vectors_reads_npy_fixture(tmp_path) -> None:
    vectors = np.array([[1.0, 0.0, 2.0]], dtype=np.float32)
    np.save(tmp_path / "vectors.npy", vectors)

    loaded = load_vectors("synthetic", str(tmp_path), split="train")

    assert loaded.dtype == np.float32
    assert np.array_equal(loaded, vectors)


def test_load_vectors_reads_ann_benchmarks_hdf5(tmp_path) -> None:
    path = tmp_path / "sift-128-euclidean.hdf5"
    vectors = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("train", data=vectors)

    loaded = load_vectors("sift1m", str(path), split="train", subset_size=2)

    assert loaded.dtype == np.float32
    assert np.array_equal(loaded, vectors[:2])


def test_load_vectors_reads_glove_ann_benchmarks_hdf5_file(tmp_path) -> None:
    path = tmp_path / "glove-100-angular.hdf5"
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("train", data=vectors)

    loaded = load_vectors("glove100", str(path), split="train")

    assert loaded.dtype == np.float32
    assert np.array_equal(loaded, vectors)


def test_load_queries_and_gt_reads_fvecs_and_ivecs(tmp_path) -> None:
    queries = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gt = np.array([[7, 8, 9], [1, 2, 3]], dtype=np.int64)
    _write_fvecs(tmp_path / "queries.fvecs", queries)
    _write_ivecs(tmp_path / "groundtruth.ivecs", gt)

    loaded_queries, loaded_gt = load_queries_and_gt("deep1b", str(tmp_path), num_queries=1)

    assert loaded_queries.dtype == np.float32
    assert loaded_gt.dtype == np.int64
    assert np.array_equal(np.asarray(loaded_queries), queries[:1])
    assert np.array_equal(loaded_gt, gt[:1])


def test_load_queries_and_gt_reads_fbin_and_ibin(tmp_path) -> None:
    queries = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gt = np.array([[7, 8, 9], [1, 2, 3]], dtype=np.int32)
    write_fbin(tmp_path / "query.fbin", queries)
    write_ibin(tmp_path / "groundtruth.ibin", gt)

    loaded_queries, loaded_gt = load_queries_and_gt("deep1b", str(tmp_path), num_queries=1)

    assert loaded_queries.dtype == np.float32
    assert loaded_gt.dtype == np.int64
    assert np.array_equal(np.asarray(loaded_queries), queries[:1])
    assert np.array_equal(loaded_gt, gt[:1])


def test_load_queries_and_gt_reads_i8bin_and_i32bin(tmp_path) -> None:
    queries = np.array([[-1, 2], [3, -4]], dtype=np.int8)
    gt = np.array([[7, 8, 9], [1, 2, 3]], dtype=np.int32)
    _write_bin(tmp_path / "query.30K.i8bin", queries, np.dtype("i1"))
    _write_bin(tmp_path / "groundtruth.30K.i32bin", gt, np.dtype("<i4"))

    loaded_queries, loaded_gt = load_queries_and_gt("spacev", str(tmp_path), num_queries=1)

    assert loaded_queries.dtype == np.float32
    assert loaded_gt.dtype == np.int64
    assert np.array_equal(loaded_queries, queries[:1].astype(np.float32))
    assert np.array_equal(loaded_gt, gt[:1])


def test_load_queries_and_gt_reads_ann_sift1b_public_query_and_idx_names(tmp_path) -> None:
    queries = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    gt = np.array([[7, 8, 9], [1, 2, 3]], dtype=np.int32)
    _write_bin(tmp_path / "query.public.10K.u8bin", queries, np.dtype("u1"))
    _write_ivecs(tmp_path / "idx_100M.ivecs", gt)

    loaded_queries, loaded_gt = load_queries_and_gt("bigann", str(tmp_path), num_queries=1)

    assert loaded_queries.dtype == np.float32
    assert loaded_gt.dtype == np.int64
    assert np.array_equal(loaded_queries, queries[:1].astype(np.float32))
    assert np.array_equal(loaded_gt, gt[:1])


def test_load_queries_and_gt_uses_explicit_manifest_paths(tmp_path) -> None:
    data_dir = tmp_path / "data"
    manifest_dir = tmp_path / "manifest"
    queries = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gt = np.array([[7, 8, 9], [1, 2, 3]], dtype=np.int32)
    write_fbin(data_dir / "deep_query.fbin", queries)
    write_ibin(data_dir / "deep_gt.ibin", gt)
    manifest_dir.mkdir()

    loaded_queries, loaded_gt = load_queries_and_gt(
        "deep1b",
        str(manifest_dir),
        num_queries=2,
        query_path=str(data_dir / "deep_query.fbin"),
        ground_truth_path=str(data_dir / "deep_gt.ibin"),
    )

    assert np.array_equal(np.asarray(loaded_queries), queries)
    assert np.array_equal(loaded_gt, gt)


def test_load_queries_and_gt_converts_laion_hdf5_knns_to_zero_based_ids(tmp_path) -> None:
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    one_based_knns = np.array([[3, 1, 2], [2, 3, 1]], dtype=np.int32)
    query_path = tmp_path / "query.h5"
    gt_path = tmp_path / "groundtruth.h5"
    with h5py.File(query_path, "w") as f:
        f.create_dataset("emb", data=queries)
    with h5py.File(gt_path, "w") as f:
        f.create_dataset("knns", data=one_based_knns)

    loaded_queries, loaded_gt = load_queries_and_gt(
        "laion5b",
        str(tmp_path),
        num_queries=2,
        query_path=str(query_path),
        ground_truth_path=str(gt_path),
    )

    assert np.array_equal(loaded_queries, queries)
    assert np.array_equal(loaded_gt, one_based_knns - 1)


def test_load_queries_and_gt_transposes_documented_laion_knns_layout(tmp_path) -> None:
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    knns_by_neighbor_then_query = np.array(
        [
            [3, 2],
            [1, 3],
            [2, 1],
        ],
        dtype=np.int32,
    )
    query_path = tmp_path / "query.h5"
    gt_path = tmp_path / "groundtruth.h5"
    with h5py.File(query_path, "w") as f:
        f.create_dataset("emb", data=queries)
    with h5py.File(gt_path, "w") as f:
        f.create_dataset("knns", data=knns_by_neighbor_then_query)

    _loaded_queries, loaded_gt = load_queries_and_gt(
        "laion5b_subset",
        str(tmp_path),
        num_queries=2,
        query_path=str(query_path),
        ground_truth_path=str(gt_path),
    )

    expected = knns_by_neighbor_then_query.T - 1
    assert np.array_equal(loaded_gt, expected)


def test_load_queries_and_gt_reads_npy_fixture(tmp_path) -> None:
    queries = np.array([[1.0, 2.0]], dtype=np.float32)
    gt = np.array([[4, 5]], dtype=np.int64)
    np.save(tmp_path / "queries.npy", queries)
    np.save(tmp_path / "ground_truth.npy", gt)

    loaded_queries, loaded_gt = load_queries_and_gt("synthetic", str(tmp_path), num_queries=10)

    assert np.array_equal(loaded_queries, queries)
    assert np.array_equal(loaded_gt, gt)


def test_load_queries_and_gt_reads_ann_benchmarks_hdf5(tmp_path) -> None:
    path = tmp_path / "sift-128-euclidean.hdf5"
    queries = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    neighbors = np.array([[7, 8, 9], [1, 2, 3]], dtype=np.int32)
    with h5py.File(path, "w") as f:
        f.create_dataset("test", data=queries)
        f.create_dataset("neighbors", data=neighbors)

    loaded_queries, loaded_gt = load_queries_and_gt("sift1m", str(path), num_queries=1)

    assert loaded_queries.dtype == np.float32
    assert loaded_gt.dtype == np.int64
    assert np.array_equal(loaded_queries, queries[:1])
    assert np.array_equal(loaded_gt, neighbors[:1])


def test_matrix_metadata_reads_supported_headers_without_payload_load(tmp_path) -> None:
    vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gt = np.array([[7, 8, 9], [1, 2, 3]], dtype=np.int32)
    write_fbin(tmp_path / "base.fbin", vectors)
    _write_ivecs(tmp_path / "groundtruth.ivecs", gt)

    base_meta = matrix_metadata(tmp_path / "base.fbin")
    u8_meta = (
        matrix_metadata(tmp_path / "base.u8bin") if (tmp_path / "base.u8bin").exists() else None
    )
    gt_meta = matrix_metadata(tmp_path / "groundtruth.ivecs")

    assert base_meta.shape == (2, 2)
    assert base_meta.dtype == np.float32
    assert u8_meta is None
    assert gt_meta.shape == (2, 3)
    assert gt_meta.dtype == np.int32


def test_matrix_metadata_reads_bigann_binary_suffixes(tmp_path) -> None:
    vectors = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    queries = np.array([[-1, 2], [3, -4]], dtype=np.int8)
    gt = np.array([[7, 8, 9], [1, 2, 3]], dtype=np.int32)
    _write_bin(tmp_path / "base.u8bin", vectors, np.dtype("u1"))
    _write_bin(tmp_path / "query.i8bin", queries, np.dtype("i1"))
    _write_bin(tmp_path / "groundtruth.i32bin", gt, np.dtype("<i4"))

    assert matrix_metadata(tmp_path / "base.u8bin").shape == (2, 2)
    assert matrix_metadata(tmp_path / "base.u8bin").dtype == np.uint8
    assert matrix_metadata(tmp_path / "query.i8bin").shape == (2, 2)
    assert matrix_metadata(tmp_path / "query.i8bin").dtype == np.int8
    assert matrix_metadata(tmp_path / "groundtruth.i32bin").shape == (2, 3)
    assert matrix_metadata(tmp_path / "groundtruth.i32bin").dtype == np.int32
