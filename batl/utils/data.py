"""Dataset loading helpers for BATL experiments.

The fbin/ibin helpers adapt the binary matrix convention from dbaranchuk's
``io_utils`` Pastebin snippet:
https://pastebin.com/BAf6bM5L
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

import h5py
import numpy as np


def as_float32_matrix(values: np.ndarray, name: str) -> np.ndarray:
    """Ensure an array is a 2D float32 matrix."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    return array


def load_vectors(
    dataset_name: str,
    path: str,
    split: str,
    subset_size: int | None = None,
    *,
    base_path: str | None = None,
) -> np.ndarray:
    """Load database vectors for a dataset split.

    ``base_path`` is an optional manifest override for large datasets where the
    base, query, and ground-truth files live at separate explicit paths.
    """
    if subset_size is not None and subset_size <= 0:
        raise ValueError("subset_size must be positive when provided.")

    source = _resolve_vector_file(Path(path), split, explicit_path=base_path)
    if subset_size is not None:
        metadata = matrix_metadata(source, hdf5_key="train")
        if subset_size > metadata.shape[0]:
            raise ValueError(
                f"subset_size={subset_size} exceeds available rows in {source}: {metadata.shape[0]}"
            )
    return _load_float_matrix(source, subset_size=subset_size, hdf5_key="train")


def load_queries_and_gt(
    dataset_name: str,
    path: str,
    num_queries: int = 10_000,
    *,
    query_path: str | None = None,
    ground_truth_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load query vectors and ground-truth neighbor IDs.

    ``query_path`` and ``ground_truth_path`` are optional manifest overrides.
    When neither is provided and ``path`` is an ANN-Benchmarks HDF5 file, the
    legacy SIFT/GloVe ``test``/``neighbors`` layout is used unchanged.
    """
    root = Path(path)
    if query_path is None and ground_truth_path is None and root.suffix in {".hdf5", ".h5"}:
        return _load_hdf5_queries_and_gt(root, num_queries)

    query_file = _resolve_query_file(root, explicit_path=query_path)
    gt_file = _resolve_ground_truth_file(root, explicit_path=ground_truth_path)
    queries = _load_float_matrix(query_file, subset_size=num_queries, hdf5_key="test")
    gt_subset_size = (
        None
        if _is_laion_dataset(dataset_name) and gt_file.suffix in {".hdf5", ".h5"}
        else num_queries
    )
    ground_truth = _load_int_matrix(gt_file, subset_size=gt_subset_size, hdf5_key="neighbors")
    ground_truth = _normalize_ground_truth_for_dataset(
        dataset_name=dataset_name,
        ground_truth=ground_truth,
        query_count=queries.shape[0],
    )

    return queries, ground_truth


@dataclass(frozen=True)
class MatrixMetadata:
    """Lightweight shape/dtype metadata for supported matrix files."""

    path: Path
    shape: tuple[int, int]
    dtype: np.dtype


def _npy_metadata(source, **kwargs):
    arr = np.load(source, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"{source} must contain a 2D matrix.")
    return MatrixMetadata(source, tuple(arr.shape), np.dtype(arr.dtype))


def _fvecs_metadata(source, **kwargs):
    dim, total = _vecs_metadata(source, value_dtype=np.float32)
    return MatrixMetadata(source, (total, dim), np.dtype(np.float32))


def _ivecs_metadata(source, **kwargs):
    dim, total = _vecs_metadata(source, value_dtype=np.int32)
    return MatrixMetadata(source, (total, dim), np.dtype(np.int32))


def _fbin_metadata(source, **kwargs):
    nvecs, dim = _bin_metadata(source, np.dtype("<f4"))
    return MatrixMetadata(source, (nvecs, dim), np.dtype(np.float32))


def _f32bin_metadata(source, **kwargs):
    nvecs, dim = _bin_metadata(source, np.dtype("<f4"))
    return MatrixMetadata(source, (nvecs, dim), np.dtype(np.float32))


def _u8bin_metadata(source, **kwargs):
    nvecs, dim = _bin_metadata(source, np.dtype("u1"))
    return MatrixMetadata(source, (nvecs, dim), np.dtype(np.uint8))


def _i8bin_metadata(source, **kwargs):
    nvecs, dim = _bin_metadata(source, np.dtype("i1"))
    return MatrixMetadata(source, (nvecs, dim), np.dtype(np.int8))


def _ibin_metadata(source, **kwargs):
    nvecs, dim = _bin_metadata(source, np.dtype("<i4"))
    return MatrixMetadata(source, (nvecs, dim), np.dtype(np.int32))


def _i32bin_metadata(source, **kwargs):
    nvecs, dim = _bin_metadata(source, np.dtype("<i4"))
    return MatrixMetadata(source, (nvecs, dim), np.dtype(np.int32))


def _hdf5_metadata(source, hdf5_key=None, **kwargs):
    with h5py.File(source, "r") as f:
        keys = list(f.keys())
        if hdf5_key is None or hdf5_key not in f:
            if len(keys) == 1:
                hdf5_key = keys[0]
            elif "emb" in keys:
                hdf5_key = "emb"
            elif hdf5_key is None:
                raise ValueError("hdf5_key is required for HDF5 metadata and could not be guessed.")
            else:
                raise ValueError(
                    f"{source} does not contain HDF5 key {hdf5_key!r}. Available: {keys}"
                )
        dataset = f[hdf5_key]
        if dataset.ndim != 2:  # type: ignore[reportAttributeAccessIssue]
            raise ValueError(f"{source}:{hdf5_key} must contain a 2D matrix.")
        return MatrixMetadata(
            source,
            tuple(dataset.shape),  # type: ignore[reportArgumentType]
            np.dtype(dataset.dtype),  # type: ignore[reportArgumentType]
        )


_METADATA_LOADERS = {
    ".npy": _npy_metadata,
    ".fvecs": _fvecs_metadata,
    ".ivecs": _ivecs_metadata,
    ".fbin": _fbin_metadata,
    ".f32bin": _f32bin_metadata,
    ".u8bin": _u8bin_metadata,
    ".i8bin": _i8bin_metadata,
    ".ibin": _ibin_metadata,
    ".i32bin": _i32bin_metadata,
    ".hdf5": _hdf5_metadata,
    ".h5": _hdf5_metadata,
}


def matrix_metadata(path: str | Path, *, hdf5_key: str | None = None) -> MatrixMetadata:
    """Return matrix shape and dtype without loading the full payload."""
    source = Path(path)
    loader = _METADATA_LOADERS.get(source.suffix)
    if loader is None:
        raise ValueError(f"Unsupported matrix file format for {source}")
    return loader(source, hdf5_key=hdf5_key)


def _resolve_vector_file(root: Path, split: str, *, explicit_path: str | None = None) -> Path:
    if explicit_path is not None:
        return _existing_path(explicit_path, "base vector file")
    if root.is_file():
        return root
    candidates = [
        root / f"{split}.fvecs",
        root / f"{split}.fbin",
        root / f"{split}.u8bin",
        root / f"{split}.i8bin",
        root / f"{split}.npy",
        root / "vectors.npy",
        root / "base.fvecs",
        root / "base.fbin",
        root / "base.u8bin",
        root / "base.i8bin",
        root / "base.100M.u8bin",
        root / "base.100M.i8bin",
        root / "base.100m.u8bin",
        root / "base.100m.i8bin",
        root / "base.npy",
        root / "sift-128-euclidean.hdf5",
    ]
    return _first_existing(candidates, f"Could not find vector file in {root}")


def _resolve_query_file(root: Path, *, explicit_path: str | None = None) -> Path:
    if explicit_path is not None:
        return _existing_path(explicit_path, "query file")
    if root.is_file():
        raise ValueError(
            "load_queries_and_gt expects a directory containing queries and ground truth."
        )
    candidates = [
        root / "queries.fvecs",
        root / "queries.fbin",
        root / "queries.u8bin",
        root / "queries.i8bin",
        root / "query.fvecs",
        root / "query.fbin",
        root / "query.u8bin",
        root / "query.i8bin",
        root / "query.public.10K.u8bin",
        root / "query.public.10k.u8bin",
        root / "query.10K.u8bin",
        root / "query.10k.u8bin",
        root / "query.30K.i8bin",
        root / "query.30k.i8bin",
        root / "queries.npy",
        root / "query.npy",
    ]
    return _first_existing(candidates, f"Could not find query file in {root}")


def _resolve_ground_truth_file(root: Path, *, explicit_path: str | None = None) -> Path:
    if explicit_path is not None:
        return _existing_path(explicit_path, "ground-truth file")
    candidates = [
        root / "groundtruth.ivecs",
        root / "groundtruth.ibin",
        root / "groundtruth.i32bin",
        root / "groundtruth.10K.i32bin",
        root / "groundtruth.10k.i32bin",
        root / "groundtruth.30K.i32bin",
        root / "groundtruth.30k.i32bin",
        root / "idx_100M.ivecs",
        root / "idx_100m.ivecs",
        root / "gnd" / "idx_100M.ivecs",
        root / "gnd" / "idx_100m.ivecs",
        root / "ground_truth.ivecs",
        root / "ground_truth.ibin",
        root / "ground_truth.i32bin",
        root / "gt.ivecs",
        root / "gt.ibin",
        root / "gt.i32bin",
        root / "ground_truth.npy",
        root / "groundtruth.npy",
        root / "gt.npy",
    ]
    return _first_existing(candidates, f"Could not find ground-truth file in {root}")


def _existing_path(path: str, label: str) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Explicit {label} does not exist: {candidate}")
    return candidate


def _first_existing(candidates: list[Path], message: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(message)


def _load_npy_float(path, subset_size, **kwargs):
    vectors = np.load(path, mmap_mode="r")
    result = vectors[:subset_size] if subset_size is not None else vectors
    return np.asarray(result, dtype=np.float32)


def _load_fvecs_float(path, subset_size, **kwargs):
    return _load_fvecs(path, subset_size=subset_size)


def _load_fbin_float(path, subset_size, **kwargs):
    return read_fbin(path, chunk_size=subset_size)


def _load_u8bin_float(path, subset_size, **kwargs):
    return np.asarray(read_u8bin(path, chunk_size=subset_size), dtype=np.float32)


def _load_i8bin_float(path, subset_size, **kwargs):
    return np.asarray(read_i8bin(path, chunk_size=subset_size), dtype=np.float32)


_HDF5_FLOAT_FALLBACKS = ("emb",)
_HDF5_INT_FALLBACKS = ("knns", "neighbors")


def _resolve_hdf5_key(path: Path, requested: str, fallbacks: tuple[str, ...]) -> str:
    with h5py.File(path, "r") as f:
        if requested in f:
            return requested
        for candidate in fallbacks:
            if candidate in f:
                return candidate
        keys = list(f.keys())
        if len(keys) == 1:
            return keys[0]
        raise ValueError(f"{path} does not contain HDF5 key {requested!r}. Available: {keys}")


def _load_hdf5_float(path, subset_size, hdf5_key, **kwargs):
    key = _resolve_hdf5_key(path, hdf5_key, _HDF5_FLOAT_FALLBACKS)
    return _load_hdf5_matrix(path, subset_size=subset_size, key=key, dtype=np.float32)


_FLOAT_LOADERS = {
    ".npy": _load_npy_float,
    ".fvecs": _load_fvecs_float,
    ".fbin": _load_fbin_float,
    ".f32bin": _load_fbin_float,
    ".u8bin": _load_u8bin_float,
    ".i8bin": _load_i8bin_float,
    ".hdf5": _load_hdf5_float,
    ".h5": _load_hdf5_float,
}


def _load_float_matrix(path: Path, subset_size: int | None, hdf5_key: str) -> np.ndarray:
    loader = _FLOAT_LOADERS.get(path.suffix)
    if loader is None:
        raise ValueError(f"Unsupported float matrix file format for {path}")
    return loader(path, subset_size=subset_size, hdf5_key=hdf5_key)


def _load_npy_int(path, subset_size, **kwargs):
    values = np.load(path, mmap_mode="r")
    result = values[:subset_size] if subset_size is not None else values
    return np.asarray(result, dtype=np.int64)


def _load_ivecs_int(path, subset_size, **kwargs):
    return _load_ivecs(path, subset_size=subset_size)


def _load_ibin_int(path, subset_size, **kwargs):
    return np.asarray(read_ibin(path, chunk_size=subset_size), dtype=np.int64)


def _load_i32bin_int(path, subset_size, **kwargs):
    return np.asarray(read_i32bin(path, chunk_size=subset_size), dtype=np.int64)


def _load_hdf5_int(path, subset_size, hdf5_key, **kwargs):
    key = _resolve_hdf5_key(path, hdf5_key, _HDF5_INT_FALLBACKS)
    return _load_hdf5_matrix(path, subset_size=subset_size, key=key, dtype=np.int64)


_INT_LOADERS = {
    ".npy": _load_npy_int,
    ".ivecs": _load_ivecs_int,
    ".ibin": _load_ibin_int,
    ".i32bin": _load_i32bin_int,
    ".hdf5": _load_hdf5_int,
    ".h5": _load_hdf5_int,
}


def _load_int_matrix(path: Path, subset_size: int | None, hdf5_key: str) -> np.ndarray:
    loader = _INT_LOADERS.get(path.suffix)
    if loader is None:
        raise ValueError(f"Unsupported integer matrix file format for {path}")
    return loader(path, subset_size=subset_size, hdf5_key=hdf5_key)


def _is_laion_dataset(dataset_name: str) -> bool:
    return dataset_name in {"laion5b", "laion5b_subset"}


def _normalize_ground_truth_for_dataset(
    *,
    dataset_name: str,
    ground_truth: np.ndarray,
    query_count: int,
) -> np.ndarray:
    if not _is_laion_dataset(dataset_name):
        return ground_truth

    oriented = _orient_ground_truth_by_query(ground_truth, query_count=query_count)
    if oriented.size > 0 and int(oriented.min()) >= 1:
        return oriented - 1
    return oriented


def _orient_ground_truth_by_query(ground_truth: np.ndarray, query_count: int) -> np.ndarray:
    if ground_truth.shape[0] == query_count:
        return ground_truth
    if ground_truth.shape[1] >= query_count:
        return np.ascontiguousarray(ground_truth.T[:query_count], dtype=np.int64)
    raise ValueError(
        "ground truth row count does not match query count and cannot be transposed "
        f"to match: ground_truth={ground_truth.shape}, query_count={query_count}"
    )


def _load_hdf5_queries_and_gt(path: Path, num_queries: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        missing = [key for key in ("test", "neighbors") if key not in f]
        if missing:
            raise ValueError(f"{path} is missing ANN-Benchmarks keys: {missing}")
        rows = min(num_queries, f["test"].shape[0])  # type: ignore[reportAttributeAccessIssue]
        queries = np.asarray(f["test"][:rows], dtype=np.float32)  # type: ignore[reportIndexIssue]
        ground_truth = np.asarray(f["neighbors"][:rows], dtype=np.int64)  # type: ignore[reportIndexIssue]
    return queries, ground_truth


def _load_hdf5_matrix(
    path: Path,
    subset_size: int | None,
    key: str,
    dtype: type[np.float32] | type[np.int64],
) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if key not in f:
            raise ValueError(f"{path} does not contain HDF5 key {key!r}.")
        rows = f[key].shape[0] if subset_size is None else min(subset_size, f[key].shape[0])  # type: ignore[reportAttributeAccessIssue]
        return np.asarray(f[key][:rows], dtype=dtype)  # type: ignore[reportIndexIssue]


def _load_fvecs(path: Path, subset_size: int | None) -> np.ndarray:
    dim, total = _vecs_metadata(path, value_dtype=np.float32)
    rows = total if subset_size is None else min(subset_size, total)
    raw = np.memmap(path, dtype=np.float32, mode="r", shape=(total, dim + 1))
    return raw[:rows, 1:]


def _load_ivecs(path: Path, subset_size: int | None) -> np.ndarray:
    dim, total = _vecs_metadata(path, value_dtype=np.int32)
    rows = total if subset_size is None else min(subset_size, total)
    raw = np.memmap(path, dtype=np.int32, mode="r", shape=(total, dim + 1))
    return np.asarray(raw[:rows, 1:], dtype=np.int64)


def read_fbin(
    filename: str | Path,
    start_idx: int = 0,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Read vectors from a binary float32 matrix with `[nvecs, dim]` header."""
    return _read_bin_matrix(
        Path(filename),
        value_dtype=np.dtype("<f4"),
        start_idx=start_idx,
        chunk_size=chunk_size,
    )


def read_u8bin(
    filename: str | Path,
    start_idx: int = 0,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Read vectors from a BigANN uint8 matrix with `[nvecs, dim]` header."""
    return _read_bin_matrix(
        Path(filename),
        value_dtype=np.dtype("u1"),
        start_idx=start_idx,
        chunk_size=chunk_size,
    )


def read_i8bin(
    filename: str | Path,
    start_idx: int = 0,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Read vectors from a BigANN int8 matrix with `[nvecs, dim]` header."""
    return _read_bin_matrix(
        Path(filename),
        value_dtype=np.dtype("i1"),
        start_idx=start_idx,
        chunk_size=chunk_size,
    )


def read_ibin(
    filename: str | Path,
    start_idx: int = 0,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Read vectors from a binary int32 matrix with `[nvecs, dim]` header."""
    return _read_bin_matrix(
        Path(filename),
        value_dtype=np.dtype("<i4"),
        start_idx=start_idx,
        chunk_size=chunk_size,
    )


def read_i32bin(
    filename: str | Path,
    start_idx: int = 0,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Read vectors from a BigANN int32 matrix with `[nvecs, dim]` header."""
    return _read_bin_matrix(
        Path(filename),
        value_dtype=np.dtype("<i4"),
        start_idx=start_idx,
        chunk_size=chunk_size,
    )


def write_fbin(filename: str | Path, vecs: np.ndarray) -> None:
    """Write a 2D array as `[nvecs, dim]` followed by row-major float32 values."""
    _write_bin_matrix(Path(filename), vecs, value_dtype=np.dtype("<f4"))


def write_ibin(filename: str | Path, vecs: np.ndarray) -> None:
    """Write a 2D array as `[nvecs, dim]` followed by row-major int32 values."""
    _write_bin_matrix(Path(filename), vecs, value_dtype=np.dtype("<i4"))


def _vecs_metadata(path: Path, value_dtype: type[np.float32] | type[np.int32]) -> tuple[int, int]:
    with path.open("rb") as f:
        dim_bytes = f.read(4)
    if len(dim_bytes) != 4:
        raise ValueError(f"{path} is too small to contain vecs metadata.")

    dim = int(np.frombuffer(dim_bytes, dtype=np.int32)[0])
    if dim <= 0:
        raise ValueError(f"{path} has invalid vector dimension {dim}.")

    record_bytes = 4 + dim * np.dtype(value_dtype).itemsize
    file_size = path.stat().st_size
    if file_size % record_bytes != 0:
        raise ValueError(f"{path} size is not divisible by vecs record size.")
    return dim, file_size // record_bytes


def _read_bin_matrix(
    path: Path,
    value_dtype: np.dtype,
    start_idx: int,
    chunk_size: int | None,
) -> np.ndarray:
    if start_idx < 0:
        raise ValueError("start_idx must be non-negative.")
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be positive when provided.")

    nvecs, dim = _bin_metadata(path, value_dtype)
    if start_idx > nvecs:
        raise ValueError(f"start_idx={start_idx} is past the end of {path} ({nvecs} rows).")

    rows_available = nvecs - start_idx
    rows = rows_available if chunk_size is None else min(chunk_size, rows_available)
    raw = np.memmap(
        path,
        dtype=value_dtype,
        mode="r",
        offset=8,
        shape=(nvecs, dim),
    )
    return raw[start_idx : start_idx + rows]


def _write_bin_matrix(
    path: Path,
    vecs: np.ndarray,
    value_dtype: np.dtype,
) -> None:
    if vecs.ndim != 2:
        raise ValueError("Input array must have 2 dimensions.")

    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(vecs, dtype=value_dtype)
    nvecs, dim = matrix.shape
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with tmp_path.open("wb") as f:
            np.asarray([nvecs, dim], dtype=np.dtype("<i4")).tofile(f)
            matrix.tofile(f)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _bin_metadata(path: Path, value_dtype: np.dtype) -> tuple[int, int]:
    with path.open("rb") as f:
        nvecs, dim = _read_bin_header(f, path)

    itemsize = np.dtype(value_dtype).itemsize
    expected_size = 8 + nvecs * dim * itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{path} size does not match fbin/ibin header: "
            f"expected {expected_size} bytes, found {actual_size}."
        )
    return nvecs, dim


def _read_bin_header(f: BinaryIO, path: Path) -> tuple[int, int]:
    header = f.read(8)
    if len(header) != 8:
        raise ValueError(f"{path} is too small to contain fbin/ibin metadata.")

    nvecs, dim = np.frombuffer(header, dtype=np.dtype("<i4"), count=2)
    if nvecs < 0:
        raise ValueError(f"{path} has invalid vector count {int(nvecs)}.")
    if dim <= 0:
        raise ValueError(f"{path} has invalid vector dimension {int(dim)}.")
    return int(nvecs), int(dim)
