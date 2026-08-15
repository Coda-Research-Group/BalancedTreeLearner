import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path("experiments/scripts/bigann").resolve()))

import fetch_bigann_100m_base as fetch_mod


def _write_u8bin(path: Path, vectors: np.ndarray) -> None:
    nvecs, dim = vectors.shape
    with path.open("wb") as f:
        f.write(struct.pack("<ii", nvecs, dim))
        f.write(vectors.tobytes())


def test_copy_slice_extracts_prefix_vectors_and_rewrites_header(tmp_path: Path) -> None:
    dim = 8
    total_source_vectors = 50
    num_vectors = 20

    rng = np.random.default_rng(42)
    source_vectors = rng.integers(0, 256, size=(total_source_vectors, dim), dtype=np.uint8)
    source_path = tmp_path / "base.1B.u8bin"
    _write_u8bin(source_path, source_vectors)

    handle = fetch_mod.SourceHandle(str(source_path))
    src_nvecs, src_dim = fetch_mod.read_source_header(handle)
    assert (src_nvecs, src_dim) == (total_source_vectors, dim)

    output_path = tmp_path / "base.100M.u8bin"
    fetch_mod.copy_slice(
        handle,
        num_vectors=num_vectors,
        dim=dim,
        output_path=output_path,
        chunk_size=13,  # deliberately not a multiple of dim, to exercise chunk boundaries
        progress=False,
    )

    fetch_mod.verify_output(output_path, num_vectors=num_vectors, dim=dim)

    with output_path.open("rb") as f:
        nvecs, dim_written = struct.unpack("<ii", f.read(8))
        payload = np.frombuffer(f.read(), dtype=np.uint8).reshape(nvecs, dim_written)

    assert (nvecs, dim_written) == (num_vectors, dim)
    np.testing.assert_array_equal(payload, source_vectors[:num_vectors])


def test_copy_slice_resumes_from_partial_temp_file(tmp_path: Path) -> None:
    dim = 4
    num_vectors = 10

    rng = np.random.default_rng(7)
    source_vectors = rng.integers(0, 256, size=(num_vectors, dim), dtype=np.uint8)
    source_path = tmp_path / "base.1B.u8bin"
    _write_u8bin(source_path, source_vectors)

    output_path = tmp_path / "base.100M.u8bin"
    tmp_path_file = output_path.with_suffix(output_path.suffix + ".part")
    # partial_bytes = 6 * dim  # first 6 of 10 vectors already "downloaded"
    with tmp_path_file.open("wb") as f:
        f.write(struct.pack("<ii", num_vectors, dim))
        f.write(source_vectors[:6].tobytes())

    handle = fetch_mod.SourceHandle(str(source_path))
    fetch_mod.copy_slice(
        handle,
        num_vectors=num_vectors,
        dim=dim,
        output_path=output_path,
        progress=False,
    )

    fetch_mod.verify_output(output_path, num_vectors=num_vectors, dim=dim)
    with output_path.open("rb") as f:
        f.read(8)
        payload = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_vectors, dim)
    np.testing.assert_array_equal(payload, source_vectors)


def test_verify_output_rejects_size_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad.u8bin"
    with path.open("wb") as f:
        f.write(struct.pack("<ii", 10, 4))
        f.write(b"\x00" * (10 * 4 - 1))  # one byte short

    with pytest.raises(ValueError, match="size mismatch"):
        fetch_mod.verify_output(path, num_vectors=10, dim=4)
