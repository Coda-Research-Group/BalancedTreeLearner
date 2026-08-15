from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.utils.prepare_laion_memmap import convert, verify


def _write_hdf5(path: Path, values: np.ndarray, *, key: str = "emb") -> str:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=values)
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def _convert_fixture(tmp_path: Path) -> tuple[Path, Path, str, np.ndarray]:
    values = np.arange(18, dtype=np.float16).reshape(6, 3)
    source = tmp_path / "source.h5"
    source_md5 = _write_hdf5(source, values)
    output = tmp_path / "output.npy"
    manifest = tmp_path / "output.manifest.json"
    convert(
        source,
        output,
        manifest,
        key="emb",
        expected_md5=source_md5,
        expected_shape=values.shape,
        chunk_rows=2,
        pbs_job_id="123.server",
        git_commit="abc123",
    )
    return output, manifest, source_md5, values


def test_convert_writes_float32_memmap_and_complete_manifest(tmp_path: Path) -> None:
    output, manifest, source_md5, values = _convert_fixture(tmp_path)

    converted = np.load(output, mmap_mode="r")
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert isinstance(converted, np.memmap)
    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, values.astype(np.float32))
    assert payload["schema_version"] == 1
    assert payload["source"]["md5"] == source_md5
    assert payload["source"]["key"] == "emb"
    assert payload["source"]["shape"] == [6, 3]
    assert payload["source"]["dtype"] == "float16"
    assert payload["output"]["filename"] == output.name
    assert payload["output"]["shape"] == [6, 3]
    assert payload["output"]["dtype"] == "float32"
    assert payload["output"]["data_nbytes"] == values.size * np.dtype(np.float32).itemsize
    assert len(payload["output"]["sha256"]) == 64
    assert payload["provenance"]["pbs_job_id"] == "123.server"
    assert payload["provenance"]["git_commit"] == "abc123"
    assert payload["provenance"]["created_utc"].endswith("Z")
    assert (
        verify(
            output,
            manifest,
            expected_source_md5=source_md5,
            expected_shape=values.shape,
        )
        == payload
    )


def test_convert_rejects_wrong_source_md5(tmp_path: Path) -> None:
    values = np.zeros((4, 2), dtype=np.float16)
    source = tmp_path / "source.h5"
    _write_hdf5(source, values)

    with pytest.raises(ValueError, match="source MD5"):
        convert(
            source,
            tmp_path / "output.npy",
            tmp_path / "manifest.json",
            key="emb",
            expected_md5="0" * 32,
            expected_shape=values.shape,
            chunk_rows=2,
            pbs_job_id="job",
            git_commit="commit",
        )


@pytest.mark.parametrize(
    ("values", "expected_shape", "message"),
    [
        (np.zeros((4, 2), dtype=np.float16), (5, 2), "source shape"),
        (np.zeros((4, 2), dtype=np.float32), (4, 2), "source dtype"),
    ],
)
def test_convert_rejects_wrong_source_layout(
    tmp_path: Path,
    values: np.ndarray,
    expected_shape: tuple[int, int],
    message: str,
) -> None:
    source = tmp_path / "source.h5"
    source_md5 = _write_hdf5(source, values)

    with pytest.raises(ValueError, match=message):
        convert(
            source,
            tmp_path / "output.npy",
            tmp_path / "manifest.json",
            key="emb",
            expected_md5=source_md5,
            expected_shape=expected_shape,
            chunk_rows=2,
            pbs_job_id="job",
            git_commit="commit",
        )


@pytest.mark.parametrize("existing_name", ["output.npy", "manifest.json"])
def test_convert_refuses_existing_artifact(tmp_path: Path, existing_name: str) -> None:
    values = np.zeros((4, 2), dtype=np.float16)
    source = tmp_path / "source.h5"
    source_md5 = _write_hdf5(source, values)
    (tmp_path / existing_name).write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        convert(
            source,
            tmp_path / "output.npy",
            tmp_path / "manifest.json",
            key="emb",
            expected_md5=source_md5,
            expected_shape=values.shape,
            chunk_rows=2,
            pbs_job_id="job",
            git_commit="commit",
        )


def test_verify_rejects_corrupted_output(tmp_path: Path) -> None:
    output, manifest, source_md5, values = _convert_fixture(tmp_path)
    with output.open("r+b") as handle:
        handle.seek(-1, 2)
        final_byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([final_byte[0] ^ 0xFF]))

    with pytest.raises(ValueError, match="output SHA-256"):
        verify(
            output,
            manifest,
            expected_source_md5=source_md5,
            expected_shape=values.shape,
        )
