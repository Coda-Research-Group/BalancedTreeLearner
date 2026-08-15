"""Convert a LAION HDF5 embedding matrix to a manifest-backed NumPy memmap."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

_HASH_CHUNK_BYTES = 16 * 1024 * 1024


def hash_file(
    path: Path,
    algorithm: str,
    chunk_bytes: int = _HASH_CHUNK_BYTES,
) -> str:
    """Return a streaming digest without loading ``path`` into memory."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive.")
    digest = hashlib.new(algorithm, usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source(
    source: Path,
    *,
    key: str,
    expected_md5: str,
    expected_shape: tuple[int, int],
) -> None:
    actual_md5 = hash_file(source, "md5")
    if actual_md5 != expected_md5:
        raise ValueError(f"source MD5 mismatch: expected {expected_md5}, got {actual_md5}.")
    with h5py.File(source, "r") as handle:
        if key not in handle:
            raise ValueError(f"source is missing HDF5 dataset {key!r}.")
        dataset = handle[key]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"source HDF5 object {key!r} is not a dataset.")
        if tuple(dataset.shape) != expected_shape:
            raise ValueError(
                f"source shape mismatch: expected {expected_shape}, got {dataset.shape}."
            )
        if np.dtype(dataset.dtype) != np.dtype(np.float16):
            raise ValueError(f"source dtype mismatch: expected float16, got {dataset.dtype}.")


def _load_manifest(manifest: Path) -> dict[str, Any]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"{manifest}: unsupported or missing manifest schema.")
    if not isinstance(payload.get("source"), dict) or not isinstance(payload.get("output"), dict):
        raise ValueError(f"{manifest}: incomplete source/output metadata.")
    return payload


def verify(
    output: Path,
    manifest: Path,
    *,
    expected_source_md5: str,
    expected_shape: tuple[int, int],
    verify_sha256: bool = True,
) -> dict[str, Any]:
    """Validate a converted memmap and return its manifest."""
    payload = _load_manifest(manifest)
    source_metadata = payload["source"]
    output_metadata = payload["output"]
    if source_metadata.get("md5") != expected_source_md5:
        raise ValueError(
            "manifest source MD5 mismatch: "
            f"expected {expected_source_md5}, got {source_metadata.get('md5')}."
        )
    if source_metadata.get("shape") != list(expected_shape):
        raise ValueError(
            "manifest source shape mismatch: "
            f"expected {list(expected_shape)}, got {source_metadata.get('shape')}."
        )
    if source_metadata.get("dtype") != "float16":
        raise ValueError("manifest source dtype must be float16.")

    converted = np.load(output, mmap_mode="r", allow_pickle=False)
    if not isinstance(converted, np.memmap):
        raise ValueError(f"{output}: expected a NumPy memmap.")
    if tuple(converted.shape) != expected_shape:
        raise ValueError(
            f"output shape mismatch: expected {expected_shape}, got {converted.shape}."
        )
    if converted.dtype != np.dtype(np.float32):
        raise ValueError(f"output dtype mismatch: expected float32, got {converted.dtype}.")
    if output_metadata.get("shape") != list(expected_shape):
        raise ValueError("manifest output shape does not match the expected shape.")
    if output_metadata.get("dtype") != "float32":
        raise ValueError("manifest output dtype must be float32.")
    expected_nbytes = int(np.prod(expected_shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    if output_metadata.get("data_nbytes") != expected_nbytes:
        raise ValueError("manifest output data_nbytes is inconsistent.")
    del converted

    if verify_sha256:
        actual_sha256 = hash_file(output, "sha256")
        if output_metadata.get("sha256") != actual_sha256:
            raise ValueError(
                "output SHA-256 mismatch: "
                f"expected {output_metadata.get('sha256')}, got {actual_sha256}."
            )
    return payload


def convert(
    source: Path,
    output: Path,
    manifest: Path,
    *,
    key: str,
    expected_md5: str,
    expected_shape: tuple[int, int],
    chunk_rows: int,
    pbs_job_id: str,
    git_commit: str,
) -> dict[str, Any]:
    """Convert one exact float16 HDF5 matrix to float32 `.npy` in chunks."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive.")
    existing = [path for path in (output, manifest) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifact(s): {existing}")
    _validate_source(
        source,
        key=key,
        expected_md5=expected_md5,
        expected_shape=expected_shape,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest.with_name(f"{manifest.name}.tmp")
    try:
        converted = np.lib.format.open_memmap(
            output,
            mode="w+",
            dtype=np.float32,
            shape=expected_shape,
        )
        with h5py.File(source, "r") as handle:
            dataset = handle[key]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"source HDF5 object {key!r} is not a dataset.")
            for start in range(0, expected_shape[0], chunk_rows):
                end = min(start + chunk_rows, expected_shape[0])
                converted[start:end] = dataset[start:end]
                print(f"converted_rows={end}/{expected_shape[0]}", flush=True)
        converted.flush()
        del converted

        output_sha256 = hash_file(output, "sha256")
        payload: dict[str, Any] = {
            "schema_version": 1,
            "source": {
                "path": str(source),
                "md5": expected_md5,
                "key": key,
                "shape": list(expected_shape),
                "dtype": "float16",
            },
            "output": {
                "filename": output.name,
                "sha256": output_sha256,
                "shape": list(expected_shape),
                "dtype": "float32",
                "data_nbytes": (
                    int(np.prod(expected_shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
                ),
            },
            "provenance": {
                "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "pbs_job_id": pbs_job_id,
                "git_commit": git_commit,
            },
        }
        temporary_manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest)
        return verify(
            output,
            manifest,
            expected_source_md5=expected_md5,
            expected_shape=expected_shape,
        )
    except Exception:
        output.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        raise


def _shape(values: list[int]) -> tuple[int, int]:
    if len(values) != 2 or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("shape requires two positive integers")
    return values[0], values[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--source", type=Path, required=True)
    convert_parser.add_argument("--output", type=Path, required=True)
    convert_parser.add_argument("--manifest", type=Path, required=True)
    convert_parser.add_argument("--key", default="emb")
    convert_parser.add_argument("--expected-md5", required=True)
    convert_parser.add_argument("--expected-shape", nargs=2, type=int, required=True)
    convert_parser.add_argument("--chunk-rows", type=int, required=True)
    convert_parser.add_argument("--pbs-job-id", required=True)
    convert_parser.add_argument("--git-commit", required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output", type=Path, required=True)
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--expected-source-md5", required=True)
    verify_parser.add_argument("--expected-shape", nargs=2, type=int, required=True)
    verify_parser.add_argument("--skip-sha256", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    expected_shape = _shape(args.expected_shape)
    if args.command == "convert":
        payload = convert(
            args.source,
            args.output,
            args.manifest,
            key=args.key,
            expected_md5=args.expected_md5,
            expected_shape=expected_shape,
            chunk_rows=args.chunk_rows,
            pbs_job_id=args.pbs_job_id,
            git_commit=args.git_commit,
        )
    else:
        payload = verify(
            args.output,
            args.manifest,
            expected_source_md5=args.expected_source_md5,
            expected_shape=expected_shape,
            verify_sha256=not args.skip_sha256,
        )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
