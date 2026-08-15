#!/usr/bin/env python3
"""Fetch the first N vectors of the ANN_SIFT1B (BigANN) base set as a u8bin file.

The full base set (`base.1B.u8bin`, ~1e9 x 128 uint8 vectors) is far too large to
download in full when only a 100M-vector subset is needed. This tool reads just
the byte range covering the first `--num-vectors` vectors -- via HTTP Range
requests when the source is a URL, or a plain seek when it's a local file -- and
writes them out with a corrected `[nvecs, dim]` u8bin header, matching the format
`batl.utils.data.read_u8bin` / `_bin_metadata` expect.

Per the ANN_SIFT1B README, ground truth for the 100M subset was computed against
these base vectors specifically -- do not substitute `learn.100M.u8bin`.

Usage:
    python fetch_bigann_100m_base.py \\
        --source https://comp21storage.blob.core.windows.net/publiccontainer/comp21/bigann/base.1B.u8bin \\
        --output /storage/brno2/home/jozefsprlak/repos/data/bigann/base.100M.u8bin

    # Or slice from an already-downloaded full base file sitting on shared storage:
    python fetch_bigann_100m_base.py \\
        --source /path/to/base.1B.u8bin \\
        --output ./base.100M.u8bin
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

HEADER_SIZE = 8
DEFAULT_NUM_VECTORS = 100_000_000
DEFAULT_DIM = 128
DEFAULT_CHUNK_SIZE = 64 * 1024 * 1024
DEFAULT_MAX_RETRIES = 5


class SourceHandle:
    """Uniform range-read interface over an HTTP(S) URL or a local file."""

    def __init__(self, source: str):
        self.source = source
        self.is_url = source.startswith("http://") or source.startswith("https://")
        self.size = self._probe_size()

    def _probe_size(self) -> int:
        if self.is_url:
            req = urllib.request.Request(self.source, method="HEAD")
            with urllib.request.urlopen(req, timeout=30) as resp:
                length = resp.headers.get("Content-Length")
                if length is None:
                    raise ValueError(f"{self.source} did not report Content-Length.")
                return int(length)
        return Path(self.source).stat().st_size

    def read_range(self, start: int, length: int, *, chunk_size: int = DEFAULT_CHUNK_SIZE):
        """Yield byte chunks covering [start, start + length)."""
        if length <= 0:
            return
        end_inclusive = start + length - 1
        if self.is_url:
            yield from self._read_range_http(start, end_inclusive, chunk_size)
        else:
            with open(self.source, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    data = f.read(min(chunk_size, remaining))
                    if not data:
                        raise EOFError(f"{self.source} ended before expected offset.")
                    remaining -= len(data)
                    yield data

    def _read_range_http(self, start: int, end_inclusive: int, chunk_size: int):
        req = urllib.request.Request(self.source)
        req.add_header("Range", f"bytes={start}-{end_inclusive}")
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status not in (200, 206):
                raise ValueError(f"Unexpected HTTP status {resp.status} for {self.source}")
            while True:
                data = resp.read(chunk_size)
                if not data:
                    break
                yield data


def read_source_header(handle: SourceHandle) -> tuple[int, int]:
    header = b"".join(handle.read_range(0, HEADER_SIZE))
    if len(header) != HEADER_SIZE:
        raise ValueError(f"{handle.source} is too small to contain a u8bin header.")
    nvecs, dim = struct.unpack("<ii", header)
    return nvecs, dim


def copy_slice(
    handle: SourceHandle,
    *,
    num_vectors: int,
    dim: int,
    output_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    progress: bool = True,
) -> None:
    """Write the first `num_vectors` vectors from `handle` to `output_path` as u8bin."""
    payload_bytes = num_vectors * dim
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    if tmp_path.exists():
        on_disk_payload = max(0, tmp_path.stat().st_size - HEADER_SIZE)
        written = min(on_disk_payload, payload_bytes)

    mode = "r+b" if written else "wb"
    with open(tmp_path, mode) as out:
        out.seek(0)
        out.write(struct.pack("<ii", num_vectors, dim))
        out.seek(HEADER_SIZE + written)

        remaining = payload_bytes - written
        attempt = 0
        while remaining > 0:
            try:
                for chunk in handle.read_range(
                    HEADER_SIZE + written, remaining, chunk_size=chunk_size
                ):
                    out.write(chunk)
                    written += len(chunk)
                    remaining -= len(chunk)
                    if progress:
                        _print_progress(written, payload_bytes)
                attempt = 0
            except (URLError, HTTPError, TimeoutError, ConnectionError, EOFError) as exc:
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(
                        f"Giving up after {max_retries} retries at byte {written}/{payload_bytes}"
                    ) from exc
                backoff = min(2**attempt, 60)
                print(
                    f"\nTransfer error ({exc}); retrying from byte {written} "
                    f"in {backoff}s (attempt {attempt}/{max_retries})",
                    file=sys.stderr,
                )
                time.sleep(backoff)
                out.seek(HEADER_SIZE + written)

    if progress:
        print(file=sys.stderr)
    tmp_path.rename(output_path)


def _print_progress(written: int, total: int) -> None:
    pct = 100.0 * written / total if total else 100.0
    print(
        f"\r  {written / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:.1f}%)",
        end="",
        file=sys.stderr,
        flush=True,
    )


def verify_output(path: Path, *, num_vectors: int, dim: int) -> None:
    expected_size = HEADER_SIZE + num_vectors * dim
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{path} size mismatch: expected {expected_size} bytes, found {actual_size}."
        )
    with path.open("rb") as f:
        nvecs, dim_written = struct.unpack("<ii", f.read(HEADER_SIZE))
    if (nvecs, dim_written) != (num_vectors, dim):
        raise ValueError(
            f"{path} header mismatch: expected ({num_vectors}, {dim}), found ({nvecs}, {dim_written})."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        required=True,
        help="URL (http/https) or local path to the full ANN_SIFT1B base.1B.u8bin file.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Destination .u8bin path.")
    parser.add_argument("--num-vectors", type=int, default=DEFAULT_NUM_VECTORS)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if a valid --output already exists."
    )
    args = parser.parse_args(argv)

    if args.output.exists() and not args.force:
        try:
            verify_output(args.output, num_vectors=args.num_vectors, dim=args.dim)
            print(f"{args.output} already matches the requested slice; nothing to do.")
            return 0
        except ValueError:
            pass  # existing file doesn't match; re-slice below

    handle = SourceHandle(args.source)
    src_nvecs, src_dim = read_source_header(handle)
    print(
        f"Source {args.source}: {src_nvecs:,} vectors x {src_dim} dims ({handle.size / 1e9:.1f} GB)"
    )

    if src_dim != args.dim:
        raise SystemExit(f"Source dim {src_dim} does not match --dim {args.dim}.")
    if src_nvecs < args.num_vectors:
        raise SystemExit(f"Source only has {src_nvecs:,} vectors; requested {args.num_vectors:,}.")

    payload_bytes = args.num_vectors * args.dim
    if HEADER_SIZE + payload_bytes > handle.size:
        raise SystemExit("Requested slice extends past the end of the source file.")

    print(
        f"Writing first {args.num_vectors:,} vectors ({payload_bytes / 1e9:.1f} GB payload) to {args.output}"
    )
    copy_slice(
        handle,
        num_vectors=args.num_vectors,
        dim=args.dim,
        output_path=args.output,
        chunk_size=args.chunk_size,
        max_retries=args.max_retries,
    )
    verify_output(args.output, num_vectors=args.num_vectors, dim=args.dim)
    print(f"Done: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
