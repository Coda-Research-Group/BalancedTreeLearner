"""Convert the YandexTI-100M ground truth file to the plain ibin layout BATL reads.

The file at GT_100M/text2image-100M (dl.fbaipublicfiles.com) is NOT a plain
ibin: it's `[n, k] (uint32) + ids (n*k int32) + distances (n*k float32)`, the
big-ann-benchmarks competition's `knn_result_read` layout. batl.utils.data's
read_ibin/_bin_metadata expects `[n, k] (int32) + values (n*k int32)` only and
will raise a size-mismatch ValueError on the raw file. This drops the
distances block and rewrites the header/ids in that plain layout.

Usage:
    python convert_yandexti_100m_groundtruth.py text2image-100M groundtruth.100M.ibin
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def convert(src: Path, dst: Path) -> None:
    n, k = np.fromfile(src, dtype="<u4", count=2).astype(np.int64)
    expected_size = 8 + n * k * (4 + 4)
    actual_size = src.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{src} does not match the knn_result_read layout: "
            f"expected {expected_size} bytes for n={n}, k={k}, found {actual_size}."
        )

    with src.open("rb") as f:
        f.seek(8)
        ids = np.fromfile(f, dtype="<i4", count=n * k).reshape(n, k)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    with tmp.open("wb") as f:
        np.asarray([n, k], dtype="<i4").tofile(f)
        ids.tofile(f)
    tmp.replace(dst)
    print(f"Wrote {dst}: n={n}, k={k}, {dst.stat().st_size} bytes")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {sys.argv[0]} <src text2image-100M> <dst groundtruth.100M.ibin>")
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
