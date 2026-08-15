"""Balanced k-ary tree data structure for BATL."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BATLTree:
    K: int
    H: int
    alpha: float
    N: int
    paths: np.ndarray
    _leaf_buckets: dict[tuple[int, ...], np.ndarray] = field(init=False, repr=False)
    _leaf_buckets_by_id: dict[int, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate()
        self.paths = np.asarray(self.paths, dtype=np.uint16)
        self._leaf_buckets, self._leaf_buckets_by_id = self._build_leaf_buckets()

    @classmethod
    def random_init(cls, N: int, K: int, H: int, alpha: float, seed: int) -> BATLTree:
        """Create a balanced random assignment of vectors to leaves."""
        num_leaves = K**H
        leaf_ids = np.arange(N, dtype=np.int64) % num_leaves
        rng = np.random.default_rng(seed)
        shuffled_leaf_ids = leaf_ids[rng.permutation(N)]
        paths = leaf_id_to_path(shuffled_leaf_ids, K=K, H=H)
        return cls(K=K, H=H, alpha=alpha, N=N, paths=paths)

    def get_leaf_indices(self, path: tuple[int, ...]) -> np.ndarray:
        """Return database indices assigned to a leaf path."""
        self._validate_path_tuple(path)
        return self._get_leaf_indices_by_id(path_to_leaf_id(path, self.K))

    def _get_leaf_indices_by_id(self, leaf_id: int) -> np.ndarray:
        """Return database indices assigned to a numeric leaf ID."""
        if leaf_id < 0 or leaf_id >= self.K**self.H:
            raise ValueError("leaf_id must be in [0, K**H).")
        self._ensure_leaf_id_buckets()
        return self._leaf_buckets_by_id.get(leaf_id, np.empty(0, dtype=np.int32))

    def leaf_size_stats(self) -> dict[str, float | int]:
        """Return balance diagnostics over all leaves, including empty leaves."""
        sizes = np.zeros(self.K**self.H, dtype=np.int64)
        for leaf_id, indices in self._leaf_buckets_by_id.items():
            sizes[leaf_id] = len(indices)

        mean = float(sizes.mean()) if sizes.size else 0.0
        return {
            "mean": mean,
            "std": float(sizes.std()),
            "min": int(sizes.min()) if sizes.size else 0,
            "max": int(sizes.max()) if sizes.size else 0,
            "gini": float(_gini(sizes)),
            "num_empty": int(np.count_nonzero(sizes == 0)),
        }

    def save(self, path: str) -> None:
        """Persist this tree using pickle."""
        payload = {
            "K": self.K,
            "H": self.H,
            "alpha": self.alpha,
            "N": self.N,
            "paths": self.paths,
        }
        with Path(path).open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> BATLTree:
        """Load a tree saved with ``save``.

        Only load files written by this project's own ``save`` method.
        pickle.load executes arbitrary code — never load untrusted files.
        """
        with Path(path).open("rb") as f:
            payload: dict[str, Any] = pickle.load(f)
        return cls(**payload)

    def _build_leaf_buckets(
        self,
    ) -> tuple[dict[tuple[int, ...], np.ndarray], dict[int, np.ndarray]]:
        if self.N == 0:
            return {}, {}

        powers = (self.K ** np.arange(self.H - 1, -1, -1, dtype=np.int64)).astype(np.int64)
        leaf_ids = self.paths.astype(np.int64) @ powers
        order = np.argsort(leaf_ids, kind="stable")
        sorted_leaf_ids = leaf_ids[order]
        boundaries = np.flatnonzero(np.diff(sorted_leaf_ids)) + 1
        groups = np.split(order.astype(np.int32, copy=False), boundaries)
        unique_leaf_ids = sorted_leaf_ids[np.r_[0, boundaries]]
        unique_paths = leaf_id_to_path(unique_leaf_ids, self.K, self.H)

        by_path = {
            tuple(int(part) for part in path): group
            for path, group in zip(unique_paths, groups, strict=True)
        }
        by_id = {
            int(leaf_id): group for leaf_id, group in zip(unique_leaf_ids, groups, strict=True)
        }
        return by_path, by_id

    def _ensure_leaf_id_buckets(self) -> None:
        if "_leaf_buckets_by_id" in self.__dict__:
            return
        if "_leaf_buckets" in self.__dict__:
            self._leaf_buckets_by_id = {
                path_to_leaf_id(path, self.K): indices
                for path, indices in self._leaf_buckets.items()
            }
            return
        self._leaf_buckets, self._leaf_buckets_by_id = self._build_leaf_buckets()

    def _validate(self) -> None:
        paths = np.asarray(self.paths)
        if paths.shape != (self.N, self.H):
            raise ValueError(f"paths must have shape ({self.N}, {self.H}), got {paths.shape}")
        if np.any(paths < 0) or np.any(paths >= self.K):
            raise ValueError("paths must contain branch IDs in [0, K).")
        if self.K > np.iinfo(np.uint16).max + 1:
            raise ValueError("uint16 path storage requires K <= 65536.")

    def _validate_path_tuple(self, path: tuple[int, ...]) -> None:
        if len(path) != self.H:
            raise ValueError(f"path must have length {self.H}, got {len(path)}")
        if any(branch < 0 or branch >= self.K for branch in path):
            raise ValueError("path must contain branch IDs in [0, K).")


def leaf_id_to_path(leaf_ids: np.ndarray, K: int, H: int) -> np.ndarray:
    """Convert leaf IDs to base-K path rows."""
    if K <= 0 or H <= 0:
        raise ValueError("K and H must be positive.")
    ids = np.asarray(leaf_ids, dtype=np.int64).copy()
    if np.any(ids < 0) or np.any(ids >= K**H):
        raise ValueError("leaf_ids must be in [0, K**H).")

    paths = np.empty((ids.size, H), dtype=np.uint16)
    for h in range(H - 1, -1, -1):
        paths[:, h] = (ids % K).astype(np.uint16)
        ids //= K
    return paths


def path_to_leaf_id(path: tuple[int, ...] | np.ndarray, K: int) -> int:
    """Convert a base-K path to its leaf ID."""
    if K <= 0:
        raise ValueError("K must be positive.")
    leaf_id = 0
    for branch in path:
        value = int(branch)
        if value < 0 or value >= K:
            raise ValueError("path must contain branch IDs in [0, K).")
        leaf_id = leaf_id * K + value
    return leaf_id


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or values.mean() == 0.0:
        return 0.0
    n = values.size
    sorted_v = np.sort(values)
    cumsum = np.cumsum(sorted_v)
    # Standard sorted-array identity: O(n log n) time, O(n) memory.
    # Equivalent to the mean-absolute-difference formula but avoids O(n^2) outer product.
    return float((n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n)
