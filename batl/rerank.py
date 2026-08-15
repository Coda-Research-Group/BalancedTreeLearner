"""GPU-resident exact reranking for BATL candidate sets.

The default ``numpy_cpu`` rerank path gathers ``database[candidates]`` on the
host once per query. At Deep100M with M=100 leaves that is roughly 125k
scattered rows (~48 MB) read from a memmap per query, which dominates search
wall-clock (see ``docs/performance_gap_analysis.md`` §2.1).

``ResidentGpuReranker`` uploads the database to the GPU once per process and
reranks a whole query chunk with batched matrix products, so the per-query
scattered host read disappears entirely.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch

from batl.constants import (
    DEFAULT_MAX_GATHER_BYTES,
    DEFAULT_UPLOAD_CHUNK_ROWS,
    DEFAULT_VRAM_HEADROOM_BYTES,
    GATHER_FRACTION_OF_FREE_VRAM,
    MIN_HEALTHY_MICRO_BATCH_ROWS,
)

logger = logging.getLogger(__name__)


class RerankGpuMemoryError(RuntimeError):
    """Raised when the database does not fit in the device's free memory."""


@contextmanager
def _full_fp32_matmul() -> Iterator[None]:
    """Disable TF32 for the duration of a rerank.

    ``set_seed`` enables TF32 globally for training throughput. TF32 keeps only
    a 10-bit mantissa, which reorders near-equal candidate distances; exact
    rerank is the one place in the pipeline where that ordering is the result.
    """
    if not torch.cuda.is_available():
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def resident_rerank_bytes(n_vectors: int, dim: int) -> int:
    """Return device bytes needed to hold the database plus its squared norms."""
    return n_vectors * dim * 4 + n_vectors * 4


def check_device_capacity(
    *,
    n_vectors: int,
    dim: int,
    device: torch.device,
    headroom_bytes: int = DEFAULT_VRAM_HEADROOM_BYTES,
) -> None:
    """Raise ``RerankGpuMemoryError`` if the resident database will not fit.

    Non-CUDA devices are not budgeted: host memory is the operating system's
    problem and there is no ``mem_get_info`` equivalent to consult.
    """
    if device.type != "cuda":
        return
    required = resident_rerank_bytes(n_vectors, dim) + headroom_bytes
    free_bytes, _total = torch.cuda.mem_get_info(device)
    if required > free_bytes:
        raise RerankGpuMemoryError(
            f"resident rerank needs {required / 1024**3:.2f} GiB "
            f"({n_vectors} x {dim} float32 + norms + "
            f"{headroom_bytes / 1024**3:.2f} GiB headroom) but only "
            f"{free_bytes / 1024**3:.2f} GiB is free on {device}."
        )


class ResidentGpuReranker:
    """Hold a database on-device and rerank padded candidate matrices.

    Construct once per process and reuse across a ``num_leaves`` sweep; the
    upload is the expensive part and is equivalent to index-load time for
    baseline methods.
    """

    def __init__(
        self,
        database: np.ndarray,
        device: torch.device | str,
        upload_chunk_rows: int = DEFAULT_UPLOAD_CHUNK_ROWS,
        vram_headroom_bytes: int = DEFAULT_VRAM_HEADROOM_BYTES,
        max_gather_bytes: int = DEFAULT_MAX_GATHER_BYTES,
    ) -> None:
        if upload_chunk_rows <= 0:
            raise ValueError("upload_chunk_rows must be positive.")
        if max_gather_bytes <= 0:
            raise ValueError("max_gather_bytes must be positive.")
        if database.ndim != 2:
            raise ValueError("database must be a 2D (n_vectors, dim) matrix.")

        self.device = torch.device(device)
        self.n_vectors, self.dim = (int(database.shape[0]), int(database.shape[1]))

        check_device_capacity(
            n_vectors=self.n_vectors,
            dim=self.dim,
            device=self.device,
            headroom_bytes=vram_headroom_bytes,
        )

        # Allocate once and fill in chunks: a memmap database must never be
        # materialized as a single float32 host copy.
        self.db = torch.empty((self.n_vectors, self.dim), dtype=torch.float32, device=self.device)
        self.db_sqnorms = torch.empty(self.n_vectors, dtype=torch.float32, device=self.device)
        for start in range(0, self.n_vectors, upload_chunk_rows):
            end = min(start + upload_chunk_rows, self.n_vectors)
            # np.ascontiguousarray returns the memmap itself when it is already
            # contiguous float32, and torch warns that the buffer is read-only.
            # The upload copies regardless, so ask for a writable array and the
            # warning goes away without an extra copy in the common case.
            chunk = np.array(database[start:end], dtype=np.float32, copy=True, order="C")
            block = torch.from_numpy(chunk).to(self.device, non_blocking=False)
            self.db[start:end] = block
            self.db_sqnorms[start:end] = block.square().sum(dim=1)

        # Only measurable once the database is actually resident.
        self.max_gather_bytes = self._resolve_gather_budget(max_gather_bytes)

    def _resolve_gather_budget(self, requested: int) -> int:
        """Clamp the gather budget to memory that exists after the upload.

        ``check_device_capacity`` reserves ``vram_headroom_bytes`` and this
        budget was sized independently of it, so on a tight fit the gather can
        claim headroom the check reserved for everything else. Deep100M on a
        40 GB card is exactly that case: the check passes with 1.5 GiB spare
        while the default budget asks for 2 GiB.
        """
        if self.device.type != "cuda":
            return requested
        free_bytes, _total = torch.cuda.mem_get_info(self.device)
        affordable = int(free_bytes * GATHER_FRACTION_OF_FREE_VRAM)
        resolved = max(1, min(requested, affordable))
        if resolved < requested:
            logger.warning(
                "rerank gather budget clamped to %.0f MiB of the requested %.0f MiB "
                "(%.2f GiB free after upload). The query micro-batch shrinks with it, "
                "and a small enough micro-batch serializes the GPU path badly enough "
                "for numpy_cpu to win.",
                resolved / 1024**2,
                requested / 1024**2,
                free_bytes / 1024**3,
            )
        return resolved

    def _micro_batch_rows(self, candidates_per_query: int) -> int:
        """Return how many queries can be gathered at once under the byte cap.

        This is the whole performance story of the resident path. The gather is
        the only batched work; if the budget divided by the per-query cost lands
        near 1, every query becomes its own set of tiny kernels and the CPU path
        wins outright. ``candidates_per_query`` is the *padded* matrix width, so
        one long row shrinks the batch for the entire chunk.
        """
        per_query_bytes = max(1, candidates_per_query * self.dim * 4)
        rows = max(1, self.max_gather_bytes // per_query_bytes)
        if rows < MIN_HEALTHY_MICRO_BATCH_ROWS:
            logger.warning(
                "rerank micro-batch is %d quer%s (%.0f MiB budget / %.0f MiB per query "
                "at %d padded candidates). Below ~%d the resident path is launch-bound "
                "and slower than numpy_cpu; give the job a larger card or fewer leaves.",
                rows,
                "y" if rows == 1 else "ies",
                self.max_gather_bytes / 1024**2,
                per_query_bytes / 1024**2,
                candidates_per_query,
                MIN_HEALTHY_MICRO_BATCH_ROWS,
            )
        return rows

    @torch.inference_mode()
    def rerank_batch(
        self,
        queries: np.ndarray,
        candidates: np.ndarray,
        top_k: int,
        metric: str = "euclidean",
    ) -> np.ndarray:
        """Return the ``top_k`` nearest candidate ids per query, ``-1`` padded.

        ``candidates`` is a ``(n_queries, max_candidates)`` int64 matrix whose
        short rows are padded with ``-1``. Queries with fewer than ``top_k``
        real candidates get a ``-1``-padded tail; queries with none get a full
        ``-1`` row.
        """
        metric = "angular" if metric == "cosine" else metric
        if metric not in {"euclidean", "angular", "inner_product"}:
            raise ValueError(
                "resident rerank supports only euclidean, angular, cosine, and inner_product."
            )
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_matrix = np.ascontiguousarray(queries, dtype=np.float32)
        candidate_matrix = np.ascontiguousarray(candidates, dtype=np.int64)
        if query_matrix.ndim != 2 or candidate_matrix.ndim != 2:
            raise ValueError("queries and candidates must both be 2D matrices.")
        if query_matrix.shape[0] != candidate_matrix.shape[0]:
            raise ValueError("queries and candidates must have the same number of rows.")
        if query_matrix.shape[1] != self.dim:
            raise ValueError("queries must have the same dimension as the database.")

        n_queries, max_candidates = candidate_matrix.shape
        results = np.full((n_queries, top_k), -1, dtype=np.int64)
        if n_queries == 0 or max_candidates == 0:
            return results

        keep = min(top_k, max_candidates)
        step = self._micro_batch_rows(max_candidates)
        with _full_fp32_matmul():
            for start in range(0, n_queries, step):
                end = min(start + step, n_queries)
                selected = self._rerank_micro_batch(
                    queries=query_matrix[start:end],
                    candidates=candidate_matrix[start:end],
                    keep=keep,
                    metric=metric,
                )
                results[start:end, :keep] = selected
        return results

    def _rerank_micro_batch(
        self,
        *,
        queries: np.ndarray,
        candidates: np.ndarray,
        keep: int,
        metric: str,
    ) -> np.ndarray:
        rows_count, max_candidates = candidates.shape
        query_tensor = torch.from_numpy(queries).to(self.device)
        candidate_tensor = torch.from_numpy(candidates).to(self.device)
        is_real = candidate_tensor >= 0
        # Padding slots read row 0 and are masked out afterwards.
        gather_ids = candidate_tensor.clamp(min=0).reshape(-1)

        rows = self.db.index_select(0, gather_ids).view(rows_count, max_candidates, self.dim)
        dots = torch.bmm(rows, query_tensor.unsqueeze(2)).squeeze(2)
        sqnorms = self.db_sqnorms.index_select(0, gather_ids).view(rows_count, max_candidates)

        if metric == "euclidean":
            # Squared L2 minus the per-query constant ||q||^2 — rank-equivalent
            # to the L2 norm the numpy path returns, and one GEMM instead of a
            # materialized difference tensor.
            distances = sqnorms - 2.0 * dots
        elif metric == "inner_product":
            # Raw dot product, unlike the angular/cosine branch below: no
            # normalization by ||row|| * ||query||.
            distances = -dots
        else:
            denom = sqnorms.sqrt() * torch.linalg.norm(query_tensor, dim=1, keepdim=True)
            safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
            similarities = torch.where(denom > 0, dots / safe_denom, torch.zeros_like(dots))
            distances = 1.0 - similarities

        distances = distances.masked_fill(~is_real, float("inf"))
        best = torch.topk(distances, keep, dim=1, largest=False)
        selected = torch.gather(candidate_tensor, 1, best.indices)
        # A padding slot can still be selected when a query has < keep real
        # candidates; those positions stay -1.
        selected = torch.where(torch.isinf(best.values), torch.full_like(selected, -1), selected)
        return selected.cpu().numpy()
