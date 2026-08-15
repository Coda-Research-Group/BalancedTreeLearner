"""Run a BATL search sweep over a trained index.

Requires a trained index (run build.py first).

Usage:
    python search.py config.yaml
    python search.py config.yaml --result-dir results/
    python search.py config.yaml --num-leaves 10 20 40 80
    python search.py config.yaml --n-queries 1000 --batch-search 50
    python search.py config.yaml --index-path /scratch/index.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import cast

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from batl.profiling import (
    StageProfiler,
    memory_metadata,
    repetition_summary,
    stage_reconciliation,
)
from batl.rerank import RerankGpuMemoryError, ResidentGpuReranker
from batl.search import RerankBackend as SearchRerankBackend
from batl.search import search_batch
from batl.training import AssignmentMode, AssignmentOrder
from batl.utils.arguments import (
    add_batch_search_arg,
    add_config_arg,
    add_datapath_arg,
    add_index_path_arg,
    add_log_arg,
    add_n_queries_arg,
    add_num_leaves_arg,
    add_result_dir_arg,
    add_skip_sanity_checks_arg,
)
from batl.utils.config_parsing import (
    load_config_with_device,
    run_final_config_sanity_checks,
)
from batl.utils.index_parsing import batl_index_path, load_batl_index_checked
from batl.utils.io import (
    jsonable,
    load_run_data,
    save_benchmark_artifacts,
    write_rows,
)
from batl.utils.logging_utils import print_query_progress, search_logging, standard_run_metadata
from batl.utils.metrics import (
    index_size_mb,
)

LOGGER = logging.getLogger(__name__)


def _resolve_search_batch(
    requested: int,
    *,
    beam_size: int,
    num_heads: int,
    n_queries: int,
    device: str,
) -> int:
    """Return a CUDA-grid-safe progress chunk for beam search.

    Mirrors `batl.tree_update._cuda_attention_batch_guard`: PyTorch's CUDA
    SDPA kernels error with `invalid configuration argument` when the
    effective batch-by-heads work dimension exceeds the CUDA grid limit of
    65,535. Per-call decoder rows are `n_queries_per_call * beam_size`, so
    the safe per-call query count is ``(65535 // num_heads) // beam_size``.
    The user's ``--batch-search`` is clamped down with one WARN log when it
    would otherwise overflow; ``--batch-search 0`` (no chunking) is honoured
    only when the full query set already fits.
    """
    if device != "cuda":
        return requested
    guard = max(1, 65535 // max(1, num_heads))
    cap = max(1, guard // max(1, beam_size))
    effective = n_queries if requested == 0 else requested
    if effective <= cap:
        return requested
    LOGGER.warning(
        "capping --batch-search from %d to %d "
        "(CUDA attention grid: 65535 // num_heads=%d // beam_size=%d)",
        effective,
        cap,
        num_heads,
        beam_size,
    )
    return cap


def _build_reranker(cfg, database: np.ndarray) -> tuple[ResidentGpuReranker | None, float]:
    """Build the process-wide resident reranker, downgrading if it will not fit.

    ``cfg.rerank_backend`` is mutated to ``numpy_cpu`` when the database does
    not fit in VRAM, so the resolved value reported in run artifacts is the
    backend that actually ran. Upload time is returned separately and is kept
    out of ``search_time_s`` — it is index-load work, not query work.
    """
    if cfg.rerank_backend != "torch_gpu_resident":
        return None, 0.0
    start = time.perf_counter()
    try:
        reranker = ResidentGpuReranker(database, torch.device(cfg.train.device))
    except RerankGpuMemoryError as exc:
        LOGGER.warning("resident GPU rerank unavailable, falling back to numpy_cpu: %s", exc)
        cfg.rerank_backend = "numpy_cpu"
        return None, 0.0
    elapsed = time.perf_counter() - start
    LOGGER.info(
        "uploaded %d x %d database to %s for resident rerank in %.1fs",
        database.shape[0],
        database.shape[1],
        cfg.train.device,
        elapsed,
    )
    return reranker, elapsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a BATL beam search over a trained index.")
    add_config_arg(parser)
    add_datapath_arg(parser)
    add_log_arg(parser)
    add_result_dir_arg(parser)
    add_skip_sanity_checks_arg(parser)
    add_index_path_arg(parser)
    add_num_leaves_arg(parser)
    add_n_queries_arg(parser)
    add_batch_search_arg(parser)
    return parser


def _search_with_progress(
    *,
    models,
    trees,
    database: np.ndarray,
    queries: np.ndarray,
    beam_size: int,
    num_return_leaves: int,
    progress_every: int,
    label: str,
    metric: str,
    rerank_backend: str,
    min_trees: int | None = None,
    reranker: ResidentGpuReranker | None = None,
    profiler: StageProfiler | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative.")
    if progress_every == 0:
        result = search_batch(
            models=models,
            trees=trees,
            database=database,
            queries=queries,
            beam_size=beam_size,
            return_candidate_counts=True,
            num_return_leaves=num_return_leaves,
            min_trees=min_trees,
            metric=metric,
            rerank_backend=cast(SearchRerankBackend, rerank_backend),
            reranker=reranker,
            profiler=profiler,
        )
        return cast(tuple[np.ndarray, np.ndarray], result)

    retrieved_parts = []
    candidate_parts = []
    start_time = time.perf_counter()
    total = queries.shape[0]
    for start in range(0, total, progress_every):
        end = min(start + progress_every, total)
        retrieved, n_candidates = search_batch(
            models=models,
            trees=trees,
            database=database,
            queries=queries[start:end],
            beam_size=beam_size,
            return_candidate_counts=True,
            num_return_leaves=num_return_leaves,
            min_trees=min_trees,
            metric=metric,
            rerank_backend=cast(SearchRerankBackend, rerank_backend),
            reranker=reranker,
            profiler=profiler,
        )
        retrieved_parts.append(retrieved)
        candidate_parts.append(n_candidates)
        print_query_progress(
            label=label,
            done=end,
            total=total,
            elapsed_s=time.perf_counter() - start_time,
        )
    return np.vstack(retrieved_parts), np.concatenate(candidate_parts)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.log:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    cfg = load_config_with_device(args.config, args)
    run_final_config_sanity_checks(cfg, skip=args.skip_sanity_checks)

    assignment_mode = cast(AssignmentMode, cfg.tree_assignment_mode)
    assignment_order = cast(AssignmentOrder, cfg.tree_assignment_order)
    index_path = Path(
        args.index_path or batl_index_path(cfg.output_dir, assignment_order, assignment_mode)
    )

    vectors, queries, ground_truth = load_run_data(cfg)
    queries = queries[: cfg.num_queries]
    ground_truth = ground_truth[: cfg.num_queries]

    models, trees = load_batl_index_checked(
        index_path,
        cfg,
        expected_n=vectors.shape[0],
        require_num_trees=cfg.model.num_trees,
        slice_to_num_trees=True,
    )
    for model in models:
        model.to(torch.device(cfg.train.device))

    reranker, rerank_build_s = _build_reranker(cfg, vectors)

    model_id = f"K{cfg.model.branching_factor}_H{cfg.model.tree_height}"
    metric = cfg.dataset_metric or "euclidean"
    model_index_size_mb = index_size_mb(models, trees)
    rows: list[dict] = []

    effective_batch_search = _resolve_search_batch(
        args.batch_search,
        beam_size=cfg.beam_size,
        num_heads=cfg.model.num_heads,
        n_queries=queries.shape[0],
        device=cfg.train.device,
    )

    profile_stages: dict[str, dict] = {}
    for num_return_leaves in cfg.num_leaves:
        print(f"{model_id}: searching beam_size={cfg.beam_size}, num_leaves={num_return_leaves}")
        profiler = StageProfiler(enabled=cfg.performance_profile, device=cfg.train.device)
        repetition_times: list[float] = []
        for repetition in range(cfg.search_repetitions):
            # Only the last repetition's timings are kept: earlier ones warm
            # caches, and mixing warm and cold stages would blur the profile.
            profiler.reset()
            _synchronize_search_device(cfg.train.device)
            start = time.perf_counter()
            retrieved, n_candidates = _search_with_progress(
                models=models,
                trees=trees,
                database=vectors,
                queries=queries,
                beam_size=cfg.beam_size,
                num_return_leaves=num_return_leaves,
                progress_every=effective_batch_search,
                label=f"{model_id} b={cfg.beam_size} M={num_return_leaves}",
                metric=metric,
                rerank_backend=cfg.rerank_backend,
                min_trees=cfg.min_trees,
                reranker=reranker,
                profiler=profiler,
            )
            _synchronize_search_device(cfg.train.device)
            search_time_s = time.perf_counter() - start
            repetition_times.append(search_time_s)
            if cfg.search_repetitions > 1:
                LOGGER.info(
                    "M=%d repetition %d/%d: %.2fs",
                    num_return_leaves,
                    repetition + 1,
                    cfg.search_repetitions,
                    search_time_s,
                )

        timing = repetition_summary(repetition_times)
        # Median across repetitions is the reported time when repeating; with a
        # single repetition it is that one measurement.
        search_time_s = cast(float, timing["median_s"])
        if cfg.performance_profile:
            profile_stages[f"num_leaves_{num_return_leaves}"] = {
                "stages": profiler.to_dict(),
                "reconciliation": stage_reconciliation(profiler, repetition_times[-1]),
                "timing": timing,
                "memory": memory_metadata(cfg.train.device),
            }
        row = search_logging(
            cfg=cfg,
            model_id=model_id,
            trees=trees,
            queries=queries.shape[0],
            retrieved=retrieved,
            ground_truth=ground_truth,
            n_candidates=n_candidates,
            search_time_s=search_time_s,
            index_path=index_path,
            database_size=vectors.shape[0],
            num_return_leaves=num_return_leaves,
            model_index_size_mb=model_index_size_mb,
        )
        rows.append(row)
        print(json.dumps(jsonable(row), indent=2))

    for model in models:
        model.to(torch.device("cpu"))

    write_rows(cfg.output_dir, "search_rows", rows)
    save_benchmark_artifacts(
        output_dir=cfg.output_dir,
        rows=rows,
        run_plan={
            "config": args.config,
            "index_path": str(index_path),
            "num_leaves": cfg.num_leaves,
            "n_queries": cfg.num_queries,
            "beam_size": cfg.beam_size,
            "min_trees": cfg.resolved_min_trees(),
        },
        cfg=cfg,
        seed=cfg.seed,
        run_metadata=standard_run_metadata(cfg.train.device),
        extra_metrics={
            "beam_size": cfg.beam_size,
            "min_trees": cfg.resolved_min_trees(),
            "num_leaves": cfg.num_leaves,
            "tree_assignment_mode": cfg.tree_assignment_mode,
            "tree_assignment_order": cfg.tree_assignment_order,
            "rerank_backend": cfg.rerank_backend,
            # Database upload for the resident reranker; excluded from
            # search_time_s, same as index-load time for baseline methods.
            "rerank_build_s": rerank_build_s,
            "performance_profile": cfg.performance_profile,
            "search_repetitions": cfg.search_repetitions,
            **({"profile": profile_stages} if cfg.performance_profile else {}),
        },
    )


def _synchronize_search_device(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


if __name__ == "__main__":
    main()
