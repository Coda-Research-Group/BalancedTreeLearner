"""Logging and runtime metadata helpers for benchmark scripts."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch

from batl.constants import DEFAULT_ENSEMBLE_MIN_TREE_MATCHES, DEFAULT_RETRIEVAL_TOP_K
from batl.profiling import device_metadata
from batl.utils.config import ExperimentConfig
from batl.utils.metrics import estimate_candidate_set_size, recall_at_k


def standard_run_metadata(device: str) -> dict[str, dict[str, Any]]:
    """Return standard environment and hardware metadata for a benchmark run.

    The hardware section carries GPU name and VRAM, thread counts, and the TF32
    setting: without them, QPS rows measured on different nodes are not
    comparable, and the node a run landed on is not recoverable afterward.
    """
    return {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "hardware": {
            "device": device,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
            "platform": platform.platform(),
            **device_metadata(device),
        },
    }


def print_query_progress(
    *,
    label: str,
    done: int,
    total: int,
    elapsed_s: float,
) -> None:
    """Print one standard benchmark query-progress line."""
    rate = done / elapsed_s if elapsed_s > 0 else float("inf")
    print(
        f"{label}: queries done {done}/{total}, remaining {total - done}, "
        f"elapsed {elapsed_s:.1f}s, {rate:.2f} q/s",
        flush=True,
    )


def per_query_recall_stats(per_query: np.ndarray) -> dict[str, float]:
    """Summarise a per-query recall array for outlier/tail diagnostics."""
    if per_query.ndim != 1:
        raise ValueError("per_query must be a 1D array.")
    if per_query.size == 0:
        raise ValueError("per_query must be non-empty.")
    return {
        "p5": float(np.percentile(per_query, 5)),
        "p25": float(np.percentile(per_query, 25)),
        "p50": float(np.percentile(per_query, 50)),
        "p75": float(np.percentile(per_query, 75)),
        "p95": float(np.percentile(per_query, 95)),
        "min": float(per_query.min()),
        "max": float(per_query.max()),
        "zero_count": int((per_query == 0.0).sum()),
        "below_half_count": int((per_query < 0.5).sum()),
    }


def search_logging(
    cfg: ExperimentConfig,
    model_id: str,
    trees: list,
    queries: int,
    retrieved: np.ndarray,
    ground_truth: np.ndarray,
    n_candidates: np.ndarray,
    search_time_s: float,
    model_index_size_mb: float,
    index_path: Path,
    database_size: float,
    num_return_leaves: int,
) -> dict[str, Any]:
    per_query_recall = recall_at_k(retrieved, ground_truth, DEFAULT_RETRIEVAL_TOP_K)
    mean_recall = float(np.mean(per_query_recall)) if per_query_recall.size else 0.0
    recall_stats = per_query_recall_stats(per_query_recall)
    candidate_counts = np.asarray(n_candidates, dtype=np.int64)
    mean_candidate_distcomp = float(np.mean(candidate_counts)) if candidate_counts.size else 0.0
    routing_evals_exact = int(
        len(trees)
        * cfg.model.branching_factor
        * sum(
            min(cfg.beam_size, cfg.model.branching_factor**depth)
            for depth in range(cfg.model.tree_height)
        )
    )
    routing_evals_paper_upper_bound = int(
        len(trees) * cfg.model.branching_factor * cfg.beam_size * cfg.model.tree_height
    )
    leaf_count = cfg.model.branching_factor**cfg.model.tree_height
    single_tree_bound = num_return_leaves * database_size / leaf_count
    return {
        "method": "batl",
        "dataset": cfg.dataset_name,
        "model_id": model_id,
        "config": cfg.name,
        "seed": cfg.seed,
        "beam_size": cfg.beam_size,
        "num_trees": len(trees),
        "min_trees": (
            cfg.min_trees
            if cfg.min_trees is not None
            else (1 if len(trees) == 1 else DEFAULT_ENSEMBLE_MIN_TREE_MATCHES)
        ),
        "tree_assignment_mode": cfg.tree_assignment_mode,
        "tree_assignment_order": cfg.tree_assignment_order,
        "label_refresh": cfg.train.label_refresh,
        "knob_name": "num_leaves",
        "knob_value": num_return_leaves,
        "recall@10": mean_recall,
        "recall@10_zero_count": recall_stats["zero_count"],
        "recall@10_below_half_count": recall_stats["below_half_count"],
        "mean_distcomp": float(np.mean(n_candidates)),
        "std_n_distcomp": float(np.std(n_candidates)),
        "estimated_candidate_set_size": estimate_candidate_set_size(trees, num_return_leaves),
        "routing_evals_exact": routing_evals_exact,
        "routing_evals_paper_upper_bound": routing_evals_paper_upper_bound,
        "mean_n_total_work_upper_bound": (
            mean_candidate_distcomp + routing_evals_paper_upper_bound
        ),
        "n_queries": queries,
        "search_time_s": search_time_s,
        "qps": queries / search_time_s if search_time_s > 0 else float("inf"),
        "index_size_mb": model_index_size_mb,
        "index_path": str(index_path),
        "paper_single_tree_candidate_bound": float(single_tree_bound),
        "paper_ensemble_candidate_bound": float(single_tree_bound * len(trees)),
    }
