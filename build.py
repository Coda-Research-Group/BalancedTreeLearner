"""Train a BATL index for a single experiment config.

Usage:
    python build.py config.yaml
    python build.py config.yaml --log
    python build.py config.yaml --index-path /scratch/index.pkl
    python build.py config.yaml --batch-train 512 --batch-tree-update 1024
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import cast

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from batl.model import BATLModel
from batl.profiling import (
    StageProfiler,
    memory_metadata,
    stage_reconciliation,
)
from batl.search import RerankBackend as SearchRerankBackend
from batl.search import per_query_rerank_backend
from batl.training import (
    AssignmentMode,
    AssignmentOrder,
    TrainingDiagnosticsConfig,
    alternating_train,
)
from batl.tree import BATLTree
from batl.utils.arguments import (
    add_batch_train_arg,
    add_batch_tree_update_arg,
    add_config_arg,
    add_datapath_arg,
    add_index_path_arg,
    add_log_arg,
    add_result_dir_arg,
    add_skip_sanity_checks_arg,
    positive_int,
)
from batl.utils.config import ExperimentConfig
from batl.utils.config_parsing import (
    load_config_with_device,
    run_final_config_sanity_checks,
)
from batl.utils.index_parsing import batl_index_path, load_batl_index_checked, save_index
from batl.utils.io import load_run_data, load_run_database, save_benchmark_artifacts
from batl.utils.logging_utils import standard_run_metadata
from batl.utils.reproducibility import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and cache a BATL index for a single experiment config."
    )
    add_config_arg(parser)
    add_datapath_arg(parser)
    add_log_arg(parser)
    add_result_dir_arg(parser)
    add_skip_sanity_checks_arg(parser)
    add_index_path_arg(parser)
    add_batch_train_arg(parser)
    add_batch_tree_update_arg(parser)
    parser.add_argument(
        "--tree-index",
        type=int,
        default=None,
        help=(
            "Train exactly one ensemble tree with seed=config.seed+tree_index "
            "and save to <index>_tree_<tree_index>.pkl."
        ),
    )
    parser.add_argument(
        "--cycle-diagnostics",
        action="store_true",
        help=(
            "During training, write per-cycle JSONL diagnostics with training losses, "
            "post-model top-bucket recall, and post-tree-update top-bucket recall."
        ),
    )
    parser.add_argument(
        "--cycle-diagnostics-path",
        default=None,
        help="JSONL path for --cycle-diagnostics. Defaults to <output_dir>/cycle_diagnostics.jsonl.",
    )
    parser.add_argument(
        "--cycle-diagnostics-queries",
        type=positive_int,
        default=None,
        help="Number of evaluation queries used for per-cycle recall diagnostics.",
    )
    parser.add_argument(
        "--cycle-diagnostics-loss-pairs",
        type=positive_int,
        default=100_000,
        help="Maximum sampled training pairs used for post-block diagnostic eval loss.",
    )
    return parser


def tree_index_path(index_path: str | Path, tree_index: int) -> Path:
    """Return the per-tree index path derived from an ensemble index path."""
    path = Path(index_path)
    return path.with_name(f"{path.stem}_tree_{tree_index}{path.suffix}")


def build_batl_index(
    *,
    cfg: ExperimentConfig,
    vectors: np.ndarray,
    index_path: str | Path,
    assignment_mode: AssignmentMode = "round",
    assignment_order: AssignmentOrder = "input",
    metric: str = "euclidean",
    require_cached_index: bool = False,
    tree_index: int | None = None,
    diagnostics: TrainingDiagnosticsConfig | None = None,
    profiler: StageProfiler | None = None,
) -> tuple[list[BATLModel], list[BATLTree], float | None]:
    """Load a cached BATL index or train/resume the requested ensemble."""
    path = Path(index_path)
    if tree_index is not None:
        if tree_index < 0 or tree_index >= cfg.model.num_trees:
            raise ValueError(f"tree_index must be in [0, {cfg.model.num_trees}), got {tree_index}.")
        path = tree_index_path(path, tree_index)

    if require_cached_index and not path.exists():
        raise FileNotFoundError(f"Evaluation-only mode will not train a new index. Missing: {path}")
    models: list[BATLModel] = []
    trees: list[BATLTree] = []
    if assignment_mode not in {"round", "sequential"}:
        raise ValueError("assignment_mode must be 'round' or 'sequential'.")
    if assignment_order not in {"input", "confidence", "margin"}:
        raise ValueError("assignment_order must be 'input', 'confidence', or 'margin'.")

    if path.exists():
        print(f"loading cached index: {path}")
        models, trees = load_batl_index_checked(path, cfg, vectors.shape[0])
        if tree_index is not None:
            if len(models) != 1:
                raise ValueError(
                    f"Per-tree index {path} must contain exactly one tree, got {len(models)}."
                )
            return models, trees, None
        if len(models) >= cfg.model.num_trees:
            if len(models) > cfg.model.num_trees:
                print(
                    f"cached index has {len(models)} model(s); using first "
                    f"{cfg.model.num_trees} for this config"
                )
            return models[: cfg.model.num_trees], trees[: cfg.model.num_trees], None
        print(
            f"cached index has {len(models)}/{cfg.model.num_trees} ensemble "
            "tree(s); resuming training"
        )
    elif path.exists():
        print(f"ignoring cached index because --force-train was set: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    tree_indices = (
        [tree_index] if tree_index is not None else range(len(models), cfg.model.num_trees)
    )
    for tree_idx in tree_indices:
        assert tree_idx is not None
        model_seed = cfg.seed + tree_idx
        print(
            f"training {cfg.name} independent ensemble tree "
            f"{tree_idx + 1}/{cfg.model.num_trees} on {vectors.shape[0]} vectors"
        )
        set_seed(model_seed)
        model = BATLModel(cfg.model)
        model, tree = alternating_train(
            model,
            vectors,
            cfg.train,
            cfg.model,
            seed=model_seed,
            assignment_mode=assignment_mode,
            assignment_order=assignment_order,
            metric=metric,
            diagnostics=diagnostics,
            profiler=profiler,
        )
        model.to("cpu")
        models.append(model)
        trees.append(tree)
        save_index(models, trees, str(path))
        if tree_index is None:
            print(
                f"saved partial index: {path} "
                f"({len(models)}/{cfg.model.num_trees} ensemble tree(s))"
            )
        else:
            print(f"saved per-tree index: {path} (tree_index={tree_index})")

    train_time_s = time.perf_counter() - start
    print(f"saved index: {path} ({train_time_s:.2f}s this run)")
    return models, trees, train_time_s


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
    index_path = args.index_path or batl_index_path(
        cfg.output_dir,
        assignment_order,
        assignment_mode,
    )
    if args.tree_index is not None and args.tree_index < 0:
        raise ValueError("--tree-index must be non-negative.")

    if args.cycle_diagnostics:
        vectors, queries, ground_truth = load_run_data(cfg)
        diagnostics_path = args.cycle_diagnostics_path or str(
            Path(cfg.output_dir) / "cycle_diagnostics.jsonl"
        )
        diagnostics = TrainingDiagnosticsConfig(
            queries=queries,
            ground_truth=ground_truth,
            output_path=diagnostics_path,
            beam_size=cfg.beam_size,
            recall_k=max(cfg.recall_at),
            num_return_leaves=1,
            max_queries=args.cycle_diagnostics_queries,
            max_loss_pairs=args.cycle_diagnostics_loss_pairs,
            rerank_backend=per_query_rerank_backend(cast(SearchRerankBackend, cfg.rerank_backend)),
            metric=cfg.dataset_metric or "euclidean",
        )
    else:
        vectors = load_run_database(cfg)
        diagnostics = None
    profiler = StageProfiler(enabled=cfg.performance_profile, device=cfg.train.device)
    models, trees, train_time_s = build_batl_index(
        cfg=cfg,
        vectors=vectors,
        index_path=index_path,
        assignment_mode=assignment_mode,
        assignment_order=assignment_order,
        metric=cfg.dataset_metric or "euclidean",
        tree_index=args.tree_index,
        diagnostics=diagnostics,
        profiler=profiler,
    )
    actual_index_path = (
        tree_index_path(index_path, args.tree_index)
        if args.tree_index is not None
        else Path(index_path)
    )

    from batl.utils.metrics import index_size_mb

    save_benchmark_artifacts(
        output_dir=cfg.output_dir,
        rows=[],
        run_plan={"config": args.config, "index_path": str(actual_index_path)},
        cfg=cfg,
        seed=cfg.seed,
        run_metadata=standard_run_metadata(cfg.train.device),
        extra_metrics={
            "train_time_s": train_time_s,
            "index_size_mb": index_size_mb(models, trees),
            "num_trees_built": len(models),
            "index_path": str(actual_index_path),
            "performance_profile": cfg.performance_profile,
            **(
                {
                    "profile": {
                        "stages": profiler.to_dict(),
                        "reconciliation": stage_reconciliation(profiler, train_time_s or 0.0),
                        "memory": memory_metadata(cfg.train.device),
                    }
                }
                if cfg.performance_profile
                else {}
            ),
        },
    )


if __name__ == "__main__":
    main()
