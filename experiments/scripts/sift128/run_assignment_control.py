"""Create deterministic tree-assignment variants from one saved BATL index."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from batl.model import BATLModel
from batl.training import AssignmentMode, AssignmentOrder
from batl.tree import BATLTree
from batl.tree_update import (
    _resolve_tree_update_batch_size,
    _resolve_tree_update_cache_embeddings,
    update_tree,
)
from batl.utils.arguments import add_config_arg, add_datapath_arg, add_log_arg
from batl.utils.config_parsing import load_config_with_device, run_final_config_sanity_checks
from batl.utils.index_parsing import batl_index_path, load_batl_index_checked, save_index
from batl.utils.io import jsonable, load_run_database, save_benchmark_artifacts
from batl.utils.logging_utils import standard_run_metadata

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssignmentArm:
    name: str
    mode: AssignmentMode
    order: AssignmentOrder


# Bit-identity is not achievable across a heterogeneous cluster (see the note
# in reassign_index). Anything under this is accepted as last-bit drift and
# logged; anything over it fails the run.
SOURCE_REPRODUCTION_TOLERANCE = 1e-3  # 0.1% of vectors


ASSIGNMENT_ARMS = (
    AssignmentArm("round_confidence", "round", "confidence"),
    AssignmentArm("round_input", "round", "input"),
    AssignmentArm("sequential_input", "sequential", "input"),
    # Added 2026-08-09 after the first run. Mode and order are independent, and
    # the first three arms showed order dominates: confidence ordering is worth
    # 1.43-1.89x on the round kernel, while switching round -> sequential at
    # input order costs 0.81-0.87x. This is the missing corner — the paper's
    # sequential kernel with the ordering that actually helps. It is a
    # diagnostic, not a paper-faithful arm, because Algorithm 1 leaves the
    # `node.data` iteration order unspecified.
    AssignmentArm("sequential_confidence", "sequential", "confidence"),
)


def leaf_divergence(
    left: list[BATLTree],
    right: list[BATLTree],
) -> dict[str, object]:
    """Report how often two ensembles place a vector in different final leaves."""
    if len(left) != len(right):
        raise ValueError("left and right must contain the same number of trees.")
    if not left:
        raise ValueError("leaf divergence requires at least one tree.")

    changed_masks: list[np.ndarray] = []
    for tree_index, (left_tree, right_tree) in enumerate(zip(left, right, strict=True)):
        if left_tree.paths.shape != right_tree.paths.shape:
            raise ValueError(f"tree {tree_index} path shapes differ.")
        changed_masks.append(np.any(left_tree.paths != right_tree.paths, axis=1))

    changed_vectors = int(sum(int(mask.sum()) for mask in changed_masks))
    total_vectors = int(sum(mask.size for mask in changed_masks))
    return {
        "per_tree_fraction": [float(mask.mean()) for mask in changed_masks],
        "overall_fraction": float(changed_vectors / total_vectors),
        "changed_vectors": changed_vectors,
        "total_vectors": total_vectors,
    }


def reassign_index(
    *,
    models: list[BATLModel],
    source_trees: list[BATLTree],
    vectors: np.ndarray,
    output_dir: str | Path,
    batch_size: int,
    device: torch.device,
    cache_embeddings: bool,
    round_top_r: int | None,
    arms: tuple[AssignmentArm, ...] = ASSIGNMENT_ARMS,
) -> dict[str, object]:
    """Apply the selected assignment arms to one ensemble and save each index.

    ``arms`` defaults to the full grid. Deep100M can only run the round arms:
    sequential mode needs the full-K branch order materialized, which is
    100M x 256 at the root, so it is restricted to K=64 scales. The ordering
    question S8 asks does not need sequential, so the round pair alone confirms
    it at 100M.
    """
    if len(models) != len(source_trees):
        raise ValueError("models and source_trees must have the same length.")
    if not models:
        raise ValueError("assignment control requires at least one model/tree pair.")
    if any(tree.N != vectors.shape[0] for tree in source_trees):
        raise ValueError("every source tree must match the database row count.")

    if not any(arm.name == "round_confidence" for arm in arms):
        raise ValueError(
            "round_confidence must be among the arms: it is the arm checked "
            "against the cached source tree, and without it nothing verifies "
            "that the harness loaded and decoded the right model."
        )

    root = Path(output_dir)
    arm_trees: dict[str, list[BATLTree]] = {}
    arm_reports: dict[str, dict[str, object]] = {}

    for arm in arms:
        updated_trees: list[BATLTree] = []
        tree_reports: list[dict[str, object]] = []
        arm_start = time.perf_counter()
        for tree_index, (model, source_tree) in enumerate(zip(models, source_trees, strict=True)):
            tree_start = time.perf_counter()
            updated, diagnostics = update_tree(
                model=model,
                vectors=vectors,
                current_tree=source_tree,
                batch_size=batch_size,
                device=device,
                assignment_mode=arm.mode,
                assignment_order=arm.order,
                cache_embeddings=cache_embeddings,
                return_diagnostics=True,
                top_r=round_top_r if arm.mode == "round" else None,
            )
            updated_trees.append(updated)
            tree_reports.append(
                {
                    "tree_index": tree_index,
                    "reassignment_time_s": time.perf_counter() - tree_start,
                    "diagnostics": asdict(diagnostics),
                }
            )

        for model in models:
            model.to(torch.device("cpu"))
        index_path = batl_index_path(root / arm.name, arm.order, arm.mode)
        save_index(models, updated_trees, str(index_path))
        arm_trees[arm.name] = updated_trees
        arm_reports[arm.name] = {
            "assignment_mode": arm.mode,
            "assignment_order": arm.order,
            "index_path": str(index_path),
            "reassignment_time_s": time.perf_counter() - arm_start,
            "trees": tree_reports,
        }

    # The guard exists to catch a harness that is loading or decoding something
    # else entirely; that failure mode is tens of percent. Bit-identity is not
    # the right test for it on a heterogeneous cluster: the same code, thread
    # count and source index reproduced exactly on one node (job 22788625) and
    # drifted by 41 vectors in 4,000,000 on another (job 22791165), because a
    # different CPU selects different BLAS kernels, hence a different reduction
    # order, which flips assignments wherever two branches are near-tied at a
    # capacity boundary.
    #
    # The tolerance sits 100x above that observed drift and roughly 250x below
    # the smallest effect the control measures (round vs sequential moves 25.7%
    # of vectors), so it cannot mask a real problem.
    source_divergence = leaf_divergence(source_trees, arm_trees["round_confidence"])
    drift = float(source_divergence["overall_fraction"])
    if drift > SOURCE_REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            "round-confidence reassignment did not reproduce the cached source trees: "
            f"{drift:.6%} of vectors changed leaf "
            f"({source_divergence['changed_vectors']} of "
            f"{source_divergence['total_vectors']}), per tree "
            f"{[f'{f:.6%}' for f in source_divergence['per_tree_fraction']]}. "
            f"That exceeds the {SOURCE_REPRODUCTION_TOLERANCE:.3%} float-drift "
            "tolerance, so this is not last-bit noise: check that the source index "
            "was built by the model it is paired with, and that tree_update_top_r "
            "and the assignment mode/order match the run that produced it."
        )
    if drift:
        LOGGER.warning(
            "round-confidence reproduced the source trees to within float drift: "
            "%.6f%% of vectors changed leaf (%d of %d). Accepted — the arms below "
            "are all recomputed in this job, so the comparison between them is "
            "unaffected.",
            drift * 100.0,
            source_divergence["changed_vectors"],
            source_divergence["total_vectors"],
        )

    return {
        "source_round_confidence_exact_match": drift == 0.0,
        "source_round_confidence_divergence": source_divergence,
        "arms": arm_reports,
        "partition_divergence": {
            # Kernel effect at each ordering, and the ordering effect at each
            # kernel. Reporting only one pair hides that the two dimensions are
            # separable — and that the kernel's ranking flips between them.
            # Pairs whose arms were not run are simply absent.
            name: leaf_divergence(arm_trees[left], arm_trees[right])
            for name, left, right in (
                ("round_input_vs_sequential_input", "round_input", "sequential_input"),
                (
                    "round_confidence_vs_sequential_confidence",
                    "round_confidence",
                    "sequential_confidence",
                ),
                ("round_input_vs_round_confidence", "round_input", "round_confidence"),
                (
                    "sequential_input_vs_sequential_confidence",
                    "sequential_input",
                    "sequential_confidence",
                ),
            )
            if left in arm_trees and right in arm_trees
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create round/sequential assignment controls from one saved BATL index."
    )
    add_config_arg(parser)
    add_datapath_arg(parser)
    add_log_arg(parser)
    parser.add_argument("--source-index", required=True, help="Saved round-confidence index.")
    parser.add_argument("--output-dir", required=True, help="Root directory for the arms.")
    parser.add_argument(
        "--arms",
        default="",
        help=(
            "Comma-separated arm names; default runs the full kernel x order grid. "
            "round_confidence is always required — it is the arm checked against the "
            "cached source tree. Deep100M must pass round_confidence,round_input: "
            "sequential mode needs the full-K branch order, which is 100M x 256 at "
            "the root."
        ),
    )
    parser.add_argument(
        "--skip-sanity-checks",
        action="store_true",
        help="Bypass final-config sanity checks for diagnostic use only.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.log:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    selected = {arm.name: arm for arm in ASSIGNMENT_ARMS}
    if args.arms:
        requested = [name.strip() for name in args.arms.split(",") if name.strip()]
        unknown = sorted(set(requested) - set(selected))
        if unknown:
            raise ValueError(f"unknown arms {unknown}; choose from {sorted(selected)}")
        arms = tuple(arm for arm in ASSIGNMENT_ARMS if arm.name in set(requested))
    else:
        arms = ASSIGNMENT_ARMS

    cfg = load_config_with_device(args.config, args)
    run_final_config_sanity_checks(cfg, skip=args.skip_sanity_checks)
    output_dir = Path(args.output_dir)
    cfg.output_dir = str(output_dir)
    vectors = load_run_database(cfg)
    models, source_trees = load_batl_index_checked(
        args.source_index,
        cfg,
        expected_n=vectors.shape[0],
        require_num_trees=cfg.model.num_trees,
        slice_to_num_trees=True,
    )
    device = torch.device(cfg.train.device)
    batch_size = _resolve_tree_update_batch_size(
        cfg.train.tree_update_batch_size,
        cfg.model.num_heads,
        device,
    )
    cache_embeddings = _resolve_tree_update_cache_embeddings(cfg.train.tree_update_cache_embeddings)
    report = reassign_index(
        models=cast(list[BATLModel], models),
        source_trees=cast(list[BATLTree], source_trees),
        vectors=vectors,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
        cache_embeddings=cache_embeddings,
        round_top_r=cfg.train.tree_update_top_r,
        arms=arms,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "assignment_control.json"
    report_path.write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_benchmark_artifacts(
        output_dir=output_dir,
        rows=[],
        run_plan={
            "config": args.config,
            "source_index": args.source_index,
            "arms": [asdict(arm) for arm in arms],
        },
        cfg=cfg,
        seed=cfg.seed,
        run_metadata=standard_run_metadata(cfg.train.device),
        extra_metrics={"assignment_control": report},
    )
    LOGGER.info("wrote assignment control report: %s", report_path)


if __name__ == "__main__":
    main()
