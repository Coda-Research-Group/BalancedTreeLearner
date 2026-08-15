"""Shared argument parsing helpers for benchmark entrypoints."""

from __future__ import annotations

import argparse

from batl.utils.config import ExperimentConfig


def apply_args_to_config(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Apply common command-line overrides to the experiment config."""
    if getattr(args, "datapath", None) is not None:
        cfg.dataset_path = args.datapath
    if getattr(args, "batch_train", None) is not None:
        cfg.train.batch_size = args.batch_train
    if getattr(args, "batch_tree_update", None) is not None:
        cfg.train.tree_update_batch_size = args.batch_tree_update
    if getattr(args, "result_dir", None) is not None:
        cfg.output_dir = args.result_dir
    if getattr(args, "num_leaves", None) is not None:
        cfg.num_leaves = args.num_leaves
    if getattr(args, "n_queries", None) is not None:
        cfg.num_queries = args.n_queries
    return cfg


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive.")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative.")
    return parsed


# ---------------------------------------------------------------------------
# Shared args (build + search)
# ---------------------------------------------------------------------------


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Positional config path: python script.py config.yaml"""
    parser.add_argument("config", help="Path to experiment YAML config.")


def add_datapath_arg(parser: argparse.ArgumentParser) -> None:
    """Override config dataset_path at runtime."""
    parser.add_argument(
        "--datapath",
        default=None,
        help="Override config dataset_path (or directory for multi-file datasets).",
    )


def add_log_arg(parser: argparse.ArgumentParser) -> None:
    """Enable verbose INFO logging."""
    parser.add_argument("--log", action="store_true", help="Enable verbose logging.")


def add_result_dir_arg(parser: argparse.ArgumentParser) -> None:
    """Override output directory for all artifacts (results, logs)."""
    parser.add_argument(
        "--result-dir",
        default=None,
        help="Output directory for all artifacts. Defaults to config output_dir.",
    )


def add_index_path_arg(parser: argparse.ArgumentParser) -> None:
    """Override index file path (build writes, search reads)."""
    parser.add_argument(
        "--index-path",
        default=None,
        help="Index file path. Build writes here; search reads from here. "
        "Defaults to <config.output_dir>/index_<assignment_order>.pkl.",
    )


# ---------------------------------------------------------------------------
# Build-only args
# ---------------------------------------------------------------------------


def add_batch_train_arg(parser: argparse.ArgumentParser) -> None:
    """Override training batch size."""
    parser.add_argument(
        "--batch-train",
        type=positive_int,
        default=None,
        help="Training batch size. Overrides config training.batch_size.",
    )


def _int_or_auto(value: str) -> int | str:
    if value == "auto":
        return "auto"
    try:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError
        return parsed
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"Expected a positive integer or 'auto', got {value!r}."
        ) from err


def add_batch_tree_update_arg(parser: argparse.ArgumentParser) -> None:
    """Override tree-update batch size."""
    parser.add_argument(
        "--batch-tree-update",
        type=_int_or_auto,
        default=None,
        help="Tree-update batch size, or 'auto' to size from free device memory. "
        "Overrides config training.tree_update_batch_size.",
    )


# ---------------------------------------------------------------------------
# Search-only args
# ---------------------------------------------------------------------------


def add_num_leaves_arg(parser: argparse.ArgumentParser) -> None:
    """Override returned-leaf sweep from config."""
    parser.add_argument(
        "--num-leaves",
        nargs="+",
        type=positive_int,
        default=None,
        help="Returned-leaf counts to sweep. Defaults to config evaluation.num_leaves.",
    )


def add_n_queries_arg(parser: argparse.ArgumentParser) -> None:
    """Cap or override query count from config."""
    parser.add_argument(
        "--n-queries",
        type=positive_int,
        default=None,
        help="Number of queries to run. Defaults to config evaluation.num_queries.",
    )


def add_batch_search_arg(parser: argparse.ArgumentParser) -> None:
    """Search progress reporting interval (queries per log line)."""
    parser.add_argument(
        "--batch-search",
        type=non_negative_int,
        default=100,
        help="Report search progress every N queries (0 = silent). Default: 100.",
    )


def add_skip_sanity_checks_arg(parser: argparse.ArgumentParser) -> None:
    """Escape hatch for diagnostic runs that knowingly break a sanity rule."""
    parser.add_argument(
        "--skip-sanity-checks",
        action="store_true",
        help="Bypass final-config sanity checks (diagnostic runs only).",
    )
