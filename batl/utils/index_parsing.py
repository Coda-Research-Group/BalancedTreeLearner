from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from batl.model import BATLModel
from batl.training import AssignmentMode, AssignmentOrder
from batl.tree import BATLTree
from batl.utils.config import ExperimentConfig, ModelConfig

_INDEX_FORMAT_VERSION = 2


def create_index_payload(models: list[BATLModel], trees: list[BATLTree]) -> dict[str, Any]:
    """Create the portable state payload for BATL models and trees."""
    from batl.utils.io import _tree_payload

    return {
        "format_version": _INDEX_FORMAT_VERSION,
        "model_configs": [asdict(model.config) for model in models],
        "model_states": [
            {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
            for model in models
        ],
        "trees": [_tree_payload(tree) for tree in trees],
    }


def save_index(models: list, trees: list, path: str) -> None:
    """Persist trained BATL models and trees in a portable state format."""
    if len(models) != len(trees):
        raise ValueError("models and trees must have the same length.")
    if not all(isinstance(model, BATLModel) for model in models):
        raise TypeError("save_index expects BATLModel instances.")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = create_index_payload(models, trees)
    torch.save(payload, output)


def load_index(path: str) -> tuple[list, list]:
    """Load BATL models and trees saved with ``save_index``."""
    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except RuntimeError:
        with source.open("rb") as f:
            legacy_payload = pickle.load(f)
        return legacy_payload["models"], legacy_payload["trees"]

    if payload.get("format_version") != _INDEX_FORMAT_VERSION:
        raise ValueError(f"Unsupported BATL index format: {payload.get('format_version')!r}")

    models = []
    for config_payload, state_dict in zip(
        payload["model_configs"], payload["model_states"], strict=True
    ):
        model = BATLModel(ModelConfig(**config_payload))
        model.load_state_dict(state_dict)
        model.to("cpu")
        models.append(model)
    trees = [BATLTree(**tree_payload) for tree_payload in payload["trees"]]
    return models, trees


def merge_indexes(input_paths: Sequence[str | Path], output_path: str | Path) -> tuple[int, int]:
    """Merge compatible single-tree or partial BATL indexes into one ensemble index."""
    if not input_paths:
        raise ValueError("merge_indexes requires at least one input path.")

    merged_models: list[BATLModel] = []
    merged_trees: list[BATLTree] = []
    reference_config: ModelConfig | None = None
    reference_tree_shape: tuple[int, int, float, int] | None = None

    for input_path in input_paths:
        models, trees = load_index(str(input_path))
        if len(models) != len(trees):
            raise ValueError(
                f"Input index {input_path} is invalid: model/tree counts differ "
                f"({len(models)} vs {len(trees)})."
            )
        if not models:
            raise ValueError(f"Input index {input_path} contains no model/tree pairs.")
        if not all(isinstance(model, BATLModel) for model in models):
            raise TypeError(f"Input index {input_path} contains non-BATLModel entries.")
        if not all(isinstance(tree, BATLTree) for tree in trees):
            raise TypeError(f"Input index {input_path} contains non-BATLTree entries.")

        for model, tree in zip(models, trees, strict=True):
            if reference_config is None:
                reference_config = model.config
            elif model.config != reference_config:
                raise ValueError("All merged indexes must use identical ModelConfig values.")

            tree_shape = (tree.K, tree.H, tree.alpha, tree.N)
            if reference_tree_shape is None:
                reference_tree_shape = tree_shape
            elif tree_shape != reference_tree_shape:
                raise ValueError(
                    "All merged indexes must satisfy tree compatibility (same K, H, alpha, and N)."
                )

            merged_models.append(model)
            merged_trees.append(tree)

    save_index(merged_models, merged_trees, str(output_path))
    return len(merged_models), len(merged_trees)


def batl_index_path(
    output_dir: str | Path,
    assignment_order: AssignmentOrder,
    assignment_mode: AssignmentMode = "round",
) -> Path:
    """Return a collision-safe cache path for an assignment mode and order."""
    if assignment_order not in {"input", "confidence", "margin"}:
        raise ValueError("assignment_order must be 'input', 'confidence', or 'margin'.")
    if assignment_mode not in {"round", "sequential"}:
        raise ValueError("assignment_mode must be 'round' or 'sequential'.")
    suffix = (
        assignment_order if assignment_mode == "round" else f"{assignment_mode}_{assignment_order}"
    )
    return Path(output_dir) / f"index_{suffix}.pkl"


def load_batl_index_checked(
    index_path: str | Path,
    cfg: ExperimentConfig,
    expected_n: int,
    *,
    require_num_trees: int | None = None,
    slice_to_num_trees: bool = False,
) -> tuple[list[BATLModel], list[BATLTree]]:
    """Load and validate a cached BATL index."""
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Cached index not found: {path}")

    models, trees = load_index(str(path))
    if len(models) != len(trees):
        raise ValueError(
            f"Cached index {path} is invalid: model/tree counts differ "
            f"({len(models)} vs {len(trees)})."
        )
    if not models:
        raise ValueError(f"Cached index {path} contains no model/tree pairs.")

    for tree in trees:
        if (
            tree.K != cfg.model.branching_factor
            or tree.H != cfg.model.tree_height
            or tree.N != expected_n
        ):
            raise ValueError(
                f"Cached index {path} does not match config "
                f"(expected K={cfg.model.branching_factor}, H={cfg.model.tree_height}, "
                f"N={expected_n}; got K={tree.K}, H={tree.H}, N={tree.N})."
            )

    if require_num_trees is not None:
        if require_num_trees <= 0:
            raise ValueError("require_num_trees must be positive when provided.")
        if len(models) < require_num_trees:
            raise ValueError(
                f"Cached index {path} has only {len(models)}/{require_num_trees} required tree(s)."
            )
        if len(models) > require_num_trees:
            if not slice_to_num_trees:
                raise ValueError(
                    f"Cached index {path} has {len(models)} tree(s), but "
                    f"{require_num_trees} were requested. Pass "
                    "slice_to_num_trees=True to use the prefix explicitly."
                )
            models = models[:require_num_trees]
            trees = trees[:require_num_trees]

    return models, trees


def to_run_plan_dataset_dict(
    dataset_name,
    dataset_path,
    dataset_base_path,
    dataset_query_path,
    dataset_ground_truth_path,
    split,
    subset_size,
    dataset_metric,
    dataset_normalize,
    dataset_storage_mode,
    rerank_backend,
) -> dict[str, Any]:
    """Return the benchmark-standard dataset dictionary."""
    return {
        "name": dataset_name,
        "path": dataset_path,
        "base_path": dataset_base_path,
        "query_path": dataset_query_path,
        "ground_truth_path": dataset_ground_truth_path,
        "split": split,
        "subset_size": subset_size,
        "metric": dataset_metric,
        "normalize": dataset_normalize,
        "storage_mode": dataset_storage_mode,
        "rerank_backend": rerank_backend,
    }
