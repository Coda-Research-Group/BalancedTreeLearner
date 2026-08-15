"""Config loading and sanity checks for benchmark entrypoints."""

from __future__ import annotations

import argparse
import difflib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, get_args

import faiss
import torch
import yaml

from batl.constants import LARGE_DATASET_RANDOM_SUBSET_THRESHOLD
from batl.utils.arguments import apply_args_to_config
from batl.utils.config import (
    _FIXED_MODEL,
    _FIXED_TRAIN,
    _RERANK_BACKEND_ERROR,
    RERANK_BACKENDS,
    ExperimentConfig,
    ModelConfig,
    NeighborSearchBackend,
    RerankBackend,
    TrainConfig,
    _check_fixed,
    _section,
)

LOGGER = logging.getLogger(__name__)


def load_config_with_device(
    path: str,
    argv: argparse.Namespace,
    device_override: str | None = None,
) -> ExperimentConfig:
    """Load a config and resolve its benchmark device."""
    cfg = load_experiment_config(path)
    apply_args_to_config(cfg, argv)
    cfg.train.device = resolve_device(device_override or cfg.train.device)
    cfg.train.neighbor_search_backend = resolve_neighbor_search_backend(
        cfg.train.neighbor_search_backend
    )
    cfg.rerank_backend = resolve_rerank_backend(cfg.rerank_backend, cfg.train.device)
    return cfg


def resolve_device(requested: str) -> str:
    """Return a usable torch device string, falling back to CPU if needed."""
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def resolve_neighbor_search_backend(requested: str) -> NeighborSearchBackend:
    """Return a usable FAISS backend policy, falling back to CPU."""
    if requested == "auto":
        return "faiss_gpu" if is_faiss_gpu_available() else "faiss_cpu"
    if requested == "faiss_gpu" and not is_faiss_gpu_available():
        return "faiss_cpu"
    valid = get_args(NeighborSearchBackend)
    if requested not in valid:
        raise ValueError(
            f"neighbor_search_backend {requested!r} is not valid; must be one of {valid}."
        )
    return cast(NeighborSearchBackend, requested)


def resolve_rerank_backend(requested: str, device: str) -> RerankBackend:
    """Return a usable rerank backend policy, falling back to NumPy CPU.

    ``auto`` prefers the GPU-resident reranker, which is the fastest path when
    the database fits in VRAM. Whether it actually fits is only knowable once
    the database is loaded, so the caller must catch
    ``batl.rerank.RerankGpuMemoryError`` at construction time and fall back to
    ``numpy_cpu`` from there.
    """
    valid = get_args(RerankBackend)
    if requested not in valid:
        raise ValueError(f"rerank_backend {requested!r} is not valid; must be one of {valid}.")
    cuda_available = device == "cuda" and torch.cuda.is_available()
    if requested == "auto":
        return "torch_gpu_resident" if cuda_available else "numpy_cpu"
    if requested in {"torch_gpu", "torch_gpu_resident"} and not cuda_available:
        return "numpy_cpu"
    return cast(RerankBackend, requested)


def is_faiss_gpu_available() -> bool:
    return hasattr(faiss, "StandardGpuResources") and torch.cuda.is_available()


@dataclass
class FinalConfigSanityChecker:
    """Validate that a config is safe enough for committed benchmark runs.

    This checker enforces reproducibility and artifact-shape invariants, not
    experiment-specific algorithm choices. Dataset-specific rules such as SIFT
    preserving the paper small-scale leaf budget should stay in the benchmark
    script that owns that experiment.
    """

    forbid_legacy_num_epochs: bool = True

    def check(self, cfg: ExperimentConfig) -> list[str]:
        """Return sanity-check problems without raising."""
        errors: list[str] = []

        if self.forbid_legacy_num_epochs and cfg.train.num_epochs is not None:
            errors.append("training.num_epochs must be omitted for final convergence-driven runs.")

        if cfg.model.embed_dim % cfg.model.num_heads != 0:
            errors.append("model.embed_dim must be divisible by model.num_heads.")
        if cfg.model.alpha < 1.0:
            errors.append("model.alpha must be >= 1.0.")
        if any(r <= 0 for r in cfg.recall_at):
            errors.append("evaluation.recall_at values must be positive.")
        if cfg.dataset_storage_mode not in {"auto", "memmap", "preload"}:
            errors.append("dataset.storage_mode must be one of: auto, memmap, preload.")
        if cfg.beam_size <= 0:
            errors.append("evaluation.beam_size must be positive.")
        if cfg.train.neighbor_search_backend not in {"auto", "faiss_cpu", "faiss_gpu"}:
            errors.append(
                "training.neighbor_search_backend must be one of: auto, faiss_cpu, faiss_gpu."
            )
        if str(cfg.train.tree_update_cache_embeddings) not in {
            "auto",
            "True",
            "False",
            "true",
            "false",
        }:
            errors.append(
                "training.tree_update_cache_embeddings must be one of: auto, true, false."
            )
        if cfg.train.tree_update_top_r is not None and cfg.train.tree_update_top_r < 2:
            # Margin-ordered assignment scores need the top two probabilities.
            errors.append("training.tree_update_top_r must be >= 2 or omitted for full K.")
        if (
            cfg.tree_assignment_mode == "sequential"
            and cfg.train.tree_update_top_r is not None
            and cfg.train.tree_update_top_r < cfg.model.branching_factor
        ):
            errors.append(
                "sequential tree assignment requires full-K branch order: "
                "training.tree_update_top_r must be omitted or >= model.branching_factor."
            )
        if cfg.rerank_backend not in RERANK_BACKENDS:
            errors.append(_RERANK_BACKEND_ERROR)
        return errors

    def advisories(self, cfg: ExperimentConfig) -> list[str]:
        """Return concerns that should be logged, not enforced.

        Kept separate from ``check`` because a rule that every committed
        artifact violates cannot be a blocking error without invalidating every
        result already produced.
        """
        advisories: list[str] = []
        if (
            cfg.subset_size is not None
            and cfg.subset_size >= LARGE_DATASET_RANDOM_SUBSET_THRESHOLD
            and cfg.train.neighbor_search_mode == "random_subset"
        ):
            # Advisory rather than an error: all 51 large-scale wrapper scripts
            # in this repo use random_subset at >= 10M, including every run
            # behind the current results. The predicate is also mis-aimed — it
            # keys off the dataset size, while the quantity that drives random
            # memmap I/O is the mining subset (training.neighbor_search_subset)
            # and the storage medium. Revisit with a measurement, then either
            # re-aim the predicate or promote it back to an error.
            advisories.append(
                "training.neighbor_search_mode=random_subset with "
                f"subset_size={cfg.subset_size}: label mining will fancy-index "
                f"{cfg.train.neighbor_search_subset} rows out of a memmap. Prefer "
                "sequential_chunked if mining is slow on this storage."
            )
        return advisories


def run_final_config_sanity_checks(
    cfg: ExperimentConfig,
    *,
    skip: bool = False,
    checker: FinalConfigSanityChecker | None = None,
) -> None:
    """Fail fast on configs that are unsafe for a committed benchmark run."""
    if skip:
        LOGGER.warning("final-config sanity checks skipped by --skip-sanity-checks")
        return
    checker = checker or FinalConfigSanityChecker()
    for advisory in checker.advisories(cfg):
        LOGGER.warning("config advisory: %s", advisory)
    errors = checker.check(cfg)
    if errors:
        raise ValueError("config failed sanity checks:\n- " + "\n- ".join(errors))


def should_preload_dataset(
    *,
    storage_mode: str,
    estimated_nbytes: int,
    available_ram_bytes: int | None = None,
    max_ram_fraction: float = 0.5,
) -> bool:
    """Return whether a dataset payload should be copied into RAM."""
    if storage_mode == "memmap":
        return False
    if storage_mode == "preload":
        return True
    if storage_mode != "auto":
        raise ValueError("storage_mode must be one of: auto, memmap, preload.")
    if available_ram_bytes is None:
        available_ram_bytes = _available_ram_bytes()
    return estimated_nbytes <= int(available_ram_bytes * max_ram_fraction)


_FALLBACK_RAM_BYTES = 8 * 1024**3  # 8 GiB — assumed minimum for local dev machines


def _available_ram_bytes() -> int:
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        kib = int(meminfo.split("MemAvailable:")[1].split()[0])
        return kib * 1024
    except (FileNotFoundError, IndexError, ValueError):
        # /proc/meminfo is Linux-only; assume a reasonable floor for macOS/Windows dev.
        return _FALLBACK_RAM_BYTES


_DATASET_PROVENANCE_KEYS = frozenset({"source_name", "source_url"})
"""Recorded in configs for artifact provenance, not read by ExperimentConfig.

Accepted so the unknown-key check does not reject documented metadata that 18
committed configs carry.
"""


def _accepted_config_keys() -> dict[str, set[str]]:
    """Keys each section may carry, derived from ``ExperimentConfig`` fields.

    Derived rather than hardcoded so a new field never needs a second edit.
    The loader accepts any field name from either ``experiment`` or
    ``evaluation`` (it checks both), so both sections share the same set;
    narrowing that would reject configs that load today.
    """
    from dataclasses import fields

    names = {f.name for f in fields(ExperimentConfig)} - {"model", "train"}
    dataset = {"split", "subset_size"}
    dataset |= {name[len("dataset_") :] for name in names if name.startswith("dataset_")}
    return {
        "experiment": names,
        "evaluation": names,
        "dataset": dataset | set(_DATASET_PROVENANCE_KEYS),
    }


def _reject_unknown_keys(
    *,
    experiment: dict[str, Any],
    dataset: dict[str, Any],
    evaluation: dict[str, Any],
) -> None:
    """Fail on unrecognised keys instead of dropping them.

    ``model`` and ``training`` already raise through their dataclass
    constructors; these three sections were built by name lookup, so a typo
    such as ``num_leavse`` was silently ignored and the run used the default —
    discovered, at best, hours later when the artifact looked wrong.
    """
    accepted = _accepted_config_keys()
    problems: list[str] = []
    for section_name, section in (
        ("experiment", experiment),
        ("dataset", dataset),
        ("evaluation", evaluation),
    ):
        valid = accepted[section_name]
        unknown = sorted(set(section) - valid)
        for key in unknown:
            suggestion = difflib.get_close_matches(key, sorted(valid), n=1, cutoff=0.7)
            hint = f" (did you mean {suggestion[0]!r}?)" if suggestion else ""
            problems.append(f"  {section_name}.{key}{hint}")
    if problems:
        raise ValueError(
            "Unknown config keys:\n"
            + "\n".join(problems)
            + "\nCheck for typos. Valid keys per section:\n"
            + "\n".join(f"  {name}: {sorted(accepted[name])}" for name in sorted(accepted))
        )


def load_experiment_config(path: str, *, strict: bool = True) -> ExperimentConfig:
    """Load a BATL experiment config from YAML.

    strict=True (default) raises ValueError if any fixed paper hyperparameter
    is overridden in the YAML. Set strict=False only for smoke/test configs that
    intentionally use small values for fast local validation.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config {path!r} must contain a YAML mapping.")

    experiment = _section(raw, "experiment")
    dataset = _section(raw, "dataset")
    evaluation = _section(raw, "evaluation")
    model_raw = _section(raw, "model", required=False)
    train_raw = _section(raw, "training", required=False)

    if strict:
        _check_fixed("model", model_raw, _FIXED_MODEL)
        _check_fixed("training", train_raw, _FIXED_TRAIN)

    _reject_unknown_keys(experiment=experiment, dataset=dataset, evaluation=evaluation)

    kwargs = {"model": ModelConfig(**model_raw), "train": TrainConfig(**train_raw)}
    from dataclasses import fields

    for f in fields(ExperimentConfig):
        name = f.name
        if name in ("model", "train"):
            continue
        if name in experiment:
            kwargs[name] = experiment[name]
        elif name in evaluation:
            kwargs[name] = evaluation[name]
        elif name in ("split", "subset_size"):
            if name in dataset:
                kwargs[name] = dataset[name]
        elif name.startswith("dataset_"):
            k = name[8:]
            if k in dataset:
                kwargs[name] = dataset[k]

    kwargs.setdefault("subset_size", None)

    try:
        return ExperimentConfig(**kwargs)
    except TypeError as e:
        raise ValueError(f"Missing required config fields: {e}") from e
