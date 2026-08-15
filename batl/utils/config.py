"""Configuration dataclasses and YAML loading for BATL experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from batl.constants import (
    DEFAULT_CONVERGENCE_MIN_DELTA,
    DEFAULT_CONVERGENCE_PATIENCE,
    DEFAULT_DATASET_EMBEDDING_DIM,
    DEFAULT_ENSEMBLE_MIN_TREE_MATCHES,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NEIGHBOR_SEARCH_CHUNK_SIZE,
    DEFAULT_NEIGHBOR_SEARCH_SUBSET,
    DEFAULT_NUM_LEAVES,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_TRAINING_NEIGHBORS_TOP_K,
    DEFAULT_WEIGHT_DECAY,
    PAPER_ALTERNATING_INTERVAL,
    PAPER_BALANCE_ALPHA,
    PAPER_BEAM_SIZE,
    PAPER_BRANCHING_FACTOR,
    PAPER_DECODER_FF_DIM,
    PAPER_DROPOUT,
    PAPER_ENCODER_HIDDEN,
    PAPER_ENSEMBLE_NUM_TREES,
    PAPER_MODEL_EMBED_DIM,
    PAPER_NUM_ATTENTION_HEADS,
    PAPER_NUM_DECODER_LAYERS,
    PAPER_TREE_HEIGHT,
)

StorageMode = Literal["auto", "memmap", "preload"]
NeighborSearchBackend = Literal["auto", "faiss_cpu", "faiss_gpu"]
LabelRefresh = Literal["once", "per_cycle"]
TreeUpdateCacheEmbeddings = Literal["auto"] | bool
TreeUpdateBatchSize = Literal["auto"] | int
RerankBackend = Literal["auto", "numpy_cpu", "torch_gpu", "torch_gpu_resident"]
TreeAssignmentMode = Literal["round", "sequential"]
RERANK_BACKENDS = get_args(RerankBackend)
_RERANK_BACKEND_ERROR = (
    "evaluation.rerank_backend must be one of: auto, numpy_cpu, torch_gpu, torch_gpu_resident."
)
LOGGER = logging.getLogger(__name__)

_FIXED_MODEL = {
    "encoder_hidden": (
        PAPER_ENCODER_HIDDEN,
        "paper §3.2.1: encoder is always Linear(d,1024)→ReLU→Linear(1024,256)",
    ),
    "num_decoder_layers": (PAPER_NUM_DECODER_LAYERS, "paper §3.2.2: decoder is always 1 layer"),
}

_FIXED_TRAIN = {
    "alternating_interval": (
        PAPER_ALTERNATING_INTERVAL,
        "paper §3.3: routing model and tree always updated alternately every 2 epochs",
    ),
}

_VARIABLE_MODEL = {
    "embed_dim": (
        PAPER_MODEL_EMBED_DIM,
        "paper §3.2.2 default; change only in ablation studies (e.g. {64,128,256})",
    ),
    "alpha": (
        PAPER_BALANCE_ALPHA,
        "paper baseline; change only in balance-factor ablation studies (e.g. {1.0,1.5,2.0})",
    ),
    "num_trees": (
        PAPER_ENSEMBLE_NUM_TREES,
        "paper §5.1.2 default; override to 1 only for fast/smoke tests",
    ),
}

_VARIABLE_EXPERIMENT = {
    "beam_size": (
        PAPER_BEAM_SIZE,
        "paper §3.4 default; change only when a different candidate set size is needed for a specific dataset",
    ),
}

_VARIABLE_TRAIN = {
    "top_k_neighbors": (
        DEFAULT_TRAINING_NEIGHBORS_TOP_K,
        "paper §3.3 default; change only in training-neighbor ablation studies",
    ),
}


@dataclass
class ModelConfig:
    branching_factor: int = PAPER_BRANCHING_FACTOR
    tree_height: int = PAPER_TREE_HEIGHT
    embedding_dim: int = DEFAULT_DATASET_EMBEDDING_DIM
    encoder_hidden: int = PAPER_ENCODER_HIDDEN
    embed_dim: int = PAPER_MODEL_EMBED_DIM
    num_decoder_layers: int = PAPER_NUM_DECODER_LAYERS
    num_heads: int = PAPER_NUM_ATTENTION_HEADS
    ff_dim: int = PAPER_DECODER_FF_DIM
    dropout: float = PAPER_DROPOUT
    alpha: float = PAPER_BALANCE_ALPHA
    num_trees: int = PAPER_ENSEMBLE_NUM_TREES

    def __post_init__(self) -> None:
        for field_name, (default, reason) in _VARIABLE_MODEL.items():
            actual = getattr(self, field_name)
            if actual != default:
                LOGGER.warning(
                    "ModelConfig.%s=%r differs from default %r. %s",
                    field_name,
                    actual,
                    default,
                    reason,
                )


@dataclass
class TrainConfig:
    batch_size: int = DEFAULT_TRAIN_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    num_epochs: int | None = None
    alternating_interval: int = PAPER_ALTERNATING_INTERVAL
    max_alternating_cycles: int | None = None
    device: str = "cuda"
    top_k_neighbors: int = DEFAULT_TRAINING_NEIGHBORS_TOP_K
    neighbor_search_subset: int = DEFAULT_NEIGHBOR_SEARCH_SUBSET
    neighbor_search_mode: Literal["random_subset", "sequential_chunked"] = "random_subset"
    neighbor_search_chunk_size: int = DEFAULT_NEIGHBOR_SEARCH_CHUNK_SIZE
    convergence_patience: int = DEFAULT_CONVERGENCE_PATIENCE
    convergence_min_delta: float = DEFAULT_CONVERGENCE_MIN_DELTA
    neighbor_search_backend: NeighborSearchBackend = "auto"
    # Query and neighbor identities may be mined once per independent tree or
    # refreshed every alternating cycle. Per-cycle preserves historical runs.
    label_refresh: LabelRefresh = "per_cycle"
    # Explicit user knobs (not paper hyperparameters). Keep
    # tree_update_cache_embeddings=false for very large N. For performance
    # tree-building runs, prefer setting tree_update_batch_size to an explicit
    # integer in the YAML or CLI so the run documents the chosen batch size.
    # The "auto" value exists as a conservative fallback after an observed CUDA
    # attention launch-shape failure; on CUDA, both auto and oversized explicit
    # integers are still clamped by _cuda_attention_batch_guard with a WARN log.
    tree_update_cache_embeddings: TreeUpdateCacheEmbeddings = "auto"
    tree_update_batch_size: TreeUpdateBatchSize | None = "auto"
    # Branches per vector retained by the balanced assignment before a full-K
    # re-decode is needed. Results are identical for any value; only speed and
    # host memory change. None means full K — no truncation, no stragglers, and
    # the same work as before top-R existed. Set it from
    # `min_top_r_covering_999` in the tree-update diagnostics once a build has
    # reported that dataset's chosen-rank tail; too small an R re-decodes more
    # than it saves. DEFAULT_ASSIGNMENT_TOP_R is a starting suggestion, not a
    # measured value.
    tree_update_top_r: int | None = None

    def __post_init__(self) -> None:
        if self.top_k_neighbors <= 0:
            raise ValueError("training.top_k_neighbors must be positive.")
        if self.neighbor_search_subset <= 0:
            raise ValueError("training.neighbor_search_subset must be positive.")
        if self.top_k_neighbors > self.neighbor_search_subset:
            raise ValueError(
                "training.top_k_neighbors cannot exceed training.neighbor_search_subset."
            )
        if self.label_refresh not in {"once", "per_cycle"}:
            raise ValueError("training.label_refresh must be 'once' or 'per_cycle'.")
        for field_name, (default, reason) in _VARIABLE_TRAIN.items():
            actual = getattr(self, field_name)
            if actual != default:
                LOGGER.warning(
                    "TrainConfig.%s=%r differs from default %r. %s",
                    field_name,
                    actual,
                    default,
                    reason,
                )


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    output_dir: str
    dataset_name: str
    dataset_path: str
    split: str
    subset_size: int | None
    recall_at: list[int]
    num_queries: int
    beam_size: int = PAPER_BEAM_SIZE
    num_leaves: list[int] = field(default_factory=lambda: [DEFAULT_NUM_LEAVES])
    min_trees: int | None = None
    tree_assignment_mode: TreeAssignmentMode = "round"
    tree_assignment_order: str = "confidence"
    dataset_base_path: str | None = None
    dataset_query_path: str | None = None
    dataset_ground_truth_path: str | None = None
    dataset_metric: str | None = None
    dataset_normalize: bool = False
    dataset_storage_mode: StorageMode = "auto"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    rerank_backend: RerankBackend = "auto"
    # Stage timing and hardware capture. Off for headline timing runs unless
    # the artifact is labelled as profiled: enabling it adds CUDA
    # synchronization at stage boundaries, which perturbs the number being
    # measured.
    performance_profile: bool = False
    # Repeat the search sweep to expose run-to-run variance. Keep at 1 for
    # ordinary runs; a profiled run wanting a median should use 3 or more.
    search_repetitions: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiment.name must be non-empty.")
        if not isinstance(self.seed, int):
            raise ValueError("experiment.seed must be an integer.")
        elif self.seed < 0:
            raise ValueError("experiment.seed must be non-negative.")
        if not self.output_dir:
            raise ValueError("experiment.output_dir must be non-empty.")

        if not self.dataset_name:
            raise ValueError("dataset.name must be non-empty.")
        if not self.dataset_path:
            raise ValueError("dataset.path must be non-empty.")
        for name, value in {
            "dataset.base_path": self.dataset_base_path,
            "dataset.query_path": self.dataset_query_path,
            "dataset.ground_truth_path": self.dataset_ground_truth_path,
        }.items():
            if value is not None and not value:
                raise ValueError(f"{name} must be non-empty when provided.")
        if not self.split:
            raise ValueError("dataset.split must be non-empty.")
        if self.subset_size is not None and self.subset_size <= 0:
            raise ValueError("dataset.subset_size must be positive when provided.")
        if self.dataset_storage_mode not in {"auto", "memmap", "preload"}:
            raise ValueError("dataset.storage_mode must be one of: auto, memmap, preload.")

        if not self.recall_at:
            raise ValueError("evaluation.recall_at must be non-empty.")
        elif any(k <= 0 for k in self.recall_at):
            raise ValueError("evaluation.recall_at values must be positive.")
        if self.num_queries <= 0:
            raise ValueError("evaluation.num_queries must be positive.")
        if self.beam_size <= 0:
            raise ValueError("evaluation.beam_size must be positive.")
        # Fail here rather than after a multi-GB index load: beam search holds
        # only beam_size prefixes, so M > beam_size cannot return more leaves
        # and used to be capped silently, misreporting the sweep point.
        if any(m <= 0 for m in self.num_leaves):
            raise ValueError("evaluation.num_leaves values must be positive.")
        oversized = [m for m in self.num_leaves if m > self.beam_size]
        if oversized:
            raise ValueError(
                "evaluation.num_leaves values must be <= evaluation.beam_size "
                f"({self.beam_size}); got {oversized}."
            )
        if self.min_trees is not None and not 1 <= self.min_trees <= self.model.num_trees:
            raise ValueError(
                "evaluation.min_trees must be in [1, model.num_trees] when provided; "
                f"got min_trees={self.min_trees}, num_trees={self.model.num_trees}."
            )
        if self.rerank_backend not in RERANK_BACKENDS:
            raise ValueError(_RERANK_BACKEND_ERROR)
        if self.search_repetitions < 1:
            raise ValueError("evaluation.search_repetitions must be >= 1.")
        if self.tree_assignment_mode not in {"round", "sequential"}:
            raise ValueError("evaluation.tree_assignment_mode must be 'round' or 'sequential'.")
        if self.tree_assignment_order not in {"input", "confidence", "margin"}:
            raise ValueError(
                "evaluation.tree_assignment_order must be 'input', 'confidence', or 'margin'."
            )

        for field_name, (default, reason) in _VARIABLE_EXPERIMENT.items():
            actual = getattr(self, field_name)
            if actual != default:
                LOGGER.warning(
                    "ExperimentConfig.%s=%r differs from default %r. %s",
                    field_name,
                    actual,
                    default,
                    reason,
                )

    def resolved_min_trees(self) -> int:
        """Return the configured frequency threshold with paper defaults applied."""
        if self.min_trees is not None:
            return self.min_trees
        if self.model.num_trees == 1:
            return 1
        return DEFAULT_ENSEMBLE_MIN_TREE_MATCHES


def _check_fixed(section_name: str, raw: dict[str, Any], fixed: dict[str, tuple[Any, str]]) -> None:
    for field_name, (expected, reason) in fixed.items():
        if field_name in raw and raw[field_name] != expected:
            raise ValueError(
                f"[{section_name}] {field_name}={raw[field_name]!r} deviates from the fixed paper "
                f"value {expected!r} ({reason}). Remove this field from the YAML config, or edit "
                "the source directly if you are intentionally deviating for research purposes."
            )


def _section(raw: dict[str, Any], name: str, *, required: bool = True) -> dict[str, Any]:
    value = raw.get(name)
    if value is None:
        if required:
            raise ValueError(f"Missing required config section: {name}")
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return value
