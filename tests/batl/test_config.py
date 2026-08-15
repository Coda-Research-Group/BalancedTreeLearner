import pytest

from batl.constants import (
    DEFAULT_ENSEMBLE_MIN_TREE_MATCHES,
    DEFAULT_TRAINING_NEIGHBORS_TOP_K,
    PAPER_BEAM_SIZE,
    PAPER_ENSEMBLE_NUM_TREES,
)
from batl.utils.config import ExperimentConfig, ModelConfig, TrainConfig
from batl.utils.config_parsing import load_experiment_config


def test_default_model_config_values_are_available_when_section_omitted(tmp_path) -> None:
    path = tmp_path / "minimal.yaml"
    path.write_text(
        """
experiment:
  name: minimal
  seed: 7
  output_dir: experiments/results/minimal
dataset:
  name: deep1b
  path: experiments/data/deep
  split: train
  subset_size: null
evaluation:
  recall_at: [10]
  num_queries: 100
""",
        encoding="utf-8",
    )

    config = load_experiment_config(str(path))

    assert config.model.branching_factor == 256
    assert config.model.tree_height == 2
    assert config.model.num_trees == PAPER_ENSEMBLE_NUM_TREES
    assert config.train.top_k_neighbors == DEFAULT_TRAINING_NEIGHBORS_TOP_K
    assert config.beam_size == PAPER_BEAM_SIZE
    assert config.train.num_epochs is None
    assert config.train.neighbor_search_mode == "random_subset"
    assert config.train.neighbor_search_chunk_size == 1_000_000
    assert config.train.convergence_patience == 3
    assert config.dataset_storage_mode == "auto"
    assert config.train.neighbor_search_backend == "auto"
    assert config.train.tree_update_cache_embeddings == "auto"
    assert config.train.label_refresh == "per_cycle"
    assert config.rerank_backend == "auto"
    assert config.tree_assignment_mode == "round"
    assert config.min_trees is None
    assert config.resolved_min_trees() == DEFAULT_ENSEMBLE_MIN_TREE_MATCHES


def test_load_config_preserves_dataset_metric_and_normalize_flags(tmp_path) -> None:
    path = tmp_path / "glove.yaml"
    path.write_text(
        """
experiment:
  name: glove
  seed: 42
  output_dir: experiments/results/glove
dataset:
  name: glove100
  path: experiments/data/glove-100-angular.hdf5
  split: train
  subset_size: 1183514
  source_name: ANN-Benchmarks glove-100-angular
  source_url: https://github.com/erikbern/ann-benchmarks
  metric: angular
  normalize: true
  storage_mode: preload
evaluation:
  recall_at: [10]
  num_queries: 10000
  rerank_backend: torch_gpu
training:
  neighbor_search_backend: faiss_gpu
  tree_update_cache_embeddings: true
""",
        encoding="utf-8",
    )

    config = load_experiment_config(str(path))

    assert config.dataset_metric == "angular"
    assert config.dataset_normalize is True
    assert config.dataset_storage_mode == "preload"
    assert config.train.neighbor_search_backend == "faiss_gpu"
    assert config.train.tree_update_cache_embeddings is True
    assert config.rerank_backend == "torch_gpu"


def test_load_config_preserves_dataset_manifest_paths(tmp_path) -> None:
    path = tmp_path / "deep1b_manifest.yaml"
    path.write_text(
        """
experiment:
  name: deep1b_manifest
  seed: 42
  output_dir: experiments/results/deep
dataset:
  name: deep
  path: experiments/data/deep
  base_path: /data/deep/base.fbin
  query_path: /data/deep/query.fbin
  ground_truth_path: /data/deep/groundtruth.ibin
  split: train
  subset_size: 100000000
evaluation:
  recall_at: [10]
  num_queries: 10000
""",
        encoding="utf-8",
    )

    config = load_experiment_config(str(path))

    assert config.dataset_path == "experiments/data/deep"
    assert config.dataset_base_path == "/data/deep/base.fbin"
    assert config.dataset_query_path == "/data/deep/query.fbin"
    assert config.dataset_ground_truth_path == "/data/deep/groundtruth.ibin"


def test_load_config_preserves_neighbor_search_mode(tmp_path) -> None:
    path = tmp_path / "deep.yaml"
    path.write_text(
        """
experiment:
  name: deep
  seed: 42
  output_dir: experiments/results/deep
dataset:
  name: deep
  path: experiments/data/deep/base.fvecs
  split: train
  subset_size: null
evaluation:
  recall_at: [10]
  num_queries: 10000
training:
  neighbor_search_mode: sequential_chunked
  neighbor_search_chunk_size: 128
""",
        encoding="utf-8",
    )

    config = load_experiment_config(str(path))

    assert config.train.neighbor_search_mode == "sequential_chunked"
    assert config.train.neighbor_search_chunk_size == 128


def test_load_config_preserves_once_label_refresh(tmp_path) -> None:
    path = tmp_path / "fixed_labels.yaml"
    path.write_text(
        """
experiment:
  name: fixed_labels
  seed: 42
  output_dir: out
dataset:
  name: sift1m
  path: sift.hdf5
  split: train
  subset_size: 1000000
training:
  label_refresh: once
evaluation:
  recall_at: [10]
  num_queries: 10000
""",
        encoding="utf-8",
    )

    config = load_experiment_config(str(path))

    assert config.train.label_refresh == "once"


def test_train_config_rejects_unknown_label_refresh() -> None:
    with pytest.raises(ValueError, match="label_refresh"):
        TrainConfig(label_refresh="never")  # type: ignore[arg-type]


def test_load_config_preserves_sequential_tree_assignment_mode(tmp_path) -> None:
    path = tmp_path / "sequential.yaml"
    path.write_text(
        """
experiment:
  name: sequential
  seed: 42
  output_dir: out
dataset:
  name: sift1m
  path: sift.hdf5
  split: train
  subset_size: 1000000
evaluation:
  recall_at: [10]
  num_queries: 10000
  tree_assignment_mode: sequential
  tree_assignment_order: input
""",
        encoding="utf-8",
    )

    config = load_experiment_config(str(path))

    assert config.tree_assignment_mode == "sequential"
    assert config.tree_assignment_order == "input"


def test_load_config_rejects_unknown_tree_assignment_mode(tmp_path) -> None:
    path = tmp_path / "bad_assignment_mode.yaml"
    path.write_text(
        """
experiment:
  name: bad
  seed: 42
  output_dir: out
dataset:
  name: sift1m
  path: sift.hdf5
  split: train
evaluation:
  recall_at: [10]
  num_queries: 100
  tree_assignment_mode: simultaneous
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tree_assignment_mode"):
        load_experiment_config(str(path))


# --- Fixed-field guard tests (checked on YAML load, not direct construction) ---


def test_load_config_rejects_yaml_with_wrong_encoder_hidden(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
experiment:
  name: x
  seed: 0
  output_dir: out
dataset:
  name: x
  path: p
  split: train
  subset_size: null
evaluation:
  recall_at: [10]
  num_queries: 10
model:
  encoder_hidden: 512
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="encoder_hidden"):
        load_experiment_config(str(path))


def test_load_config_rejects_yaml_with_wrong_num_decoder_layers(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
experiment:
  name: x
  seed: 0
  output_dir: out
dataset:
  name: x
  path: p
  split: train
  subset_size: null
evaluation:
  recall_at: [10]
  num_queries: 10
model:
  num_decoder_layers: 2
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="num_decoder_layers"):
        load_experiment_config(str(path))


def test_load_config_rejects_yaml_with_wrong_alternating_interval(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
experiment:
  name: x
  seed: 0
  output_dir: out
dataset:
  name: x
  path: p
  split: train
  subset_size: null
evaluation:
  recall_at: [10]
  num_queries: 10
training:
  alternating_interval: 4
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="alternating_interval"):
        load_experiment_config(str(path))


def test_load_config_allows_top_k_neighbors_ablation(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "top_k_ablation.yaml"
    path.write_text(
        """
experiment:
  name: x
  seed: 0
  output_dir: out
dataset:
  name: x
  path: p
  split: train
  subset_size: null
evaluation:
  recall_at: [10]
  num_queries: 10
training:
  top_k_neighbors: 50
""",
        encoding="utf-8",
    )

    with caplog.at_level("WARNING", logger="batl.utils.config"):
        config = load_experiment_config(str(path))

    assert config.train.top_k_neighbors == 50
    assert "top_k_neighbors" in caplog.text
    assert "ablation" in caplog.text


@pytest.mark.parametrize("top_k", [0, -1])
def test_train_config_rejects_non_positive_top_k_neighbors(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k_neighbors must be positive"):
        TrainConfig(top_k_neighbors=top_k)


def test_train_config_rejects_top_k_larger_than_neighbor_subset() -> None:
    with pytest.raises(ValueError, match=r"cannot exceed .*neighbor_search_subset"):
        TrainConfig(top_k_neighbors=101, neighbor_search_subset=100)


def test_load_config_preserves_explicit_frequency_threshold(tmp_path) -> None:
    path = tmp_path / "frequency_threshold.yaml"
    path.write_text(
        """
experiment:
  name: threshold
  seed: 0
  output_dir: out
dataset:
  name: x
  path: p
  split: train
evaluation:
  recall_at: [10]
  num_queries: 10
  min_trees: 3
model:
  num_trees: 4
""",
        encoding="utf-8",
    )

    config = load_experiment_config(str(path))

    assert config.min_trees == 3
    assert config.resolved_min_trees() == 3


@pytest.mark.parametrize("min_trees", [0, 5])
def test_experiment_config_rejects_invalid_frequency_threshold(min_trees: int) -> None:
    with pytest.raises(ValueError, match="min_trees"):
        ExperimentConfig(
            name="invalid-threshold",
            seed=0,
            output_dir="out",
            dataset_name="x",
            dataset_path="p",
            split="train",
            subset_size=None,
            recall_at=[10],
            num_queries=10,
            min_trees=min_trees,
        )


# --- Variable-field warning tests ---


def test_model_config_warns_on_non_default_embed_dim(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="batl.utils.config"):
        ModelConfig(embed_dim=128)
    assert "embed_dim" in caplog.text


def test_model_config_warns_on_non_default_alpha(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="batl.utils.config"):
        ModelConfig(alpha=1.5)
    assert "alpha" in caplog.text


def test_model_config_warns_on_non_default_num_trees(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="batl.utils.config"):
        ModelConfig(num_trees=1)
    assert "num_trees" in caplog.text


def test_experiment_config_warns_on_non_default_beam_size(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="batl.utils.config"):
        ExperimentConfig(
            name="test",
            seed=0,
            output_dir="out",
            dataset_name="x",
            dataset_path="p",
            split="train",
            subset_size=None,
            recall_at=[10],
            num_queries=10,
            beam_size=50,
            # The default num_leaves is [80], which now exceeds this beam.
            num_leaves=[50],
        )
    assert "beam_size" in caplog.text


def test_model_config_no_warning_at_defaults(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="batl.utils.config"):
        ModelConfig()
    assert caplog.text == ""


def test_train_config_no_warning_at_defaults(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="batl.utils.config"):
        TrainConfig()
    assert caplog.text == ""
