import argparse
import json

import numpy as np
import pytest

from batl.constants import LARGE_DATASET_RANDOM_SUBSET_THRESHOLD
from batl.model import BATLModel
from batl.search import search_batch
from batl.tree import BATLTree
from batl.utils.arguments import (
    add_batch_search_arg,
    add_batch_train_arg,
    add_config_arg,
    add_datapath_arg,
    add_index_path_arg,
    add_log_arg,
    add_n_queries_arg,
    add_num_leaves_arg,
    add_result_dir_arg,
    non_negative_int,
    positive_int,
)
from batl.utils.config import ExperimentConfig, ModelConfig, TrainConfig
from batl.utils.config_parsing import (
    FinalConfigSanityChecker,
    resolve_device,
    resolve_neighbor_search_backend,
    resolve_rerank_backend,
    should_preload_dataset,
)
from batl.utils.index_parsing import load_index, merge_indexes, save_index
from batl.utils.io import jsonable, save_benchmark_artifacts, write_rows
from batl.utils.logging_utils import (
    print_query_progress,
    search_logging,
    standard_run_metadata,
)
from build import (
    batl_index_path,
    build_batl_index,
    load_batl_index_checked,
    tree_index_path,
)


def _config(k: int = 2, h: int = 2) -> ExperimentConfig:
    return ExperimentConfig(
        name="bench",
        seed=123,
        output_dir="experiments/results/test",
        dataset_name="synthetic",
        dataset_path="experiments/data/synthetic",
        split="train",
        subset_size=4,
        recall_at=[10],
        num_queries=2,
        model=ModelConfig(branching_factor=k, tree_height=h),
    )


def _tiny_training_config(tmp_path, *, num_trees: int = 2) -> ExperimentConfig:
    return ExperimentConfig(
        name="merge_equivalence",
        seed=42,
        output_dir=str(tmp_path),
        dataset_name="synthetic",
        dataset_path=str(tmp_path / "synthetic"),
        split="train",
        subset_size=8,
        recall_at=[1],
        num_queries=4,
        beam_size=4,
        num_leaves=[4],
        model=ModelConfig(
            branching_factor=2,
            tree_height=2,
            embedding_dim=3,
            encoder_hidden=8,
            embed_dim=8,
            num_heads=2,
            ff_dim=16,
            dropout=0.0,
            alpha=1.0,
            num_trees=num_trees,
        ),
        train=TrainConfig(
            batch_size=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            num_epochs=1,
            alternating_interval=1,
            max_alternating_cycles=1,
            convergence_patience=0,
            device="cpu",
            top_k_neighbors=1,
            neighbor_search_subset=8,
            neighbor_search_backend="faiss_cpu",
            tree_update_cache_embeddings=False,
        ),
    )


def _save_index(path, *, n_trees: int = 1, k: int = 2, h: int = 2, n: int = 4) -> None:
    trees = [BATLTree.random_init(N=n, K=k, H=h, alpha=1.0, seed=seed) for seed in range(n_trees)]
    models = [
        BATLModel(
            ModelConfig(
                branching_factor=k,
                tree_height=h,
                embedding_dim=2,
                encoder_hidden=4,
                embed_dim=4,
                num_heads=2,
                ff_dim=8,
                dropout=0.0,
                num_trees=1,
            )
        )
        for _ in range(n_trees)
    ]
    save_index(models=models, trees=trees, path=str(path))


def test_jsonable_handles_numpy_nested_lists_and_none() -> None:
    value = {
        "scalar": np.float32(1.5),
        "count": np.int64(7),
        "items": [np.int32(1), {"nested": np.float64(2.25)}],
        "array": np.array([1, 2], dtype=np.int64),
        "none": None,
    }

    converted = jsonable(value)

    assert converted == {
        "scalar": pytest.approx(1.5),
        "count": 7,
        "items": [1, {"nested": 2.25}],
        "array": [1, 2],
        "none": None,
    }
    json.dumps(converted)


def test_label_refresh_is_recorded_in_search_row(tmp_path) -> None:
    config = _config()
    config.train.label_refresh = "once"

    tree = BATLTree.random_init(N=16, K=2, H=2, alpha=1.0, seed=7)
    ids = np.tile(np.arange(10, dtype=np.int64), (2, 1))
    row = search_logging(
        config,
        "fixed-labels",
        [tree],
        2,
        ids,
        ids,
        np.array([3, 4], dtype=np.int64),
        1.0,
        0.5,
        tmp_path / "index.pkl",
        16,
        10,
    )

    assert row["label_refresh"] == "once"
    assert row["min_trees"] == 1


def test_resolve_device_falls_back_when_accelerator_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    assert resolve_device("mps") == "cpu"
    assert resolve_device("cuda") == "cpu"
    assert resolve_device("cpu") == "cpu"


def test_backend_resolution_preserves_cpu_fallback(monkeypatch) -> None:
    monkeypatch.setattr("batl.utils.config_parsing.is_faiss_gpu_available", lambda: False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    assert resolve_neighbor_search_backend("auto") == "faiss_cpu"
    assert resolve_neighbor_search_backend("faiss_gpu") == "faiss_cpu"
    assert resolve_neighbor_search_backend("faiss_cpu") == "faiss_cpu"
    assert resolve_rerank_backend("auto", "cpu") == "numpy_cpu"
    assert resolve_rerank_backend("torch_gpu", "cpu") == "numpy_cpu"
    assert resolve_rerank_backend("numpy_cpu", "cpu") == "numpy_cpu"


def test_backend_resolution_passthrough_without_mock() -> None:
    # Exercises _faiss_gpu_available and the passthrough returns directly.
    result = resolve_neighbor_search_backend("auto")
    assert result in ("faiss_cpu", "faiss_gpu")
    result2 = resolve_rerank_backend("numpy_cpu", "cuda")
    assert result2 == "numpy_cpu"


def test_should_preload_dataset_honors_policy_and_ram_budget() -> None:
    assert should_preload_dataset(
        storage_mode="preload",
        estimated_nbytes=10**12,
        available_ram_bytes=1,
    )
    assert not should_preload_dataset(
        storage_mode="memmap",
        estimated_nbytes=1,
        available_ram_bytes=10**12,
    )
    assert should_preload_dataset(
        storage_mode="auto",
        estimated_nbytes=100,
        available_ram_bytes=1_000,
        max_ram_fraction=0.5,
    )
    assert not should_preload_dataset(
        storage_mode="auto",
        estimated_nbytes=600,
        available_ram_bytes=1_000,
        max_ram_fraction=0.5,
    )


def test_standard_run_metadata_includes_environment_and_hardware(monkeypatch) -> None:
    metadata = standard_run_metadata("cpu")

    assert "python" in metadata["environment"]
    assert "torch" in metadata["environment"]
    assert metadata["hardware"]["device"] == "cpu"


def test_print_query_progress_uses_standard_format(capsys) -> None:
    print_query_progress(label="bench", done=5, total=10, elapsed_s=2.0)

    out = capsys.readouterr().out.strip()
    assert out == "bench: queries done 5/10, remaining 5, elapsed 2.0s, 2.50 q/s"


def test_positive_and_non_negative_argparse_types() -> None:
    assert positive_int("3") == 3
    assert non_negative_int("0") == 0
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError):
        non_negative_int("-1")


def test_arg_adders_parse_build_and_search_options() -> None:
    build_parser = argparse.ArgumentParser()
    add_config_arg(build_parser)
    add_datapath_arg(build_parser)
    add_log_arg(build_parser)
    add_result_dir_arg(build_parser)
    add_index_path_arg(build_parser)
    add_batch_train_arg(build_parser)

    build_args = build_parser.parse_args(
        [
            "experiments/configs/sift1m/sift1m_h2_paper.yaml",
            "--datapath",
            "/data/sift",
            "--result-dir",
            "/tmp/results",
            "--index-path",
            "/tmp/index.pkl",
            "--batch-train",
            "512",
        ]
    )
    assert build_args.config == "experiments/configs/sift1m/sift1m_h2_paper.yaml"
    assert build_args.datapath == "/data/sift"
    assert build_args.result_dir == "/tmp/results"
    assert build_args.index_path == "/tmp/index.pkl"
    assert build_args.batch_train == 512

    search_parser = argparse.ArgumentParser()
    add_config_arg(search_parser)
    add_num_leaves_arg(search_parser)
    add_n_queries_arg(search_parser)
    add_batch_search_arg(search_parser)

    search_args = search_parser.parse_args(
        [
            "experiments/configs/sift1m/sift1m_h2_paper.yaml",
            "--num-leaves",
            "10",
            "20",
            "40",
            "--n-queries",
            "1000",
            "--batch-search",
            "50",
        ]
    )
    assert search_args.num_leaves == [10, 20, 40]
    assert search_args.n_queries == 1000
    assert search_args.batch_search == 50


def test_final_config_sanity_checker_reports_reproducibility_problems() -> None:
    config = _config()
    config.dataset_name = "sift1m"
    config.train.num_epochs = 10

    errors = FinalConfigSanityChecker().check(config)

    assert "training.num_epochs must be omitted for final convergence-driven runs." in errors


def test_final_config_sanity_checker_validates_shape_and_positive_values() -> None:
    config = _config()
    config.model.embed_dim = 31
    config.model.num_heads = 8
    config.model.alpha = 0.5
    config.beam_size = 0
    config.recall_at = [0]
    config.dataset_storage_mode = "ram"
    config.train.neighbor_search_backend = "gpu"
    config.train.tree_update_cache_embeddings = "yes"
    config.rerank_backend = "cuda"

    errors = FinalConfigSanityChecker().check(config)

    assert "model.embed_dim must be divisible by model.num_heads." in errors
    assert "model.alpha must be >= 1.0." in errors
    assert "evaluation.beam_size must be positive." in errors
    assert "evaluation.recall_at values must be positive." in errors
    assert "dataset.storage_mode must be one of: auto, memmap, preload." in errors
    assert "training.neighbor_search_backend must be one of: auto, faiss_cpu, faiss_gpu." in errors
    assert "training.tree_update_cache_embeddings must be one of: auto, true, false." in errors
    assert (
        "evaluation.rerank_backend must be one of: "
        "auto, numpy_cpu, torch_gpu, torch_gpu_resident." in errors
    )


def test_large_random_subset_memmap_io_is_an_advisory_not_an_error() -> None:
    """Demoted from error on 2026-08-05.

    All 51 large-scale wrapper scripts use random_subset at >= 10M, including
    every run behind the current results, so enforcing it would reject the
    whole benchmark suite. The predicate is also mis-aimed: it keys off dataset
    size, while random memmap I/O is driven by the mining subset.
    """
    config = _config()
    config.dataset_name = "deep1b"
    config.subset_size = LARGE_DATASET_RANDOM_SUBSET_THRESHOLD
    config.train.neighbor_search_mode = "random_subset"
    checker = FinalConfigSanityChecker()

    assert checker.check(config) == []
    advisories = checker.advisories(config)
    assert len(advisories) == 1
    assert "random_subset" in advisories[0]
    assert "sequential_chunked" in advisories[0]

    config.train.neighbor_search_mode = "sequential_chunked"
    assert checker.advisories(config) == []


def test_write_rows_writes_json_and_csv(tmp_path) -> None:
    rows = [{"method": "batl", "recall@10": np.float32(0.5), "n": np.int64(3)}]

    write_rows(tmp_path, "beam_rows", rows)

    assert json.loads((tmp_path / "beam_rows.json").read_text()) == [
        {"method": "batl", "recall@10": 0.5, "n": 3}
    ]
    csv_text = (tmp_path / "beam_rows.csv").read_text()
    assert "method,recall@10,n" in csv_text
    assert "batl,0.5,3" in csv_text


def test_save_benchmark_artifacts_includes_reports(tmp_path) -> None:
    config = _config()
    run_plan = {"mode": "test"}

    save_benchmark_artifacts(
        output_dir=tmp_path,
        rows=[],
        run_plan=run_plan,
        cfg=config,
        seed=config.seed,
        run_metadata={
            "environment": {"git_commit": "abc123"},
            "hardware": {"device": "cpu"},
        },
        config_report={"config_paths": ["experiments/configs/test.yaml"]},
        extra_metrics={"recall@10": np.float32(0.5)},
    )

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["run_plan"] == run_plan
    assert metrics["config_report"]["config_paths"] == ["experiments/configs/test.yaml"]
    assert metrics["recall@10"] == 0.5


def test_load_batl_index_checked_rejects_k_h_n_mismatch(tmp_path) -> None:
    index_path = tmp_path / "index.pkl"
    _save_index(index_path, k=2, h=2, n=4)

    with pytest.raises(ValueError, match="does not match config"):
        load_batl_index_checked(index_path, _config(k=3, h=2), expected_n=4)
    with pytest.raises(ValueError, match="does not match config"):
        load_batl_index_checked(index_path, _config(k=2, h=3), expected_n=4)
    with pytest.raises(ValueError, match="does not match config"):
        load_batl_index_checked(index_path, _config(k=2, h=2), expected_n=5)


def test_load_batl_index_checked_rejects_fewer_trees_than_required(tmp_path) -> None:
    index_path = tmp_path / "index.pkl"
    _save_index(index_path, n_trees=1)

    with pytest.raises(ValueError, match="only 1/2 required"):
        load_batl_index_checked(
            index_path,
            _config(),
            expected_n=4,
            require_num_trees=2,
        )


def test_load_batl_index_checked_requires_explicit_slice_for_extra_trees(tmp_path) -> None:
    index_path = tmp_path / "index.pkl"
    _save_index(index_path, n_trees=2)

    with pytest.raises(ValueError, match="slice_to_num_trees=True"):
        load_batl_index_checked(
            index_path,
            _config(),
            expected_n=4,
            require_num_trees=1,
        )

    models, trees = load_batl_index_checked(
        index_path,
        _config(),
        expected_n=4,
        require_num_trees=1,
        slice_to_num_trees=True,
    )
    assert len(models) == 1
    assert len(trees) == 1


def test_build_batl_index_reuses_complete_cached_prefix(tmp_path) -> None:
    index_path = tmp_path / "index.pkl"
    _save_index(index_path, n_trees=2)

    cfg = _config()
    cfg.model.num_trees = 2
    models, trees, train_time_s = build_batl_index(
        cfg=cfg,
        vectors=np.zeros((4, 2), dtype=np.float32),
        index_path=index_path,
    )

    assert len(models) == 2
    assert len(trees) == 2
    assert train_time_s is None


def test_batl_index_path_separates_assignment_order_caches(tmp_path) -> None:
    assert batl_index_path(tmp_path, "input") == tmp_path / "index_input.pkl"
    assert batl_index_path(tmp_path, "confidence") == tmp_path / "index_confidence.pkl"
    assert batl_index_path(tmp_path, "margin") == tmp_path / "index_margin.pkl"
    assert batl_index_path(tmp_path, "input", "sequential") == (
        tmp_path / "index_sequential_input.pkl"
    )


def test_tree_index_path_adds_tree_suffix_before_extension(tmp_path) -> None:
    assert tree_index_path(tmp_path / "index_confidence.pkl", 3) == (
        tmp_path / "index_confidence_tree_3.pkl"
    )


def test_build_batl_index_reuses_explicit_assignment_order_cache(tmp_path) -> None:
    index_path = tmp_path / "index.pkl"
    _save_index(index_path, n_trees=1)

    cfg = _config()
    cfg.model.num_trees = 1
    models, trees, train_time_s = build_batl_index(
        cfg=cfg,
        vectors=np.zeros((4, 2), dtype=np.float32),
        index_path=index_path,
        assignment_order="confidence",
    )

    assert len(models) == 1
    assert len(trees) == 1
    assert train_time_s is None


def test_build_batl_index_require_cached_rejects_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Evaluation-only mode will not train"):
        build_batl_index(
            cfg=_config(),
            vectors=np.zeros((4, 2), dtype=np.float32),
            index_path=tmp_path / "missing.pkl",
            require_cached_index=True,
        )


def test_build_batl_index_tree_index_builds_one_seeded_tree(tmp_path) -> None:
    cfg = _tiny_training_config(tmp_path, num_trees=2)
    vectors = np.arange(24, dtype=np.float32).reshape(8, 3) / 10
    base_path = tmp_path / "index_confidence.pkl"

    models, trees, train_time_s = build_batl_index(
        cfg=cfg,
        vectors=vectors,
        index_path=base_path,
        assignment_order="confidence",
        tree_index=1,
    )

    assert train_time_s is not None
    assert len(models) == 1
    assert len(trees) == 1
    assert not base_path.exists()
    assert tree_index_path(base_path, 1).exists()


def test_build_batl_index_rejects_tree_index_outside_config_range(tmp_path) -> None:
    cfg = _tiny_training_config(tmp_path, num_trees=2)
    with pytest.raises(ValueError, match="tree_index"):
        build_batl_index(
            cfg=cfg,
            vectors=np.zeros((8, 3), dtype=np.float32),
            index_path=tmp_path / "index.pkl",
            tree_index=2,
        )


def test_parallel_tree_index_merge_matches_sequential_search_results(tmp_path) -> None:
    cfg = _tiny_training_config(tmp_path, num_trees=2)
    vectors = np.arange(24, dtype=np.float32).reshape(8, 3) / 10
    queries = vectors[:4] + np.float32(0.001)

    sequential_path = tmp_path / "sequential.pkl"
    sequential_models, sequential_trees, _ = build_batl_index(
        cfg=cfg,
        vectors=vectors,
        index_path=sequential_path,
        assignment_order="confidence",
    )

    base_parallel_path = tmp_path / "parallel.pkl"
    for tree_index in range(2):
        build_batl_index(
            cfg=cfg,
            vectors=vectors,
            index_path=base_parallel_path,
            assignment_order="confidence",
            tree_index=tree_index,
        )

    merged_path = tmp_path / "merged.pkl"
    merge_indexes(
        [tree_index_path(base_parallel_path, 0), tree_index_path(base_parallel_path, 1)],
        merged_path,
    )
    merged_models, merged_trees = load_index(str(merged_path))

    sequential_results = search_batch(
        sequential_models,
        sequential_trees,
        vectors,
        queries,
        beam_size=4,
        top_k=2,
    )
    merged_results = search_batch(
        merged_models,
        merged_trees,
        vectors,
        queries,
        beam_size=4,
        top_k=2,
    )

    assert np.array_equal(merged_results, sequential_results)
