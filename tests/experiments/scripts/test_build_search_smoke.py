import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import yaml


def _load_entrypoint(path: str, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load entrypoint: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_then_search_on_tiny_local_smoke_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    result_dir = tmp_path / "results"
    index_path = tmp_path / "index.pkl"
    config_path = tmp_path / "smoke.yaml"

    rng = np.random.default_rng(123)
    vectors = rng.normal(size=(128, 4)).astype(np.float32)
    queries = (vectors[:3] + rng.normal(scale=0.01, size=(3, 4))).astype(np.float32)
    distances = np.linalg.norm(queries[:, None, :] - vectors[None, :, :], axis=2)
    ground_truth = np.argsort(distances, axis=1).astype(np.int64)

    data_dir.mkdir()
    np.save(data_dir / "vectors.npy", vectors)
    np.save(data_dir / "queries.npy", queries)
    np.save(data_dir / "groundtruth.npy", ground_truth)

    config = {
        "experiment": {
            "name": "tiny_build_search_smoke",
            "seed": 123,
            "output_dir": str(result_dir),
        },
        "dataset": {
            "name": "synthetic",
            "path": str(data_dir),
            "split": "train",
            "subset_size": 128,
            "metric": "euclidean",
            "storage_mode": "preload",
        },
        "model": {
            "branching_factor": 4,
            "tree_height": 2,
            "embedding_dim": 4,
            "num_trees": 1,
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-5,
            "max_alternating_cycles": 1,
            "neighbor_search_subset": 128,
            "neighbor_search_backend": "faiss_cpu",
            "tree_update_cache_embeddings": False,
            "device": "cpu",
        },
        "evaluation": {
            "recall_at": [1],
            "num_queries": 3,
            "beam_size": 4,
            "num_leaves": [1],
            "rerank_backend": "numpy_cpu",
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    build_entrypoint = _load_entrypoint("build.py", "batl_build_entrypoint")
    search_entrypoint = _load_entrypoint("search.py", "batl_search_entrypoint")

    build_entrypoint.main(
        [
            str(config_path),
            "--index-path",
            str(index_path),
            "--result-dir",
            str(result_dir),
            "--cycle-diagnostics",
            "--cycle-diagnostics-queries",
            "2",
            "--cycle-diagnostics-loss-pairs",
            "16",
        ]
    )
    assert index_path.exists()
    assert (result_dir / "metrics.json").exists()
    diagnostic_rows = [
        json.loads(line)
        for line in (result_dir / "cycle_diagnostics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["stage"] for row in diagnostic_rows] == [
        "model_train_epoch",
        "model_train_epoch",
        "after_model_training",
        "after_tree_update",
    ]
    assert sum("loss" in row or "eval_loss" in row for row in diagnostic_rows) == 4
    assert sum("recall@1_top1_bucket" in row for row in diagnostic_rows) == 2

    # The chosen-rank histogram has to survive all the way into the artifact:
    # sizing top-R assignment is the whole reason it is collected.
    tree_rows = [row for row in diagnostic_rows if row["stage"] == "after_tree_update"]
    assert len(tree_rows) == 1
    levels = tree_rows[0]["tree_update_levels"]
    assert [level["level"] for level in levels] == [0, 1]
    for level in levels:
        assert level["rank_hist_rank_0"] >= 0
        assert "max_chosen_rank" in level
        assert "min_top_r_covering_999" in level
    assert tree_rows[0]["assignment_order"] == "confidence"
    # Every non-fallback vector lands in exactly one bucket.
    histogram_total = sum(value for key, value in levels[0].items() if key.startswith("rank_hist_"))
    assert histogram_total == levels[0]["num_vectors"] - levels[0]["fallback_count"]

    search_entrypoint.main(
        [
            str(config_path),
            "--index-path",
            str(index_path),
            "--result-dir",
            str(result_dir),
            "--n-queries",
            "2",
            "--num-leaves",
            "1",
            "--batch-search",
            "0",
        ]
    )

    rows = json.loads((result_dir / "search_rows.json").read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["n_queries"] == 2
    assert rows[0]["knob_name"] == "num_leaves"
    assert rows[0]["knob_value"] == 1
    assert rows[0]["recall@10"] >= 0.0


def test_build_entrypoint_does_not_require_queries_or_ground_truth(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    result_dir = tmp_path / "results"
    index_path = tmp_path / "index.pkl"
    config_path = tmp_path / "build_only.yaml"

    rng = np.random.default_rng(321)
    vectors = rng.normal(size=(128, 4)).astype(np.float32)
    data_dir.mkdir()
    np.save(data_dir / "vectors.npy", vectors)

    config = {
        "experiment": {
            "name": "tiny_build_only",
            "seed": 321,
            "output_dir": str(result_dir),
        },
        "dataset": {
            "name": "synthetic",
            "path": str(data_dir),
            "split": "train",
            "subset_size": 128,
            "metric": "euclidean",
            "storage_mode": "preload",
        },
        "model": {
            "branching_factor": 4,
            "tree_height": 2,
            "embedding_dim": 4,
            "num_trees": 1,
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-5,
            "max_alternating_cycles": 1,
            "neighbor_search_subset": 128,
            "neighbor_search_backend": "faiss_cpu",
            "tree_update_cache_embeddings": False,
            "device": "cpu",
        },
        "evaluation": {
            "recall_at": [10],
            "num_queries": 2,
            "beam_size": 4,
            "num_leaves": [1],
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    build_entrypoint = _load_entrypoint("build.py", "batl_build_only_entrypoint")
    build_entrypoint.main(
        [
            str(config_path),
            "--index-path",
            str(index_path),
            "--result-dir",
            str(result_dir),
        ]
    )

    assert index_path.exists()
    assert (result_dir / "metrics.json").exists()
    assert (result_dir / "config.yaml").exists()
    assert (result_dir / "seed.txt").read_text(encoding="utf-8") == "321\n"


def test_search_progress_passes_search_policy_to_search_batch(
    monkeypatch,
) -> None:
    search_entrypoint = _load_entrypoint("search.py", "batl_search_backend_entrypoint")
    captured = {}

    def fake_search_batch(**kwargs):
        captured.update(kwargs)
        return (
            np.array([[0]], dtype=np.int64),
            np.array([1], dtype=np.int64),
        )

    monkeypatch.setattr(search_entrypoint, "search_batch", fake_search_batch)

    search_entrypoint._search_with_progress(
        models=[],
        trees=[],
        database=np.array([[0.0]], dtype=np.float32),
        queries=np.array([[0.0]], dtype=np.float32),
        beam_size=1,
        num_return_leaves=1,
        progress_every=0,
        label="test",
        metric="euclidean",
        rerank_backend="torch_gpu",
        min_trees=1,
    )

    assert captured["rerank_backend"] == "torch_gpu"
    assert captured["min_trees"] == 1


def test_profiled_build_and_search_emit_stages_and_device_metadata(tmp_path: Path) -> None:
    """C9 acceptance 2 and 3: every applicable stage, plus reconciliation."""
    data_dir = tmp_path / "data"
    result_dir = tmp_path / "results"
    index_path = tmp_path / "index.pkl"
    config_path = tmp_path / "profiled.yaml"

    rng = np.random.default_rng(77)
    vectors = rng.normal(size=(128, 4)).astype(np.float32)
    queries = (vectors[:3] + rng.normal(scale=0.01, size=(3, 4))).astype(np.float32)
    distances = np.linalg.norm(queries[:, None, :] - vectors[None, :, :], axis=2)
    ground_truth = np.argsort(distances, axis=1).astype(np.int64)

    data_dir.mkdir()
    np.save(data_dir / "vectors.npy", vectors)
    np.save(data_dir / "queries.npy", queries)
    np.save(data_dir / "groundtruth.npy", ground_truth)

    config = {
        "experiment": {"name": "profiled", "seed": 77, "output_dir": str(result_dir)},
        "dataset": {
            "name": "synthetic",
            "path": str(data_dir),
            "split": "train",
            "subset_size": 128,
            "metric": "euclidean",
            "storage_mode": "preload",
        },
        "model": {
            "branching_factor": 4,
            "tree_height": 2,
            "embedding_dim": 4,
            "num_trees": 1,
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-5,
            "max_alternating_cycles": 1,
            "neighbor_search_subset": 128,
            "neighbor_search_backend": "faiss_cpu",
            "tree_update_cache_embeddings": True,
            "device": "cpu",
        },
        "evaluation": {
            "recall_at": [1],
            "num_queries": 3,
            "beam_size": 4,
            "num_leaves": [1],
            "rerank_backend": "numpy_cpu",
            "performance_profile": True,
            "search_repetitions": 3,
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    build_entrypoint = _load_entrypoint("build.py", "batl_profiled_build")
    search_entrypoint = _load_entrypoint("search.py", "batl_profiled_search")

    build_args = [
        str(config_path),
        "--index-path",
        str(index_path),
        "--result-dir",
        str(result_dir),
    ]
    build_entrypoint.main(build_args)

    build_metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    assert build_metrics["performance_profile"] is True
    build_stages = build_metrics["profile"]["stages"]
    for stage in (
        "build.label_mining",
        "build.train_epoch",
        "build.tree_update.encode_all",
        "build.tree_update.decode",
        "build.tree_update.assign",
        "build.tree_update.regroup",
    ):
        assert stage in build_stages, f"missing build stage: {stage}"
        assert build_stages[stage]["calls"] >= 1
    assert "peak_host_rss_bytes" in build_metrics["profile"]["memory"]

    search_entrypoint.main(
        [
            str(config_path),
            "--index-path",
            str(index_path),
            "--result-dir",
            str(result_dir),
            "--n-queries",
            "3",
            "--num-leaves",
            "1",
            "--batch-search",
            "0",
        ]
    )

    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["search_repetitions"] == 3
    profile = metrics["profile"]["num_leaves_1"]
    for stage in ("search.beam_decode", "search.leaf_lookup", "search.select_and_rerank_cpu"):
        assert stage in profile["stages"], f"missing search stage: {stage}"
    assert profile["timing"]["repetitions"] == 3
    assert len(profile["timing"]["times_s"]) == 3
    assert profile["timing"]["min_s"] <= profile["timing"]["median_s"]
    assert profile["timing"]["median_s"] <= profile["timing"]["max_s"]
    # Acceptance 3: stages account for most of the measured search wall-clock.
    assert profile["reconciliation"]["within_tolerance"] is True

    hardware = json.loads((result_dir / "hardware.json").read_text(encoding="utf-8"))
    assert hardware["device"] == "cpu"
    assert "torch_num_threads" in hardware
    assert "tf32_matmul" in hardware
    assert "cpu_count" in hardware
