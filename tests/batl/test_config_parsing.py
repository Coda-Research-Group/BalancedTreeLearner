"""Tests for batl.config_parsing."""

import pytest
import torch
import yaml

from batl.constants import DEFAULT_ASSIGNMENT_TOP_R
from batl.utils.config_parsing import (
    resolve_device,
    resolve_neighbor_search_backend,
    resolve_rerank_backend,
    should_preload_dataset,
)


def test_resolve_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("cuda") == "cpu"
    assert resolve_device("mps") == "cpu"
    assert resolve_device("cpu") == "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda") == "cuda"


def test_resolve_neighbor_search_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("batl.utils.config_parsing.is_faiss_gpu_available", lambda: False)
    assert resolve_neighbor_search_backend("auto") == "faiss_cpu"
    assert resolve_neighbor_search_backend("faiss_gpu") == "faiss_cpu"

    monkeypatch.setattr("batl.utils.config_parsing.is_faiss_gpu_available", lambda: True)
    assert resolve_neighbor_search_backend("auto") == "faiss_gpu"
    assert resolve_neighbor_search_backend("faiss_gpu") == "faiss_gpu"


def test_resolve_neighbor_search_backend_rejects_invalid_name() -> None:
    with pytest.raises(ValueError, match="neighbor_search_backend"):
        resolve_neighbor_search_backend("invalid")


def test_resolve_rerank_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_rerank_backend("auto", "cuda") == "numpy_cpu"
    assert resolve_rerank_backend("torch_gpu", "cuda") == "numpy_cpu"
    assert resolve_rerank_backend("torch_gpu_resident", "cuda") == "numpy_cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # auto prefers the resident reranker; whether the database actually fits in
    # VRAM is only checked at construction time (search.py).
    assert resolve_rerank_backend("auto", "cuda") == "torch_gpu_resident"
    assert resolve_rerank_backend("torch_gpu", "cuda") == "torch_gpu"
    assert resolve_rerank_backend("torch_gpu_resident", "cuda") == "torch_gpu_resident"
    assert resolve_rerank_backend("auto", "cpu") == "numpy_cpu"


def test_resolve_rerank_backend_rejects_invalid_name() -> None:
    with pytest.raises(ValueError, match="rerank_backend"):
        resolve_rerank_backend("invalid", "cpu")


def test_should_preload_dataset() -> None:
    assert should_preload_dataset(storage_mode="preload", estimated_nbytes=100) is True
    assert should_preload_dataset(storage_mode="memmap", estimated_nbytes=100) is False
    assert (
        should_preload_dataset(
            storage_mode="auto",
            estimated_nbytes=100,
            available_ram_bytes=1000,
            max_ram_fraction=0.5,
        )
        is True
    )
    assert (
        should_preload_dataset(
            storage_mode="auto",
            estimated_nbytes=1000,
            available_ram_bytes=100,
            max_ram_fraction=0.5,
        )
        is False
    )
    with pytest.raises(ValueError):
        should_preload_dataset(storage_mode="invalid", estimated_nbytes=100)


def test_tree_update_top_r_round_trips_and_rejects_degenerate_values(tmp_path) -> None:
    import yaml

    from batl.utils.config import TrainConfig
    from batl.utils.config_parsing import FinalConfigSanityChecker, load_experiment_config

    # Default is full K (None): truncation is opt-in until diagnostics justify it.
    assert TrainConfig().tree_update_top_r is None
    assert DEFAULT_ASSIGNMENT_TOP_R == 16  # suggested starting point only

    config = {
        "experiment": {"name": "topr", "seed": 1, "output_dir": str(tmp_path)},
        "dataset": {"name": "synthetic", "path": str(tmp_path), "split": "train"},
        "training": {"tree_update_top_r": 32},
        "evaluation": {"recall_at": [10], "num_queries": 4},
    }
    path = tmp_path / "topr.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    loaded = load_experiment_config(str(path))
    assert loaded.train.tree_update_top_r == 32

    loaded.train.tree_update_top_r = 1
    assert (
        "training.tree_update_top_r must be >= 2 or omitted for full K."
        in FinalConfigSanityChecker().check(loaded)
    )

    loaded.train.tree_update_top_r = None
    assert not [e for e in FinalConfigSanityChecker().check(loaded) if "top_r" in e]


def test_sequential_assignment_rejects_truncated_tree_update_top_r(tmp_path) -> None:
    from batl.utils.config_parsing import FinalConfigSanityChecker, load_experiment_config

    config = {
        "experiment": {"name": "sequential", "seed": 1, "output_dir": str(tmp_path)},
        "dataset": {"name": "synthetic", "path": str(tmp_path), "split": "train"},
        "model": {"branching_factor": 64},
        "training": {"tree_update_top_r": 16},
        "evaluation": {
            "recall_at": [10],
            "num_queries": 4,
            "tree_assignment_mode": "sequential",
            "tree_assignment_order": "input",
        },
    }
    path = tmp_path / "sequential_topr.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    loaded = load_experiment_config(str(path))

    assert any("full-K" in error for error in FinalConfigSanityChecker().check(loaded))


# --- A2: unknown YAML keys must fail, not be silently dropped ---


def _write_config(tmp_path, **section_overrides):
    import yaml

    raw = {
        "experiment": {"name": "t", "seed": 0, "output_dir": "out"},
        "dataset": {"name": "d", "path": "p", "split": "train"},
        "evaluation": {"recall_at": [10], "num_queries": 10, "beam_size": 100},
    }
    for section, extra in section_overrides.items():
        raw[section].update(extra)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return str(path)


def test_typo_in_evaluation_is_rejected_with_the_section_and_a_suggestion(tmp_path) -> None:
    from batl.utils.config_parsing import load_experiment_config

    path = _write_config(tmp_path, evaluation={"num_leavse": [100]})

    with pytest.raises(ValueError) as excinfo:
        load_experiment_config(path, strict=False)

    message = str(excinfo.value)
    assert "evaluation.num_leavse" in message
    # The whole point is catching typos, so it should name the intended key.
    assert "did you mean 'num_leaves'" in message

    # The same config with the key spelled correctly loads.
    good = _write_config(tmp_path, evaluation={"num_leaves": [100]})
    assert load_experiment_config(good, strict=False).num_leaves == [100]


def test_unknown_keys_are_reported_for_every_section(tmp_path) -> None:
    from batl.utils.config_parsing import load_experiment_config

    path = _write_config(
        tmp_path,
        experiment={"sed": 1},
        dataset={"pth": "x"},
        evaluation={"beam_sizes": 10},
    )

    with pytest.raises(ValueError) as excinfo:
        load_experiment_config(path, strict=False)

    message = str(excinfo.value)
    assert "experiment.sed" in message
    assert "dataset.pth" in message
    assert "evaluation.beam_sizes" in message


def test_dataset_provenance_keys_are_accepted(tmp_path) -> None:
    """source_name/source_url are documented metadata, not typos."""
    from batl.utils.config_parsing import load_experiment_config

    path = _write_config(tmp_path, dataset={"source_name": "Deep1B", "source_url": "yandex"})

    assert load_experiment_config(path, strict=False).dataset_name == "d"


def test_accepted_keys_are_derived_from_the_dataclass() -> None:
    """A new ExperimentConfig field must not need a second edit here."""
    from dataclasses import fields

    from batl.utils.config import ExperimentConfig
    from batl.utils.config_parsing import _accepted_config_keys

    accepted = _accepted_config_keys()
    names = {f.name for f in fields(ExperimentConfig)} - {"model", "train"}

    assert accepted["experiment"] == names
    assert accepted["evaluation"] == names
    assert "storage_mode" in accepted["dataset"]
    assert "subset_size" in accepted["dataset"]
    assert "dataset_storage_mode" not in accepted["dataset"]


# --- A1: the sanity checker actually runs at the entry points ---


def test_run_final_config_sanity_checks_raises_with_every_error_joined() -> None:
    from batl.utils.config import ExperimentConfig
    from batl.utils.config_parsing import FinalConfigSanityChecker, run_final_config_sanity_checks

    class TwoProblems(FinalConfigSanityChecker):
        def check(self, cfg: ExperimentConfig) -> list[str]:
            return ["first problem.", "second problem."]

    cfg = _experiment_config()

    with pytest.raises(ValueError) as excinfo:
        run_final_config_sanity_checks(cfg, checker=TwoProblems())

    message = str(excinfo.value)
    assert "config failed sanity checks" in message
    assert "- first problem." in message
    assert "- second problem." in message


def test_run_final_config_sanity_checks_skip_warns_and_returns(caplog) -> None:
    from batl.utils.config import ExperimentConfig
    from batl.utils.config_parsing import FinalConfigSanityChecker, run_final_config_sanity_checks

    class AlwaysFails(FinalConfigSanityChecker):
        def check(self, cfg: ExperimentConfig) -> list[str]:
            return ["would have failed."]

    with caplog.at_level("WARNING", logger="batl.utils.config_parsing"):
        run_final_config_sanity_checks(_experiment_config(), skip=True, checker=AlwaysFails())

    assert "skipped by --skip-sanity-checks" in caplog.text


def test_run_final_config_sanity_checks_logs_advisories_without_failing(caplog) -> None:
    from batl.constants import LARGE_DATASET_RANDOM_SUBSET_THRESHOLD
    from batl.utils.config_parsing import run_final_config_sanity_checks

    cfg = _experiment_config()
    cfg.subset_size = LARGE_DATASET_RANDOM_SUBSET_THRESHOLD
    cfg.train.neighbor_search_mode = "random_subset"

    with caplog.at_level("WARNING", logger="batl.utils.config_parsing"):
        run_final_config_sanity_checks(cfg)

    assert "config advisory" in caplog.text
    assert "random_subset" in caplog.text


def _experiment_config():
    from batl.utils.config import ExperimentConfig

    return ExperimentConfig(
        name="t",
        seed=0,
        output_dir="out",
        dataset_name="d",
        dataset_path="p",
        split="train",
        subset_size=None,
        recall_at=[10],
        num_queries=10,
        beam_size=100,
    )
