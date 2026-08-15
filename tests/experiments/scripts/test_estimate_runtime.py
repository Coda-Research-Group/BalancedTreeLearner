"""Smoke test for the 1-cycle runtime probe on synthetic data."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import yaml


def _write_synthetic_cfg(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(512, 8)).astype(np.float32)
    queries = (vectors[:16] + rng.normal(scale=0.01, size=(16, 8))).astype(np.float32)
    dists = np.linalg.norm(queries[:, None, :] - vectors[None, :, :], axis=2)
    gt = np.argsort(dists, axis=1).astype(np.int64)
    np.save(data_dir / "vectors.npy", vectors)
    np.save(data_dir / "queries.npy", queries)
    np.save(data_dir / "groundtruth.npy", gt)

    cfg = {
        "experiment": {"name": "probe_smoke", "seed": 1, "output_dir": str(tmp_path / "out")},
        "dataset": {
            "name": "synthetic",
            "path": str(data_dir),
            "split": "train",
            "subset_size": 512,
            "metric": "euclidean",
            "storage_mode": "preload",
        },
        "model": {"branching_factor": 4, "tree_height": 2, "embedding_dim": 8, "num_trees": 4},
        "training": {
            "batch_size": 64,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-5,
            "max_alternating_cycles": 8,
            "neighbor_search_subset": 512,
            "neighbor_search_backend": "faiss_cpu",
            "tree_update_cache_embeddings": False,
            "device": "cpu",
        },
        "evaluation": {
            "recall_at": [1],
            "num_queries": 16,
            "beam_size": 4,
            "num_leaves": [1],
            "rerank_backend": "numpy_cpu",
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


def _runtime_probe_env() -> dict[str, str]:
    passthrough = ("HOME", "PATH", "TMPDIR", "LANG", "LC_ALL", "CONDA_PREFIX", "CONDA_DEFAULT_ENV")
    env = {key: os.environ[key] for key in passthrough if key in os.environ}
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "KMP_INIT_AT_FORK": "FALSE",
            "KMP_AFFINITY": "disabled",
            "PYTHONPATH": str(Path(".").resolve()),
        }
    )
    return env


def _load_estimate_runtime() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "estimate_runtime_test_entrypoint",
        "experiments/scripts/estimate_runtime.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_probe_in_process(
    cfg_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> str:
    module = _load_estimate_runtime()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "estimate_runtime.py",
            str(cfg_path),
            "--device",
            "cpu",
            "--queries",
            "8",
            "--num-leaves",
            "1",
            "--cycles",
            "5",
        ],
    )
    module.main()
    return capsys.readouterr().out


def test_estimate_runtime_one_cycle_probe_reports_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_path = _write_synthetic_cfg(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "experiments/scripts/estimate_runtime.py",
            str(cfg_path),
            "--device",
            "cpu",
            "--queries",
            "8",
            "--num-leaves",
            "1",
            "--cycles",
            "5",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_runtime_probe_env(),
    )
    if proc.returncode == 0:
        out = proc.stdout
    elif "OMP: Error #179" in proc.stderr and "Can't open SHM" in proc.stderr:
        out = _run_probe_in_process(cfg_path, monkeypatch, capsys)
    else:
        assert proc.returncode == 0, proc.stderr
        out = proc.stdout
    assert "summary" in out
    assert "host:" in out
    assert "RSS" in out and "GPU" in out
    assert "build_w" in out and "search_w" in out
    assert cfg_path.name in out
    assert re.search(r"\d+MB\s+\d+MB", out)
