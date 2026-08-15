"""Persistence helpers for BATL indexes and run artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from batl.distance import l2_normalize
from batl.tree import BATLTree
from batl.utils.config import ExperimentConfig
from batl.utils.config_parsing import should_preload_dataset
from batl.utils.data import load_queries_and_gt, load_vectors
from batl.utils.index_parsing import to_run_plan_dataset_dict


def jsonable(value: Any) -> Any:
    """Convert common NumPy containers/scalars to JSON-serializable values."""
    match value:
        case dict():
            return {k: jsonable(v) for k, v in value.items()}
        case list() | tuple():
            return [jsonable(v) for v in value]
        case np.ndarray():
            return jsonable(value.tolist())
        case np.floating():
            return float(value)
        case np.integer():
            return int(value)
        case _:
            return value


def write_rows(output_dir: str | Path, stem: str, rows: list[dict[str, Any]]) -> None:
    """Write benchmark rows to ``<stem>.json`` and ``<stem>.csv``."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / f"{stem}.json"
    csv_path = output / f"{stem}.csv"
    json_path.write_text(
        json.dumps([jsonable(row) for row in rows], indent=2) + "\n",
        encoding="utf-8",
    )
    if not rows:
        return
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(cast(Mapping[str, Any], jsonable(row)))


def save_benchmark_artifacts(
    *,
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    run_plan: Mapping[str, Any],
    cfg: ExperimentConfig,
    seed: int,
    run_metadata: Mapping[str, Any],
    config_report: Mapping[str, Any] | None = None,
    extra_metrics: Mapping[str, Any] | None = None,
) -> None:
    """Write benchmark artifacts without writing row CSV/JSON files."""
    dataset_source = to_run_plan_dataset_dict(
        cfg.dataset_name,
        cfg.dataset_path,
        cfg.dataset_base_path,
        cfg.dataset_query_path,
        cfg.dataset_ground_truth_path,
        cfg.split,
        cfg.subset_size,
        cfg.dataset_metric,
        cfg.dataset_normalize,
        cfg.dataset_storage_mode,
        cfg.rerank_backend,
    )
    dataset_source["ann_benchmarks_layout"] = "train/test/neighbors"

    metrics: dict[str, Any] = {
        "rows": [jsonable(row) for row in rows],
        "run_plan": jsonable(run_plan),
        "dataset_source": dataset_source,
    }
    if config_report is not None:
        metrics["config_report"] = jsonable(config_report)
    if extra_metrics is not None:
        metrics.update(cast(dict[str, Any], jsonable(dict(extra_metrics))))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    _write_json(output / "metrics.json", metrics)
    with (output / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
    (output / "seed.txt").write_text(f"{seed}\n", encoding="utf-8")
    _write_json(output / "environment.json", run_metadata["environment"])
    _write_json(output / "hardware.json", run_metadata["hardware"])


def _tree_payload(tree: BATLTree) -> dict:
    return {
        "K": tree.K,
        "H": tree.H,
        "alpha": tree.alpha,
        "N": tree.N,
        "paths": tree.paths,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_run_database(
    cfg: ExperimentConfig,
    copy_array: bool = False,
) -> np.ndarray:
    """Load database vectors for a BATL build without requiring query artifacts."""
    database = load_vectors(
        cfg.dataset_name,
        cfg.dataset_path,
        cfg.split,
        cfg.subset_size,
        base_path=cfg.dataset_base_path,
    )

    if cfg.dataset_normalize:
        database = l2_normalize(database)

    if should_preload_dataset(
        storage_mode=cfg.dataset_storage_mode,
        estimated_nbytes=int(database.size * database.dtype.itemsize),
    ):
        database = np.array(database, dtype=np.float32, copy=True)

    if copy_array:
        database = np.array(database, copy=True)

    return database


def load_run_data(
    cfg: ExperimentConfig,
    max_queries: int | None = None,
    copy_arrays: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load database vectors, queries, and ground truth for a BATL run."""
    if max_queries is not None and max_queries <= 0:
        raise ValueError("max_queries must be positive when provided.")

    database = load_run_database(cfg)
    queries, ground_truth = load_queries_and_gt(
        cfg.dataset_name,
        cfg.dataset_path,
        cfg.num_queries,
        query_path=cfg.dataset_query_path,
        ground_truth_path=cfg.dataset_ground_truth_path,
    )
    if max_queries is not None:
        queries = queries[:max_queries]
        ground_truth = ground_truth[:max_queries]

    if cfg.dataset_normalize:
        queries = l2_normalize(queries)

    if copy_arrays:
        database = np.array(database, copy=True)
        queries = np.array(queries, copy=True)
        ground_truth = np.array(ground_truth, copy=True)

    return database, queries, ground_truth
