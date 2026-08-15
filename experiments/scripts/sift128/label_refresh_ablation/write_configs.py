"""Generate the paired configs for the SIFT1M label-refresh ablation."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

ARMS = ("per_cycle", "once")
CONFIG_KINDS = ("build", "search")


def _arm_config(
    source: dict[str, Any],
    *,
    arm: str,
    kind: str,
    dataset_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Return one fully resolved config for an arm and execution phase."""
    if arm not in ARMS:
        raise ValueError(f"unknown label-refresh arm: {arm}")
    if kind not in CONFIG_KINDS:
        raise ValueError(f"unknown config kind: {kind}")

    config = copy.deepcopy(source)
    config["experiment"]["name"] = f"sift1m_label_refresh_{arm}_{kind}"
    config["experiment"]["output_dir"] = str(output_root / arm / kind)
    config["dataset"]["path"] = str(dataset_path)
    config["dataset"]["metric"] = "euclidean"
    config["dataset"]["storage_mode"] = "preload"

    config["model"]["num_trees"] = 1

    training = config["training"]
    training["batch_size"] = 8192
    training["convergence_patience"] = 2
    training["convergence_min_delta"] = 0.005
    training["device"] = "cpu"
    training["neighbor_search_backend"] = "faiss_cpu"
    training["tree_update_cache_embeddings"] = False
    training["label_refresh"] = arm

    evaluation = config["evaluation"]
    evaluation["num_queries"] = 10_000
    evaluation["beam_size"] = 100
    evaluation["num_leaves"] = [10, 20, 40, 80, 100]
    evaluation["tree_assignment_mode"] = "round"
    evaluation["tree_assignment_order"] = "confidence"
    evaluation["rerank_backend"] = "numpy_cpu"
    evaluation["performance_profile"] = kind == "build"
    return config


def write_arm_configs(
    *,
    source_config: str | Path,
    dataset_path: str | Path,
    output_root: str | Path,
    config_dir: str | Path,
) -> dict[str, dict[str, Path]]:
    """Write build/search configs for both policies and return their paths."""
    source = yaml.safe_load(Path(source_config).read_text(encoding="utf-8"))
    data = Path(dataset_path)
    results = Path(output_root)
    destination = Path(config_dir)
    destination.mkdir(parents=True, exist_ok=True)

    written: dict[str, dict[str, Path]] = {}
    for arm in ARMS:
        written[arm] = {}
        for kind in CONFIG_KINDS:
            path = destination / f"{arm}_{kind}.yaml"
            config = _arm_config(
                source,
                arm=arm,
                kind=kind,
                dataset_path=data,
                output_root=results,
            )
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            written[arm][kind] = path
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    written = write_arm_configs(
        source_config=args.source_config,
        dataset_path=args.dataset_path,
        output_root=args.output_root,
        config_dir=args.config_dir,
    )
    for arm, paths in written.items():
        print(f"{arm}: build={paths['build']} search={paths['search']}")


if __name__ == "__main__":
    main()
