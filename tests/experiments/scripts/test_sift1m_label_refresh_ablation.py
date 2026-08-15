import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import yaml

WRITER_PATH = Path("experiments/scripts/sift128/label_refresh_ablation/write_configs.py")
WRAPPER_PATH = Path(
    "experiments/scripts/sift128/label_refresh_ablation/"
    "metacentrum_sift1m_label_refresh_ablation.sh"
)
SOURCE_CONFIG = Path("experiments/configs/sift1m/sift1m_h2_paper.yaml")


def _load_writer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sift1m_label_refresh_configs", WRITER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _without_arm_identity(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    normalized["experiment"]["name"] = "<arm>"
    normalized["experiment"]["output_dir"] = "<arm>"
    normalized["training"]["label_refresh"] = "<arm>"
    return normalized


def test_config_writer_changes_only_the_label_refresh_policy_between_arms(tmp_path) -> None:
    writer = _load_writer()
    config_dir = tmp_path / "configs"
    output_root = tmp_path / "results"
    dataset_path = tmp_path / "sift-128-euclidean.hdf5"

    written = writer.write_arm_configs(
        source_config=SOURCE_CONFIG,
        dataset_path=dataset_path,
        output_root=output_root,
        config_dir=config_dir,
    )

    assert set(written) == {"per_cycle", "once"}
    configs: dict[str, dict[str, dict]] = {}
    for arm, paths in written.items():
        assert set(paths) == {"build", "search"}
        configs[arm] = {
            kind: yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            for kind, path in paths.items()
        }
        for kind, config in configs[arm].items():
            assert config["experiment"]["seed"] == 42
            assert config["model"]["num_trees"] == 1
            assert config["model"]["dropout"] == 0.1
            assert config["training"]["batch_size"] == 8192
            assert "max_alternating_cycles" not in config["training"]
            assert config["training"]["convergence_patience"] == 2
            assert config["training"]["convergence_min_delta"] == 0.005
            assert config["training"]["device"] == "cpu"
            assert config["training"]["neighbor_search_backend"] == "faiss_cpu"
            assert config["evaluation"]["rerank_backend"] == "numpy_cpu"
            assert config["evaluation"]["tree_assignment_mode"] == "round"
            assert config["evaluation"]["tree_assignment_order"] == "confidence"
            assert config["evaluation"]["num_leaves"] == [10, 20, 40, 80, 100]
            assert config["evaluation"]["performance_profile"] is (kind == "build")
            assert config["training"]["label_refresh"] == arm

    for kind in ("build", "search"):
        assert _without_arm_identity(configs["per_cycle"][kind]) == _without_arm_identity(
            configs["once"][kind]
        )


def test_wrapper_runs_both_arms_sequentially_on_one_cpu_node() -> None:
    text = WRAPPER_PATH.read_text(encoding="utf-8")

    assert "#PBS -l select=1:ncpus=6:mem=24gb:scratch_local=50gb" in text
    assert "ngpus" not in text
    assert "export OMP_NUM_THREADS=6" in text
    assert 'ARMS=("per_cycle" "once")' in text
    assert 'for ARM in "${ARMS[@]}"' in text
    assert text.count("build.py") == 1
    assert text.count("search.py") == 1
    assert text.index("build.py") < text.index("search.py")
    assert "--cycle-diagnostics" in text
    assert "--num-leaves 10 20 40 80 100" in text
    assert text.count("TIMESTAMP=$(date +%Y%m%d_%H%M)") == 1
    assert text.count("STORAGE_DIR=") == 1
    assert "write_configs.py" in text
