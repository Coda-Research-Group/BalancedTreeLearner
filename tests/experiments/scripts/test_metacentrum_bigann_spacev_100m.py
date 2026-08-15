from pathlib import Path

import yaml

SPACEV_DIR = Path("experiments/scripts/spacev")
BIGANN_DIR = Path("experiments/scripts/bigann")
CONFIGS_DIR = Path("experiments/configs")


def test_spacev100m_config_matches_expected_tree_shape() -> None:
    config = yaml.safe_load((CONFIGS_DIR / "spacev100m_full_t4.yaml").read_text(encoding="utf-8"))

    assert config["dataset"]["name"] == "spacev"
    assert config["dataset"]["source_name"] == "SPACEV100M"
    assert config["dataset"]["subset_size"] == 100000000
    assert config["dataset"]["storage_mode"] == "memmap"
    assert config["model"]["embedding_dim"] == 100
    assert config["model"]["branching_factor"] == 256
    assert config["model"]["tree_height"] == 2
    assert config["model"]["num_trees"] == 4
    assert config["evaluation"]["num_queries"] == 29316
    assert config["evaluation"]["rerank_backend"] == "numpy_cpu"


def test_spacev100m_parallel_tree_scripts_build_four_seeded_trees() -> None:
    common = (SPACEV_DIR / "run_spacev_100m_tree_common.sh").read_text(encoding="utf-8")

    assert "base.100M.i8bin" in common
    assert "query.30K.i8bin" in common
    assert "groundtruth.30K.i32bin" in common
    assert "experiments/configs/spacev100m_full_t4.yaml" in common
    assert 'cfg["dataset"]["base_path"] = f"{data_dir}/base.i8bin"' in common
    assert 'cfg["dataset"]["query_path"] = f"{data_dir}/query.i8bin"' in common
    assert 'cfg["dataset"]["ground_truth_path"] = f"{data_dir}/groundtruth.i32bin"' in common
    assert "module load mambaforge" in common
    assert "LD_LIBRARY_PATH" in common
    assert "import faiss" in common
    assert '--tree-index "$TREE_INDEX"' in common
    assert "search.py" not in common

    for idx in range(4):
        text = (SPACEV_DIR / f"metacentrum_spacev_100m_tree_{idx}.sh").read_text(encoding="utf-8")
        assert f"#PBS -N batl_spacev100m_tree_{idx}" in text
        assert f"TREE_INDEX={idx}" in text
        assert "run_spacev_100m_tree_common.sh" in text


def test_spacev100m_merge_search_merges_four_trees_and_searches() -> None:
    text = (SPACEV_DIR / "metacentrum_spacev_100m_merge_search.sh").read_text(encoding="utf-8")

    assert "#PBS -N batl_spacev100m_merge_search" in text
    assert "merge_index.py" in text
    for idx in range(4):
        assert f"index_confidence_tree_{idx}.pkl" in text
    assert "search.py" in text
    assert "experiments/configs/spacev100m_full_t4.yaml" in text
    assert 'cfg["evaluation"]["beam_size"] = 300' in text
    assert 'cfg["evaluation"]["num_leaves"] = [100, 150, 200, 250, 300]' in text
    assert "--num-leaves 100 150 200 250 300" in text
    assert "--n-queries 29316" in text
    assert "--batch-search 25" in text
    assert "LD_LIBRARY_PATH" in text
    assert "import faiss" in text


def test_bigann100m_config_matches_expected_tree_shape() -> None:
    config = yaml.safe_load((CONFIGS_DIR / "bigann100m_full_t4.yaml").read_text(encoding="utf-8"))

    assert config["dataset"]["name"] == "bigann"
    assert config["dataset"]["source_name"] == "BIGANN100M"
    assert config["dataset"]["subset_size"] == 100000000
    assert config["dataset"]["storage_mode"] == "memmap"
    assert config["model"]["embedding_dim"] == 128
    assert config["model"]["branching_factor"] == 256
    assert config["model"]["tree_height"] == 2
    assert config["model"]["num_trees"] == 4
    assert config["evaluation"]["num_queries"] == 10000
    assert config["evaluation"]["rerank_backend"] == "numpy_cpu"


def test_bigann100m_parallel_tree_scripts_build_four_seeded_trees() -> None:
    common = (BIGANN_DIR / "run_bigann_100m_tree_common.sh").read_text(encoding="utf-8")

    assert "base.100M.u8bin" in common
    assert "learn.100M.u8bin" in common
    assert "Do not substitute learn.100M.u8bin" in common
    assert "query.public.10K.u8bin" in common
    assert "idx_100M.ivecs" in common
    assert "experiments/configs/bigann100m_full_t4.yaml" in common
    assert 'cfg["dataset"]["base_path"] = f"{data_dir}/base.u8bin"' in common
    assert 'cfg["dataset"]["query_path"] = f"{data_dir}/query.u8bin"' in common
    assert 'cfg["dataset"]["ground_truth_path"] = f"{data_dir}/groundtruth.ivecs"' in common
    assert "module load mambaforge" in common
    assert "LD_LIBRARY_PATH" in common
    assert "import faiss" in common
    assert "index_confidence_tree_${TREE_INDEX}.sources" in common
    assert '--tree-index "$TREE_INDEX"' in common
    assert "search.py" not in common

    for idx in range(4):
        text = (BIGANN_DIR / f"metacentrum_bigann_100m_tree_{idx}.sh").read_text(encoding="utf-8")
        assert f"#PBS -N batl_bigann100m_tree_{idx}" in text
        assert f"TREE_INDEX={idx}" in text
        assert "run_bigann_100m_tree_common.sh" in text


def test_bigann100m_merge_search_merges_four_trees_and_searches() -> None:
    text = (BIGANN_DIR / "metacentrum_bigann_100m_merge_search.sh").read_text(encoding="utf-8")

    assert "#PBS -N batl_bigann100m_merge_search" in text
    assert "merge_index.py" in text
    for idx in range(4):
        assert f"index_confidence_tree_{idx}.pkl" in text
    assert "search.py" in text
    assert "experiments/configs/bigann100m_full_t4.yaml" in text
    assert 'cfg["evaluation"]["beam_size"] = 300' in text
    assert 'cfg["evaluation"]["num_leaves"] = [100, 150, 200, 250, 300]' in text
    assert "--num-leaves 100 150 200 250 300" in text
    assert "query.public.10K.u8bin" in text
    assert "idx_100M.ivecs" in text
    assert "Do not substitute learn.100M.u8bin" in text
    assert "--n-queries 10000" in text
    assert "--batch-search 25" in text
    assert "LD_LIBRARY_PATH" in text
    assert "import faiss" in text
    assert ".sources" in text
    assert "Rebuild the BIGANN trees with the current scripts" in text
