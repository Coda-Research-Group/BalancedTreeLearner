"""Tests for BATL experiment configuration correctness."""

from pathlib import Path

from batl.utils.config_parsing import load_experiment_config


def test_all_committed_configs_parse_in_smoke_mode() -> None:
    for path in sorted(Path("experiments/configs").rglob("*.yaml")):
        load_experiment_config(str(path), strict=False)


def test_small_scale_paper_configs_match_expected_tree_shapes() -> None:
    expected = {
        "sift1m_h2_paper.yaml": ("sift1m_h2_paper", 64, 2),
        "sift1m_h3_paper.yaml": ("sift1m_h3_paper", 16, 3),
        "sift1m_h4_paper.yaml": ("sift1m_h4_paper", 8, 4),
        "glove100_h2_paper.yaml": ("glove100_h2_paper", 64, 2),
        "glove100_h3_paper.yaml": ("glove100_h3_paper", 16, 3),
        "glove100_h4_paper.yaml": ("glove100_h4_paper", 8, 4),
    }

    for path in sorted(Path("experiments/configs").glob("*/*_paper.yaml")):
        if path.name not in expected:
            continue
        expected_name, expected_k, expected_h = expected[path.name]
        config = load_experiment_config(str(path), strict=False)

        assert config.name == expected_name
        assert config.model.branching_factor == expected_k
        assert config.model.tree_height == expected_h
        assert config.model.branching_factor**config.model.tree_height == 4096
        assert config.model.num_trees == 4
        assert config.beam_size == 100


def test_large_memmap_configs_use_sequential_chunked_label_search() -> None:
    paths = [
        Path("experiments/configs/laion5b/laion5b_100m_h2.yaml"),
        Path("experiments/configs/laion5b/laion5b_10m_h2.yaml"),
    ]

    for path in paths:
        config = load_experiment_config(str(path), strict=False)

        assert config.dataset_storage_mode == "memmap"
        assert config.subset_size is not None and config.subset_size >= 10_000_000
        assert config.train.neighbor_search_mode == "sequential_chunked"
