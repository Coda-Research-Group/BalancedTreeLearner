"""Contract tests for the runnable Deep100M selected-ablation jobs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from batl.utils.config_parsing import load_experiment_config

SCRIPT_DIR = Path("experiments/scripts/deep/100m/selected_ablation")
BASELINE = {
    "RESULT_NAME": "deep100m_selected_baseline",
    "NUM_TREES": "4",
    "BRANCHING_FACTOR": "256",
    "CONVERGENCE_PATIENCE": "2",
    "TOP_K_NEIGHBORS": "100",
    "NEIGHBOR_SEARCH_SUBSET": "1000000",
    "MIN_TREES": "2",
    "BEAM_SIZE": "100",
    "NUM_LEAVES": "100",
}


def _write_config(tmp_path: Path, **overrides: str):
    config_path = tmp_path / "config.yaml"
    env = (
        os.environ
        | BASELINE
        | overrides
        | {
            "CONFIG_PATH": str(config_path),
            "SCRATCHDIR": str(tmp_path),
        }
    )
    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; write_selected_config',
            "bash",
            str(SCRIPT_DIR / "write_selected_config.sh"),
        ],
        check=True,
        env=env,
    )
    return load_experiment_config(str(config_path))


def _dry_run(wrapper: str, index: int) -> dict[str, str]:
    env = os.environ | {"BATL_ARRAY_DRY_RUN": "1", "PBS_ARRAY_INDEX": str(index)}
    completed = subprocess.run(
        ["bash", str(SCRIPT_DIR / wrapper)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return dict(line.split("=", 1) for line in completed.stdout.splitlines())


def test_selected_config_is_strict_and_records_shared_baseline(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path)

    assert cfg.seed == 42
    assert cfg.model.num_trees == 4
    assert cfg.model.branching_factor == 256
    assert cfg.model.tree_height == 2
    assert cfg.train.batch_size == 16384
    assert cfg.train.alternating_interval == 2
    assert cfg.train.max_alternating_cycles is None
    assert cfg.train.convergence_patience == 2
    assert cfg.train.top_k_neighbors == 100
    assert cfg.train.neighbor_search_subset == 1_000_000
    assert cfg.train.label_refresh == "per_cycle"
    assert cfg.tree_assignment_mode == "round"
    assert cfg.tree_assignment_order == "confidence"
    assert cfg.min_trees == 2
    assert cfg.rerank_backend == "numpy_cpu"


@pytest.mark.parametrize("top_k", [10, 25, 50, 100])
def test_selected_config_allows_top_k_neighbor_arms(tmp_path: Path, top_k: int) -> None:
    cfg = _write_config(tmp_path, TOP_K_NEIGHBORS=str(top_k))
    assert cfg.train.top_k_neighbors == top_k


def test_selected_config_supports_single_tree_union_arm(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, NUM_TREES="1", MIN_TREES="1")
    assert cfg.model.num_trees == 1
    assert cfg.min_trees == 1


@pytest.mark.parametrize(
    ("overrides", "field", "expected"),
    [
        ({"BRANCHING_FACTOR": "64"}, "branching_factor", 64),
        ({"NEIGHBOR_SEARCH_SUBSET": "250000"}, "neighbor_search_subset", 250_000),
    ],
)
def test_selected_config_records_each_training_ablation(
    tmp_path: Path, overrides: dict[str, str], field: str, expected: int
) -> None:
    cfg = _write_config(tmp_path, **overrides)
    owner = cfg.model if field == "branching_factor" else cfg.train
    assert getattr(owner, field) == expected


@pytest.mark.parametrize("cycles", [1, 2, 5, 15])
def test_epochs_arm_pins_the_cycle_count_with_early_stopping_off(
    tmp_path: Path, cycles: int
) -> None:
    """The epochs arm ablates the cycle count, so it is the one arm that caps it."""
    cfg = _write_config(tmp_path, MAX_ALTERNATING_CYCLES=str(cycles), CONVERGENCE_PATIENCE="0")
    assert cfg.train.max_alternating_cycles == cycles
    assert cfg.train.convergence_patience == 0


def test_writer_rejects_a_config_with_no_stopping_condition(tmp_path: Path) -> None:
    """patience=0 and no cap would train forever; the writer must refuse it."""
    with pytest.raises(subprocess.CalledProcessError):
        _write_config(tmp_path, CONVERGENCE_PATIENCE="0")


@pytest.mark.parametrize(
    ("wrapper", "expected"),
    [
        (
            "metacentrum_deep100m_selected_baseline_build.sh",
            [("baseline", "0"), ("baseline", "1"), ("baseline", "2"), ("baseline", "3")],
        ),
        (
            "metacentrum_deep100m_selected_buckets_build.sh",
            [("buckets_k64", str(i)) for i in range(4)]
            + [("buckets_k128", str(i)) for i in range(4)],
        ),
        (
            "metacentrum_deep100m_selected_epochs_build.sh",
            [(f"epochs_{epochs}", str(i)) for epochs in (2, 4, 10, 30) for i in range(4)],
        ),
        (
            "metacentrum_deep100m_selected_topk_build.sh",
            [(f"topk_{top_k}", str(i)) for top_k in (10, 25, 50) for i in range(4)],
        ),
        (
            "metacentrum_deep100m_selected_sample_build.sh",
            [(f"sample_{sample}", str(i)) for sample in (250000, 500000) for i in range(4)],
        ),
    ],
)
def test_build_arrays_cover_every_nonbaseline_tree_once(
    wrapper: str, expected: list[tuple[str, str]]
) -> None:
    observed = []
    for index in range(len(expected)):
        row = _dry_run(wrapper, index)
        observed.append((row["ARM_NAME"], row["TREE_INDEX"]))
    assert observed == expected


@pytest.mark.parametrize(
    ("wrapper", "expected"),
    [
        (
            "metacentrum_deep100m_selected_repetitions_search.sh",
            [
                ("repetitions_t1", "1", "1"),
                ("repetitions_t2", "2", "1"),
                ("repetitions_t4", "4", "1"),
            ],
        ),
        (
            "metacentrum_deep100m_selected_buckets_search.sh",
            [("buckets_k64", "4", "2"), ("buckets_k128", "4", "2")],
        ),
        (
            "metacentrum_deep100m_selected_epochs_search.sh",
            [
                ("epochs_2", "4", "2"),
                ("epochs_4", "4", "2"),
                ("epochs_10", "4", "2"),
                ("epochs_30", "4", "2"),
            ],
        ),
        (
            "metacentrum_deep100m_selected_topk_search.sh",
            [("topk_10", "4", "2"), ("topk_25", "4", "2"), ("topk_50", "4", "2")],
        ),
        (
            "metacentrum_deep100m_selected_sample_search.sh",
            [("sample_250000", "4", "2"), ("sample_500000", "4", "2")],
        ),
    ],
)
def test_search_arrays_use_expected_tree_width_and_frequency_threshold(
    wrapper: str, expected: list[tuple[str, str, str]]
) -> None:
    observed = []
    for index in range(len(expected)):
        row = _dry_run(wrapper, index)
        observed.append((row["ARM_NAME"], row["NUM_TREES"], row["MIN_TREES"]))
    assert observed == expected


def test_common_search_refuses_partial_ensembles_and_sweeps_matched_points() -> None:
    text = (SCRIPT_DIR / "run_selected_search_common.sh").read_text(encoding="utf-8")

    assert "for ((TREE_INDEX = 0; TREE_INDEX < NUM_TREES; TREE_INDEX++))" in text
    assert 'if [ ! -f "$TREE_PATH" ]; then' in text
    assert "SEARCH_POINTS=(10 20 40 60 80 100 150 200 250 300)" in text
    assert 'BEAM_SIZE="$POINT"' in text
    assert 'NUM_LEAVES="$POINT"' in text
    assert 'MIN_TREES="$MIN_TREES"' in text
    assert 'cp -r "$POINT_DIR/." "$STORAGE_SEARCH_DIR/beam_${POINT}/"' in text


def test_submitter_links_each_search_to_its_build() -> None:
    text = (SCRIPT_DIR / "submit_selected_ablations.sh").read_text(encoding="utf-8")

    assert "BASELINE_BUILD=$(qsub metacentrum_deep100m_selected_baseline_build.sh)" in text
    for family in ("BUCKETS", "EPOCHS", "TOPK", "SAMPLE"):
        assert f"depend=afterok:$BASELINE_BUILD:${{{family}_BUILD}}" in text
    assert "metacentrum_deep100m_selected_baseline_search.sh" in text
    assert "metacentrum_deep100m_selected_repetitions_search.sh" in text
