import re
import subprocess
import sys
from pathlib import Path

import yaml

DIR = Path("experiments/scripts/sift128/batch_ablation")
TREE_COMMON = DIR / "run_sift1m_batch_tree_common.sh"
MERGE = DIR / "metacentrum_sift1m_batch256_merge_search.sh"
CONFIG_WRITER = Path("experiments/scripts/sift128/ablation/sift1m_ablation_config.sh")
PAPER_CONFIG = Path("experiments/configs/sift1m/sift1m_h2_paper.yaml")
TREES = (0, 1, 2, 3)


def _config_for(tmp_path: Path, batch_size: str) -> dict:
    out = tmp_path / f"arm_{batch_size}.yaml"
    script = f"""
    set -eu
    PYTHON_EXEC={sys.executable}
    SRC_CONFIG={PAPER_CONFIG}
    DATA_PATH=/data/sift.hdf5
    RESULT_NAME=sift1m_h2_arm
    RESULT_DIR=/results/arm
    CONFIG_PATH={out}
    DROPOUT=0.0
    BATCH_SIZE={batch_size}
    source {CONFIG_WRITER}
    write_ablation_config
    """
    subprocess.run(["bash", "-c", script], check=True, capture_output=True)
    return yaml.safe_load(out.read_text(encoding="utf-8"))


def test_one_wrapper_per_tree_with_its_own_index() -> None:
    for tree in TREES:
        text = (DIR / f"metacentrum_sift1m_batch256_tree_{tree}.sh").read_text(encoding="utf-8")
        assert f"TREE_INDEX={tree}" in text
        assert 'ARM_NAME="batch256"' in text
        assert "BATCH_SIZE=256" in text
        assert "run_sift1m_batch_tree_common.sh" in text
        # Splitting exists so a walltime kill costs one tree, not the run.
        assert "--tree-index" not in text, "the tree index must reach build.py via the common body"
    assert "--tree-index" in TREE_COMMON.read_text(encoding="utf-8")


def test_arm_differs_from_the_completed_8192_arm_in_batch_size_alone(tmp_path: Path) -> None:
    """The batch-8192 baseline is the finished dropout_00 run, not a new job.

    That only works if the two configs are otherwise identical, which is why
    this arm reuses the dropout ablation's writer and pins DROPOUT=0.0.
    """
    baseline = _config_for(tmp_path, "8192")
    treatment = _config_for(tmp_path, "256")

    assert baseline["training"].pop("batch_size") == 8192
    assert treatment["training"].pop("batch_size") == 256
    assert baseline == treatment


def test_tree_arm_pins_dropout_to_match_that_baseline() -> None:
    """A different dropout here would make the comparison two-variable."""
    assert "DROPOUT=0.0" in TREE_COMMON.read_text(encoding="utf-8")
    assert "DROPOUT=0.0" in MERGE.read_text(encoding="utf-8")


def test_merge_refuses_a_partial_ensemble() -> None:
    """Three trees would silently change the >=2-of-4 filter.

    The resulting numbers would look valid and be comparable to nothing.
    """
    text = MERGE.read_text(encoding="utf-8")

    assert "Missing tree index" in text
    assert "exit 2" in text
    merged = re.findall(r"index_confidence_tree_(\d)\.pkl", text)
    assert sorted(set(merged)) == ["0", "1", "2", "3"]


def test_per_tree_filenames_match_what_the_merge_reads() -> None:
    """build.py --tree-index N writes <stem>_tree_N.pkl; the merge must agree."""
    common = TREE_COMMON.read_text(encoding="utf-8")
    merge = MERGE.read_text(encoding="utf-8")

    assert "index_confidence_tree_${TREE_INDEX}.pkl" in common
    assert "index_confidence_tree_0.pkl" in merge


def test_tree_jobs_get_more_walltime_than_the_job_that_died() -> None:
    """Job 22764327 was killed at 12h during tree 3 of 4 at batch 8192.

    Batch 256 runs 31.8x more optimizer steps per epoch, so per-tree jobs need
    headroom even though each does a quarter of the work.
    """
    for tree in TREES:
        text = (DIR / f"metacentrum_sift1m_batch256_tree_{tree}.sh").read_text(encoding="utf-8")
        hours = int(re.search(r"walltime=(\d+):", text).group(1))
        assert hours >= 20
    assert "ngpus" not in TREE_COMMON.read_text(encoding="utf-8")
