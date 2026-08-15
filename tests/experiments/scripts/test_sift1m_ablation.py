import subprocess
import sys
from pathlib import Path

import yaml

DIR = Path("experiments/scripts/sift128/ablation")
CONFIG_WRITER = DIR / "sift1m_ablation_config.sh"
COMMON = DIR / "run_sift1m_ablation_common.sh"
PAPER_CONFIG = Path("experiments/configs/sift1m/sift1m_h2_paper.yaml")
ARMS = {"dropout_01": "0.1", "dropout_00": "0.0"}


def _config_for(tmp_path: Path, dropout: str, batch_size: str = "8192") -> dict:
    """Run the real config writer and return what it produced."""
    out = tmp_path / f"arm_{dropout}.yaml"
    script = f"""
    set -eu
    PYTHON_EXEC={sys.executable}
    SRC_CONFIG={PAPER_CONFIG}
    DATA_PATH=/data/sift.hdf5
    RESULT_NAME=sift1m_h2_arm
    RESULT_DIR=/results/arm
    CONFIG_PATH={out}
    DROPOUT={dropout}
    BATCH_SIZE={batch_size}
    source {CONFIG_WRITER}
    write_ablation_config
    """
    subprocess.run(["bash", "-c", script], check=True, capture_output=True)
    return yaml.safe_load(out.read_text(encoding="utf-8"))


def test_every_arm_has_a_wrapper_that_sets_its_one_knob() -> None:
    for arm, dropout in ARMS.items():
        text = (DIR / f"metacentrum_sift1m_{arm}.sh").read_text(encoding="utf-8")
        assert f'ARM_NAME="{arm}"' in text
        assert f"DROPOUT={dropout}" in text
        assert "BATCH_SIZE=8192" in text
        assert "run_sift1m_ablation_common.sh" in text


def test_wrappers_do_not_write_their_own_config() -> None:
    """One writer, so two arms cannot drift into differing configs."""
    for arm in ARMS:
        text = (DIR / f"metacentrum_sift1m_{arm}.sh").read_text(encoding="utf-8")
        assert "yaml" not in text.lower()
    common = COMMON.read_text(encoding="utf-8")
    assert "sift1m_ablation_config.sh" in common
    assert "write_ablation_config" in common


def test_the_two_arms_differ_only_in_dropout(tmp_path: Path) -> None:
    """The claim the whole S5 result rests on.

    If anything else differs the run measures dropout plus that, and a null
    result would be uninterpretable rather than informative.
    """
    baseline = _config_for(tmp_path, "0.1")
    treatment = _config_for(tmp_path, "0.0")

    assert baseline["model"].pop("dropout") == 0.1
    assert treatment["model"].pop("dropout") == 0.0
    assert baseline == treatment


def test_every_arm_stops_on_the_same_convergence_rule(tmp_path: Path) -> None:
    """No arm carries a cycle cap; they all stop on the same patience.

    Arms may therefore end on different cycles. S4 showed cycle count does
    not move the curve on its own, so the comparison stays one-variable in
    practice, but read arm-to-arm differences against the logged cycles.
    """
    cfg = _config_for(tmp_path, "0.0")

    assert cfg["training"]["convergence_patience"] == 2
    assert cfg["training"]["convergence_min_delta"] == 0.005
    assert "max_alternating_cycles" not in cfg["training"]


def test_arms_inherit_untouched_values_from_the_paper_config(tmp_path: Path) -> None:
    """Only the knobs listed in the writer may diverge from the paper config."""
    paper = yaml.safe_load(PAPER_CONFIG.read_text(encoding="utf-8"))
    cfg = _config_for(tmp_path, "0.1")

    assert cfg["model"]["branching_factor"] == paper["model"]["branching_factor"]
    assert cfg["model"]["tree_height"] == paper["model"]["tree_height"]
    assert cfg["model"]["embed_dim"] == paper["model"]["embed_dim"]
    assert cfg["model"]["alpha"] == paper["model"]["alpha"]
    assert cfg["experiment"]["seed"] == paper["experiment"]["seed"]
    assert cfg["training"]["learning_rate"] == paper["training"]["learning_rate"]
    assert cfg["evaluation"]["beam_size"] == paper["evaluation"]["beam_size"]


def test_search_stays_within_beam_size(tmp_path: Path) -> None:
    """num_leaves above beam_size is rejected outright (code-review A3)."""
    cfg = _config_for(tmp_path, "0.1")
    common = COMMON.read_text(encoding="utf-8")

    swept = [int(v) for v in common.split("--num-leaves", 1)[1].split("\\", 1)[0].split()]
    assert swept == [10, 20, 40, 80, 100]
    assert max(swept) <= cfg["evaluation"]["beam_size"]


def test_run_is_cpu_only_end_to_end(tmp_path: Path) -> None:
    """The point of putting S5 on CPU is that it does not queue behind a GPU."""
    cfg = _config_for(tmp_path, "0.0")

    assert cfg["training"]["device"] == "cpu"
    assert cfg["training"]["neighbor_search_backend"] == "faiss_cpu"
    assert cfg["evaluation"]["rerank_backend"] == "numpy_cpu"
    for arm in ARMS:
        text = (DIR / f"metacentrum_sift1m_{arm}.sh").read_text(encoding="utf-8")
        assert "ngpus" not in text
