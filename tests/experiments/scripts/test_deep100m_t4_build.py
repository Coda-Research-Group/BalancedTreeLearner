import re
import subprocess
from pathlib import Path

import yaml

DIR = Path("experiments/scripts/deep/100m/t4_build")
TREE_COMMON = DIR / "run_deep100m_t4_tree_common.sh"
MERGE = DIR / "metacentrum_deep100m_t4_merge_search.sh"
CONFIG_WRITER = Path("experiments/scripts/deep/100m/label_ablation/deep100m_label_config.sh")
TREES = (0, 1, 2, 3)


def _config(tmp_path: Path, **env: str) -> dict:
    out = tmp_path / f"cfg_{'_'.join(env.values()) or 'default'}.yaml"
    assignments = "\n    ".join(f"{k}={v}" for k, v in env.items())
    script = f"""
    set -eu
    SCRATCHDIR=/scratch
    RESULT_NAME=deep100m_cfg
    CONFIG_PATH={out}
    NEIGHBOR_SEARCH_SUBSET=1000000
    {assignments}
    source {CONFIG_WRITER}
    write_label_config
    """
    subprocess.run(["bash", "-c", script], check=True, capture_output=True)
    return yaml.safe_load(out.read_text(encoding="utf-8"))


def test_one_wrapper_per_tree_delegating_the_index_to_the_common_body() -> None:
    for tree in TREES:
        text = (DIR / f"metacentrum_deep100m_t4_tree_{tree}.sh").read_text(encoding="utf-8")
        assert f"TREE_INDEX={tree}" in text
        assert "run_deep100m_t4_tree_common.sh" in text
        assert "--tree-index" not in text
    assert '--tree-index "$TREE_INDEX"' in TREE_COMMON.read_text(encoding="utf-8")


def test_build_and_merge_agree_on_four_trees_and_the_label_subset(tmp_path: Path) -> None:
    """A T=4 index searched under a T=1 config would skip the frequency filter."""
    for text in (TREE_COMMON.read_text(encoding="utf-8"), MERGE.read_text(encoding="utf-8")):
        assert "NUM_TREES=4" in text
        assert "NEIGHBOR_SEARCH_SUBSET=1000000" in text

    cfg = _config(tmp_path, NUM_TREES="4")
    assert cfg["model"]["num_trees"] == 4
    assert cfg["training"]["neighbor_search_subset"] == 1_000_000


def test_ablation_arms_are_unchanged_by_the_num_trees_knob(tmp_path: Path) -> None:
    """The label arms must keep producing byte-identical configs."""
    default = _config(tmp_path)
    t4 = _config(tmp_path, NUM_TREES="4")

    assert default["model"].pop("num_trees") == 1
    assert t4["model"].pop("num_trees") == 4
    assert default == t4


def test_config_writer_heredoc_has_no_command_substitution() -> None:
    """The heredoc is unquoted, so a backtick in a comment would execute.

    One slipped in while documenting the num_trees knob and printed
    "command not found" into the build log.
    """
    body = CONFIG_WRITER.read_text(encoding="utf-8").split('cat > "$CONFIG_PATH" <<EOF', 1)[1]
    body = body.split("\nEOF", 1)[0]

    assert "`" not in body
    assert "$(" not in body


def test_merge_refuses_a_partial_ensemble() -> None:
    text = MERGE.read_text(encoding="utf-8")

    assert "Missing tree index" in text
    assert "exit 2" in text
    merged = re.findall(r"index_confidence_tree_(\d)\.pkl", text)
    assert sorted(set(merged)) == ["0", "1", "2", "3"]
    assert "build.py" not in text


def test_tree_jobs_are_sized_for_a_contended_node() -> None:
    """The 1% arm built one tree in 3.7h; a contended node cost 6x elsewhere."""
    for tree in TREES:
        text = (DIR / f"metacentrum_deep100m_t4_tree_{tree}.sh").read_text(encoding="utf-8")
        assert int(re.search(r"walltime=(\d+):", text).group(1)) >= 16
        # Only the resident-rerank control needs a large card.
        assert "gpu_mem=16gb" in text


def test_search_uses_the_same_rerank_backend_as_every_other_deep100m_sweep(
    tmp_path: Path,
) -> None:
    """Otherwise this curve is not comparable to the arms or the July baseline."""
    cfg = _config(tmp_path, NUM_TREES="4")

    assert cfg["evaluation"]["rerank_backend"] == "numpy_cpu"
    assert "RERANK_BACKEND" not in MERGE.read_text(encoding="utf-8")


def test_search_sweeps_beam_so_it_can_reach_the_paper_operating_point() -> None:
    """A fixed beam caps reachable recall, because A3 forbids M > beam_size.

    Job 22821720 swept M at beam 100 and stopped at Recall@10 0.897, short of
    the paper's 0.9539. The alpha ablations sweep one beam point per config
    with num_leaves == beam_size; this does the same.
    """
    text = MERGE.read_text(encoding="utf-8")
    points = re.search(r"SEARCH_POINTS=\(([^)]*)\)", text).group(1).split()

    assert [int(p) for p in points] == sorted(int(p) for p in points), "points must ascend"
    assert max(int(p) for p in points) >= 300, "must reach beyond the M=100 ceiling"
    assert 'BEAM_SIZE="$POINT"' in text
    assert 'NUM_LEAVES="$POINT"' in text


def test_beam_points_never_violate_the_a3_constraint(tmp_path: Path) -> None:
    """num_leaves must be <= beam_size or search.py raises at config load."""
    for point in ("10", "300"):
        cfg = _config(tmp_path, NUM_TREES="4", BEAM_SIZE=point, NUM_LEAVES=point)
        assert max(cfg["evaluation"]["num_leaves"]) <= cfg["evaluation"]["beam_size"]


def test_deep100m_jobs_ask_for_ssd_scratch() -> None:
    """numpy_cpu rerank reads scattered rows from a 38 GB memmap.

    The scratch medium is on the critical path; every ablation merge-search job
    already asks for scratch_ssd.
    """
    for path in DIR.glob("metacentrum_*.sh"):
        # The resource line only — prose may legitimately mention scratch_local
        # while explaining why it is not used.
        select = re.search(r"^#PBS -l select=.*$", path.read_text(encoding="utf-8"), re.M)
        assert select is not None, path.name
        assert "scratch_ssd" in select.group(0), path.name
        assert "scratch_local" not in select.group(0), path.name


def test_each_beam_point_is_copied_out_before_the_next_runs() -> None:
    """A failure at beam 300 must not discard the points that already ran."""
    text = MERGE.read_text(encoding="utf-8")
    loop = text[text.index("for POINT in") : text.index('echo "Done at')]

    assert 'cp -r "$RESULT_DIR/." "$STORAGE_DIR/"' in loop
