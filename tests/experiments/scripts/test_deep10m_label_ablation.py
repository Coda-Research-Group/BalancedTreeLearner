from pathlib import Path

import yaml

DIR = Path("experiments/scripts/deep/10m/label_ablation")
COMMON = DIR / "run_deep10m_label_ablation_common.sh"
ARMS = {
    "metacentrum_deep10m_label_subset_1pct.sh": ("subset_1pct", "100000"),
    "metacentrum_deep10m_label_exact.sh": ("exact", "10000000"),
}


def _config_for(arm: str, subset: str) -> dict:
    body = COMMON.read_text(encoding="utf-8").split('cat > "$CONFIG_PATH" <<EOF\n', 1)[1]
    body = body.split("\nEOF", 1)[0]
    body = (
        body.replace("$SCRATCHDIR", "/scratch")
        .replace("${RESULT_NAME}", f"deep10m_label_{arm}")
        .replace("${NEIGHBOR_SEARCH_SUBSET}", subset)
    )
    return yaml.safe_load(body)


def test_each_arm_sets_its_own_mining_subset() -> None:
    for script, (arm, subset) in ARMS.items():
        text = (DIR / script).read_text(encoding="utf-8")
        assert f'ARM_NAME="{arm}"' in text
        assert f"NEIGHBOR_SEARCH_SUBSET={subset}" in text
        assert "run_deep10m_label_ablation_common.sh" in text


def test_arms_differ_in_exactly_one_config_value() -> None:
    """The A/B is only interpretable if nothing else moves between arms."""
    baseline = _config_for("x", "100000")
    treatment = _config_for("x", "10000000")

    differing = [
        (section, key)
        for section in baseline
        for key in baseline[section]
        if baseline[section][key] != treatment[section][key]
    ]

    assert differing == [("training", "neighbor_search_subset")]


def test_ablation_config_is_single_tree_and_sweeps_recall_per_bucket() -> None:
    cfg = _config_for("exact", "10000000")

    # One tree: this measures label quality, not ensemble behaviour.
    assert cfg["model"]["num_trees"] == 1
    assert cfg["dataset"]["subset_size"] == 10_000_000
    assert cfg["training"]["top_k_neighbors"] == 100
    # A curve, not a point — recall per bucket is the quantity under test.
    assert cfg["evaluation"]["num_leaves"] == [10, 40, 80, 100, 150, 200]
    # Profiling perturbs timings and this A/B is about recall.
    assert cfg["evaluation"]["performance_profile"] is False


def test_both_arms_carry_the_faiss_libstdcxx_fix_and_a_preflight() -> None:
    text = COMMON.read_text(encoding="utf-8")

    assert "export LD_LIBRARY_PATH=" in text
    assert "refusing label-ablation run" in text
    # The chosen-rank histogram rides along on the same build.
    assert "--cycle-diagnostics" in text


def test_exact_arm_requests_enough_vram_for_the_full_flat_index() -> None:
    """10M x 96 float32 = 3.6 GiB of FAISS index must fit beside the model."""
    text = (DIR / "metacentrum_deep10m_label_exact.sh").read_text(encoding="utf-8")

    assert "gpu_mem=24gb" in text
