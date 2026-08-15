import re
from pathlib import Path

import yaml

DIR = Path("experiments/scripts/deep/100m/label_ablation")
CONFIG_WRITER = DIR / "deep100m_label_config.sh"
BUILD_COMMON = DIR / "run_deep100m_label_build_common.sh"
SEARCH_COMMON = DIR / "run_deep100m_label_search_common.sh"
ARMS = {"subset_1pct": "1000000", "subset_10m": "10000000", "exact": "100000000"}


def _config_for(subset: str) -> dict:
    body = CONFIG_WRITER.read_text(encoding="utf-8").split('cat > "$CONFIG_PATH" <<EOF\n', 1)[1]
    body = body.split("\nEOF", 1)[0]
    body = (
        body.replace("$SCRATCHDIR", "/scratch")
        .replace("${RESULT_NAME}", "deep100m_label_arm")
        .replace("${NEIGHBOR_SEARCH_SUBSET}", subset)
    )
    # Knobs the label arms leave unset resolve to their `${VAR:-default}`
    # default, which is what an arm that does not export them actually gets.
    body = re.sub(r"\$\{[A-Z_]+:-([^}]*)\}", r"\1", body)
    return yaml.safe_load(body)


def test_every_arm_has_a_build_and_a_search_wrapper() -> None:
    for arm, subset in ARMS.items():
        for phase in ("build", "search"):
            script = DIR / f"metacentrum_deep100m_label_{phase}_{arm}.sh"
            text = script.read_text(encoding="utf-8")
            assert f'ARM_NAME="{arm}"' in text
            assert f"NEIGHBOR_SEARCH_SUBSET={subset}" in text
            assert f"run_deep100m_label_{phase}_common.sh" in text


def test_both_phases_generate_the_config_from_one_source() -> None:
    """Build and search must not be able to drift into different configs."""
    for common in (BUILD_COMMON, SEARCH_COMMON):
        text = common.read_text(encoding="utf-8")
        assert "deep100m_label_config.sh" in text
        assert "write_label_config" in text
        # Neither phase may write its own config inline.
        assert 'cat > "$CONFIG_PATH" <<EOF' not in text


def test_build_persists_the_index_and_search_refuses_without_it() -> None:
    build = BUILD_COMMON.read_text(encoding="utf-8")
    search = SEARCH_COMMON.read_text(encoding="utf-8")

    # A search failure must never cost the build.
    assert "search.py" not in build
    assert "build.py" not in search
    assert 'cp "$INDEX_PATH" "$STORAGE_DIR/index_confidence.pkl"' in build
    assert "Missing index from the build job" in search
    assert "exit 2" in search


def test_arms_differ_in_exactly_one_config_value() -> None:
    configs = [_config_for(subset) for subset in ARMS.values()]
    baseline = configs[0]
    for other in configs[1:]:
        differing = [
            (section, key)
            for section in baseline
            for key in baseline[section]
            if baseline[section][key] != other[section][key]
        ]
        assert differing == [("training", "neighbor_search_subset")]


def test_sweep_never_exceeds_beam_size() -> None:
    """code-review A3: M > beam_size is silently clamped and fabricates rows."""
    cfg = _config_for("1000000")

    assert max(cfg["evaluation"]["num_leaves"]) <= cfg["evaluation"]["beam_size"]


def test_exact_build_provisions_for_chunked_mining_and_remining() -> None:
    """The exact arm must size the card for one chunk, not for the whole index.

    Provisioning 44gb for the full 35.76 GiB flat index is what job 22710911
    died on: FAISS grows its device buffer geometrically and copies, so the
    peak exceeds the resident size and no single card here is large enough.
    Mining now searches chunk by chunk, so the requirement is
    `chunk_size * dim * 4` and the arm belongs on the same queue as the others.
    """
    text = (DIR / "metacentrum_deep100m_label_build_exact.sh").read_text(encoding="utf-8")

    requested_gb = int(re.search(r"gpu_mem=(\d+)gb", text).group(1))
    chunk_size = int(re.search(r"NEIGHBOR_SEARCH_CHUNK_SIZE=(\d+)", text).group(1))
    chunk_gib = chunk_size * 96 * 4 / 1024**3
    assert chunk_gib < requested_gb * 1000**3 / 1024**3
    # A chunk that spans the whole database would defeat the point.
    assert chunk_size < 100_000_000
    walltime_h = int(re.search(r"walltime=(\d+):", text).group(1))
    assert walltime_h >= 24


def test_search_phase_stays_on_the_small_gpu_queue() -> None:
    """Rerank is pinned to numpy_cpu, so search never needs a large card."""
    for arm in ARMS:
        text = (DIR / f"metacentrum_deep100m_label_search_{arm}.sh").read_text(encoding="utf-8")
        assert "gpu_mem=16gb" in text
        assert "walltime=4:00:00" in text


def test_config_is_single_tree_with_pinned_rerank() -> None:
    cfg = _config_for("1000000")

    assert cfg["model"]["num_trees"] == 1
    assert cfg["dataset"]["subset_size"] == 100_000_000
    assert cfg["evaluation"]["rerank_backend"] == "numpy_cpu"


def test_build_and_search_wrappers_agree_on_every_ablation_variable() -> None:
    """Sourcing one writer is not enough — the inputs must match too.

    The writer is shared, but each phase supplies its own environment, so a
    variable set in the build wrapper and omitted in the search wrapper still
    produces two different configs. That is how the exact arm briefly recorded
    chunk_size 10M for the build and 1M for the search.
    """
    pattern = re.compile(r"^(NEIGHBOR_SEARCH_[A-Z_]+)=(\S+)", re.MULTILINE)
    for arm in ARMS:
        settings = {}
        for phase in ("build", "search"):
            text = (DIR / f"metacentrum_deep100m_label_{phase}_{arm}.sh").read_text(
                encoding="utf-8"
            )
            settings[phase] = dict(pattern.findall(text))
        assert settings["build"] == settings["search"], (
            f"{arm}: build sets {settings['build']}, search sets {settings['search']}"
        )
