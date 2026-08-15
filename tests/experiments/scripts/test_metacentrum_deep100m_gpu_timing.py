import re
from pathlib import Path

SCRIPT = Path("experiments/scripts/deep/100m/metacentrum_deep1b_100m.sh")
SMOKE_SCRIPT = Path("experiments/scripts/deep/100m/metacentrum_deep1b_100m_one_cycle_smoke.sh")
PARALLEL_TREE_SCRIPTS = [
    Path(f"experiments/scripts/deep/100m/metacentrum_deep1b_100m_tree_{idx}.sh") for idx in range(4)
]
MERGE_SEARCH_SCRIPT = Path("experiments/scripts/deep/100m/metacentrum_deep1b_100m_merge_search.sh")
TOP1_BUCKET_SCRIPT = Path(
    "experiments/scripts/deep/100m/metacentrum_deep1b_100m_single_tree_top1_bucket.sh"
)
ABLATION_MERGE_SEARCH_DIR = Path("experiments/scripts/deep/100m/ablation_merge_search")
ABLATION_MERGE_SEARCH_COMMON = ABLATION_MERGE_SEARCH_DIR / "run_ablation_merge_search_common.sh"


def test_deep100m_gpu_timing_script_provisions_enough_host_memory() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "mem=100gb" not in text
    assert "scratch_local=200gb" in text
    assert "ngpus=1" in text


def test_deep100m_gpu_timing_script_targets_hopper_compatible_arches() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    # cuda120 was a typo in earlier revisions; reject it explicitly. The env
    # name (batl-gpu-cu128) is asserted below to confirm we're on the
    # March-2026 FAISS-GPU build that ships SM 9.0 kernels for Hopper.
    assert "cuda120" not in text


def test_setup_batl_gpu_env_helper_installs_conda_forge_faiss_gpu() -> None:
    helper = SCRIPT.parent / "setup_batl_gpu_env.sh"
    text = helper.read_text(encoding="utf-8")

    assert 'mamba install -y -p "${ENV_PREFIX}" -c conda-forge "faiss-gpu>=1.10.0"' in text
    assert "mamba remove" in text  # strips any prior faiss install
    assert "uninstall -y faiss" in text  # also cleans pip-installed faiss
    assert "StandardGpuResources" in text  # smoke verification


def test_deep100m_gpu_timing_script_uses_bounded_tree_update_batch() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--batch-tree-update 32768" in text
    assert "--batch-tree-update auto" not in text


def test_deep100m_gpu_timing_script_uses_deep100m_inputs_and_gpu_stack() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "deep100M_base.fbin" in text
    assert "deep100M_groundtruth.ivecs" in text
    assert "/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/bin/python" in text
    assert "neighbor_search_backend: faiss_gpu" in text
    assert "rerank_backend: numpy_cpu" in text
    assert "subset_size: 100000000" in text
    assert "build.py" in text
    assert "search.py" in text


def test_deep100m_one_cycle_smoke_uses_single_tree_single_cycle_settings() -> None:
    text = SMOKE_SCRIPT.read_text(encoding="utf-8")

    assert "#PBS -N batl_deep100m_one_cycle_smoke" in text
    # Smoke bumps to 300gb to dodge the level-1 OOM that the main script hits
    # at 200gb.
    assert "mem=300gb" in text
    assert "scratch_local=200gb" in text
    assert "deep100M_base.fbin" in text
    assert "deep100M_groundtruth.ivecs" in text
    assert "subset_size: 100000000" in text
    assert "num_trees: 1" in text
    assert "max_alternating_cycles: 1" in text
    assert "convergence_patience: 0" in text
    assert "tree_update_cache_embeddings: false" in text
    assert "neighbor_search_backend: faiss_gpu" in text
    assert "rerank_backend: numpy_cpu" in text
    assert "--batch-tree-update 32768" in text
    assert "--n-queries 100" in text


def test_deep100m_parallel_tree_scripts_build_exactly_one_seeded_tree() -> None:
    for idx, script in enumerate(PARALLEL_TREE_SCRIPTS):
        text = script.read_text(encoding="utf-8")

        assert f"#PBS -N batl_deep100m_tree_{idx}" in text
        assert f"TREE_INDEX={idx}" in text
        assert '--tree-index "$TREE_INDEX"' in text
        assert "num_trees: 4" in text
        assert "index_confidence.pkl" in text
        assert "index_confidence_tree_${TREE_INDEX}.pkl" in text
        assert "batl_deep100m_full_t4_parallel_trees" in text
        assert "search.py" not in text


def test_deep100m_merge_search_script_merges_four_tree_indexes_and_searches() -> None:
    text = MERGE_SEARCH_SCRIPT.read_text(encoding="utf-8")

    assert "#PBS -N batl_deep100m_merge_search" in text
    assert "merge_index.py" in text
    assert '--output "$MERGED_INDEX_PATH"' in text
    for idx in range(4):
        assert f"index_confidence_tree_{idx}.pkl" in text
    assert "search.py" in text
    assert 'CACHED_INDEX="$MERGED_INDEX_PATH"' in text
    assert '--index-path "$CACHED_INDEX"' in text
    assert "--batch-search 2000" in text
    assert "num_trees: 4" in text
    # The sweep exists to measure the resident reranker, so it must request a
    # GPU that fits the 36.1 GiB base plus the 2 GiB capacity-check headroom,
    # and must not pin the CPU backend. Asserted as a floor rather than an
    # exact value so the request can be tuned to whatever queue is available.
    requested_gb = int(re.search(r"gpu_mem=(\d+)gb", text).group(1))
    assert requested_gb * 1000**3 / 1024**3 >= 36.14 + 2.0, (
        f"gpu_mem={requested_gb}gb cannot hold the resident Deep100M base"
    )
    assert "rerank_backend: auto" in text
    # Stage timings are the point of this run; without them it only reports
    # whether QPS moved, not where the remaining time goes.
    assert "performance_profile: true" in text
    assert "search_repetitions: 3" in text


def test_deep100m_single_tree_top1_bucket_script_builds_then_searches_one_leaf() -> None:
    text = TOP1_BUCKET_SCRIPT.read_text(encoding="utf-8")

    assert "#PBS -N batl_deep100m_t1_top1_bucket" in text
    assert "deep100M_base.fbin" in text
    assert "deep100M_groundtruth.ivecs" in text
    assert "subset_size: 100000000" in text
    assert "num_trees: 1" in text
    assert "max_alternating_cycles" not in text
    assert "convergence_patience: 2" in text
    assert "neighbor_search_subset: 1000000" in text
    assert "neighbor_search_backend: faiss_gpu" in text
    assert "rerank_backend: numpy_cpu" in text
    assert "--batch-tree-update 32768" in text
    assert "build.py" in text
    assert "search.py" in text
    assert "--cycle-diagnostics" in text
    assert "--cycle-diagnostics-queries 10000" in text
    assert "--cycle-diagnostics-loss-pairs 100000" in text
    assert "num_leaves: [1]" in text
    assert "--num-leaves 1" in text
    assert "--batch-search 50" in text


def test_deep100m_ablation_merge_search_wrappers_cover_expected_parameters() -> None:
    expected = {
        "metacentrum_deep100m_alpha_2_merge_search.sh": (
            'ABLATION_NAME="deep100m_ablation_alpha_2"',
            "BRANCHING_FACTOR=256",
            "EMBED_DIM=256",
            "ALPHA=2.0",
            "SEARCH_POINTS=(50 60 70 80 100 120 150 180 220 260 300)",
        ),
        "metacentrum_deep100m_alpha_3_merge_search.sh": (
            'ABLATION_NAME="deep100m_ablation_alpha_3"',
            "BRANCHING_FACTOR=256",
            "EMBED_DIM=256",
            "ALPHA=3.0",
            "SEARCH_POINTS=(50 60 70 80 100 120 150 180 220 260 300)",
        ),
        "metacentrum_deep100m_alpha_4_merge_search.sh": (
            'ABLATION_NAME="deep100m_ablation_alpha_4"',
            "BRANCHING_FACTOR=256",
            "EMBED_DIM=256",
            "ALPHA=4.0",
            "SEARCH_POINTS=(50 60 70 80 100 120 150 180 220 260 300)",
        ),
        "metacentrum_deep100m_k64_merge_search.sh": (
            'ABLATION_NAME="deep100m_ablation_k64"',
            "BRANCHING_FACTOR=64",
            "EMBED_DIM=256",
            "ALPHA=1.0",
            "SEARCH_POINTS=(30 40 50 60 80 100 130 160 200 260 300)",
        ),
        "metacentrum_deep100m_k128_merge_search.sh": (
            'ABLATION_NAME="deep100m_ablation_k128"',
            "BRANCHING_FACTOR=128",
            "EMBED_DIM=256",
            "ALPHA=1.0",
            "SEARCH_POINTS=(30 40 50 60 80 100 130 160 200 260 300)",
        ),
        "metacentrum_deep100m_dim64_merge_search.sh": (
            'ABLATION_NAME="deep100m_ablation_embed_dim_64"',
            "BRANCHING_FACTOR=256",
            "EMBED_DIM=64",
            "ALPHA=1.0",
            "SEARCH_POINTS=(2 3 4 5 6 8 10 15 20 30 40)",
        ),
        "metacentrum_deep100m_dim128_merge_search.sh": (
            'ABLATION_NAME="deep100m_ablation_embed_dim_128"',
            "BRANCHING_FACTOR=256",
            "EMBED_DIM=128",
            "ALPHA=1.0",
            "SEARCH_POINTS=(2 3 4 5 6 8 10 15 20 30 40)",
        ),
        "metacentrum_deep100m_dim512_merge_search.sh": (
            'ABLATION_NAME="deep100m_ablation_embed_dim_512"',
            "BRANCHING_FACTOR=256",
            "EMBED_DIM=512",
            "ALPHA=1.0",
            "SEARCH_POINTS=(2 3 4 5 6 8 10 15 20 30 40)",
        ),
    }

    for script_name, required_snippets in expected.items():
        text = (ABLATION_MERGE_SEARCH_DIR / script_name).read_text(encoding="utf-8")

        assert (
            "source /auto/brno2/home/jozefsprlak/repos/batl2/experiments/scripts/deep/100m/ablation_merge_search/run_ablation_merge_search_common.sh"
            in text
        )
        for snippet in required_snippets:
            assert snippet in text


def test_deep100m_ablation_merge_search_common_uses_matched_beam_and_leaf_points() -> None:
    text = ABLATION_MERGE_SEARCH_COMMON.read_text(encoding="utf-8")

    assert 'for POINT in "${SEARCH_POINTS[@]}"' in text
    assert "beam_size: ${point}" in text
    assert "num_leaves: [${point}]" in text
    assert "rerank_backend: numpy_cpu" in text
    assert '--num-leaves "$POINT"' in text
    assert "--batch-search 0" in text
    for idx in range(4):
        assert f"index_confidence_tree_{idx}.pkl" in text
