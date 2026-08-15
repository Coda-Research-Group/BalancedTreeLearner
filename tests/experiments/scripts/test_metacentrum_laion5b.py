from pathlib import Path

import yaml

SCRIPT_10M = Path("experiments/scripts/laion5b/metacentrum_laion_10m_gpu_timing.sh")
CLIP768V2_10M_CONFIG = Path("experiments/configs/laion5b/laion5b_10m_clip768v2_lmi.yaml")
CLIP768V2_10M_TREE0 = Path(
    "experiments/scripts/laion5b/metacentrum_laion_clip768v2_10m_gpu_tree_0.sh"
)
CLIP768V2_10M_TREE_JOBS = tuple(
    Path(f"experiments/scripts/laion5b/metacentrum_laion_clip768v2_10m_gpu_tree_{tree_index}.sh")
    for tree_index in range(4)
)
CLIP768V2_10M_TREE_COMMON = Path(
    "experiments/scripts/laion5b/run_laion_clip768v2_10m_gpu_tree_common.sh"
)
CLIP768V2_10M_MERGE_SEARCH = Path(
    "experiments/scripts/laion5b/metacentrum_laion_clip768v2_10m_gpu_merge_search.sh"
)
CLIP768V2_10M_BEAM300_CONFIG = Path(
    "experiments/configs/laion5b/laion5b_10m_clip768v2_lmi_beam300.yaml"
)
CLIP768V2_10M_BEAM300_SEARCH = Path(
    "experiments/scripts/laion5b/metacentrum_laion_clip768v2_10m_gpu_beam300_search.sh"
)
CLIP768V2_100M_CONFIG = Path("experiments/configs/laion5b/laion5b_100m_clip768v2_lmi.yaml")
CLIP768V2_100M_BEAM300_CONFIG = Path(
    "experiments/configs/laion5b/laion5b_100m_clip768v2_lmi_beam300.yaml"
)
CLIP768V2_100M_PREPARE = Path(
    "experiments/scripts/laion5b/metacentrum_laion_clip768v2_100m_prepare_memmap.sh"
)
CLIP768V2_100M_TREE_COMMON = Path(
    "experiments/scripts/laion5b/run_laion_clip768v2_100m_gpu_tree_common.sh"
)
CLIP768V2_100M_TREE_JOBS = tuple(
    Path(f"experiments/scripts/laion5b/metacentrum_laion_clip768v2_100m_gpu_tree_{tree_index}.sh")
    for tree_index in range(4)
)
CLIP768V2_100M_BEAM300_SEARCH = Path(
    "experiments/scripts/laion5b/metacentrum_laion_clip768v2_100m_gpu_beam300_search.sh"
)


def test_laion_10m_gpu_script_uses_large_dataset_memory_settings() -> None:
    text = SCRIPT_10M.read_text(encoding="utf-8")

    assert "subset_size: 10000000" in text
    assert "neighbor_search_mode: sequential_chunked" in text
    assert "tree_update_cache_embeddings: false" in text
    assert "neighbor_search_backend: faiss_gpu" in text
    assert "rerank_backend: torch_gpu" in text


def test_laion_clip768v2_10m_lmi_config_pins_dataset_and_training_contract() -> None:
    text = CLIP768V2_10M_CONFIG.read_text(encoding="utf-8")

    assert "seed: 42" in text
    assert "laion2B-en-clip768v2-n=10M.h5" in text
    assert "public-queries-10k-clip768v2.h5" in text
    assert "laion2B-en-public-gold-standard-v2-10M.h5" in text
    assert "source_name: LAION-2B English CLIP768v2 (SISAP 2023)" in text
    assert "subset_size: 10120191" in text
    assert "metric: angular" in text
    assert "normalize: false" in text
    assert "storage_mode: auto" in text
    assert "branching_factor: 256" in text
    assert "tree_height: 2" in text
    assert "embedding_dim: 768" in text
    assert "num_trees: 4" in text
    assert "batch_size: 4096" in text
    assert "max_alternating_cycles" not in text
    assert "neighbor_search_subset: 101202" in text
    assert "neighbor_search_mode: sequential_chunked" in text
    assert "neighbor_search_backend: faiss_gpu" in text
    assert "tree_update_cache_embeddings: false" in text
    assert "tree_update_batch_size: auto" in text
    assert "device: cuda" in text
    assert "num_queries: 10000" in text
    assert "rerank_backend: numpy_cpu" in text


def test_laion_clip768v2_10m_tree0_job_is_build_only_and_fail_fast() -> None:
    text = CLIP768V2_10M_TREE0.read_text(encoding="utf-8")

    assert "#PBS -N batl_laion_clip768_10m_tree_0" in text
    assert "ncpus=16" in text
    assert "ngpus=1" in text
    assert "gpu_mem=16gb" in text
    assert "mem=112gb" in text
    assert "scratch_local=50gb" in text
    assert "walltime=02:00:00" in text
    assert "TREE_INDEX=0" in text
    assert "laion2B-en-clip768v2-n=10M.h5" in text
    assert "public-queries-10k-clip768v2.h5" in text
    assert "laion2B-en-public-gold-standard-v2-10M.h5" in text
    assert "c05e4b1d2b2a0c7663ac9767753e25e1" in text
    assert "257b9eb3f7f25776e0d33b22451b7b32" in text
    assert "b68b17693253d95e1fc94c217af25e95" in text
    assert "minimum_rows=cfg.subset_size" in text
    assert "expected_dim=cfg.model.embedding_dim" in text
    assert "selected_rows=cfg.subset_size" in text
    assert "minimum_rows=cfg.num_queries" in text
    assert "expected_query_count=cfg.num_queries" in text
    assert "database_size=cfg.subset_size" in text
    assert "conda/batl-gpu-cu128" in text
    assert "faiss_gpu_available" in text
    assert "experiments/configs/laion5b/laion5b_10m_clip768v2_lmi.yaml" in text
    assert "load_experiment_config" in text
    assert "TREE_SEED=$((BASE_SEED + TREE_INDEX))" in text
    assert 'echo "git_commit=$GIT_COMMIT"' in text
    assert 'echo "seed=$TREE_SEED"' in text
    assert 'git -C "$SCRATCHDIR/BATL" status --short' in text
    assert '--tree-index "$TREE_INDEX"' in text
    assert "index_confidence_tree_${TREE_INDEX}.pkl" in text
    assert "search.py" not in text
    assert "--cycle-diagnostics" not in text
    assert 'cat > "$CONFIG_PATH"' not in text


def test_laion_clip768v2_10m_has_three_matching_remaining_tree_jobs() -> None:
    for tree_index, script in enumerate(CLIP768V2_10M_TREE_JOBS):
        wrapper_text = script.read_text(encoding="utf-8")
        shared_text = (
            "" if tree_index == 0 else CLIP768V2_10M_TREE_COMMON.read_text(encoding="utf-8")
        )
        text = wrapper_text + shared_text

        assert f"#PBS -N batl_laion_clip768_10m_tree_{tree_index}" in wrapper_text
        assert f"TREE_INDEX={tree_index}" in wrapper_text
        assert f"metacentrum_laion_clip768v2_10m_gpu_tree_{tree_index}.sh" in wrapper_text
        assert "ncpus=16" in wrapper_text
        assert "ngpus=1" in wrapper_text
        assert "gpu_mem=16gb" in wrapper_text
        assert "gpu_cap=sm_89" in wrapper_text
        assert 'EXPECTED_GPU_NAME="NVIDIA L40S"' in text
        assert "mem=112gb" in text
        assert "scratch_local=50gb" in text
        assert "walltime=02:00:00" in text
        assert "laion5b_10m_clip768v2_lmi.yaml" in text
        assert "TREE_SEED=$((BASE_SEED + TREE_INDEX))" in text
        assert 'echo "gpu_name=$ACTUAL_GPU_NAME"' in text
        assert '--tree-index "$TREE_INDEX"' in text
        assert "index_confidence_tree_${TREE_INDEX}.pkl" in text
        assert "search.py" not in text
        assert "--cycle-diagnostics" not in text


def test_laion_clip768v2_10m_merge_search_validates_and_searches_exact_t4() -> None:
    text = CLIP768V2_10M_MERGE_SEARCH.read_text(encoding="utf-8")

    assert "#PBS -N batl_laion_clip768_10m_t4_search" in text
    assert "ngpus=1" in text
    assert "gpu_mem=16gb" in text
    assert "gpu_cap=sm_89" in text
    assert 'EXPECTED_GPU_NAME="NVIDIA L40S"' in text
    assert "mem=128gb" in text
    assert "scratch_local=100gb" in text
    assert "walltime=12:00:00" in text
    assert "laion2B-en-clip768v2-n=10M.h5" in text
    assert "public-queries-10k-clip768v2.h5" in text
    assert "laion2B-en-public-gold-standard-v2-10M.h5" in text
    assert "c05e4b1d2b2a0c7663ac9767753e25e1" in text
    assert "257b9eb3f7f25776e0d33b22451b7b32" in text
    assert "b68b17693253d95e1fc94c217af25e95" in text
    assert "resolve_tree_index" in text
    assert "TREE_${tree_index}_INDEX" in text
    assert "exactly one saved index" in text
    assert 'SUMMARY_STATUS=$(summary_value "$SUMMARY_PATH" "run_status")' in text
    assert 'SUMMARY_TREE=$(summary_value "$SUMMARY_PATH" "tree_index")' in text
    assert 'SUMMARY_GPU=$(summary_value "$SUMMARY_PATH" "gpu_name")' in text
    assert 'cmp -s "$REFERENCE_CONFIG" "$TREE_CONFIG"' in text
    assert 'cmp -s "$REFERENCE_CONFIG" "$CONFIG_PATH"' in text
    assert "expected_seed = cfg.seed + tree_index" in text
    assert '"$RESULT_DIR/build_times.tsv"' in text
    assert 'metrics.get("train_time_s")' in text
    assert 'hardware.get("gpu_name")' in text
    assert '"SUM_GPU_WORK\\t-\\t"' in text
    assert '"IDEAL_PARALLEL_MAX\\t-\\t"' in text
    assert "merge_index.py" in text
    assert '"${TREE_PATHS[@]}"' in text
    assert "search.py" in text
    assert '--index-path "$MERGED_INDEX"' in text
    assert "--batch-search 50" in text
    assert "build.py" not in text
    assert 'cat > "$CONFIG_PATH"' not in text


def test_laion_clip768v2_beam300_config_only_changes_search_capacity() -> None:
    build_config = yaml.safe_load(CLIP768V2_10M_CONFIG.read_text(encoding="utf-8"))
    search_config = yaml.safe_load(CLIP768V2_10M_BEAM300_CONFIG.read_text(encoding="utf-8"))

    assert build_config["evaluation"]["beam_size"] == 100
    assert build_config["evaluation"]["num_leaves"] == [4, 10, 20, 40, 80, 100]
    assert search_config["evaluation"]["beam_size"] == 300
    assert search_config["evaluation"]["num_leaves"] == [100, 150, 200, 250, 300]

    for config in (build_config, search_config):
        config["evaluation"].pop("beam_size")
        config["evaluation"].pop("num_leaves")
    assert search_config == build_config


def test_laion_clip768v2_beam300_job_reuses_trees_and_stays_cuda_safe() -> None:
    text = CLIP768V2_10M_BEAM300_SEARCH.read_text(encoding="utf-8")

    assert "#PBS -N batl_laion_clip768_10m_b300" in text
    assert "cluster=fobos" in text
    assert 'EXPECTED_GPU_NAME="NVIDIA L40S"' in text
    assert "laion5b_10m_clip768v2_lmi.yaml" in text
    assert "laion5b_10m_clip768v2_lmi_beam300.yaml" in text
    assert "Only beam_size and num_leaves may differ" in text
    assert "cfg.beam_size != 300" in text
    assert "cfg.num_leaves != [100, 150, 200, 250, 300]" in text
    assert "merge_index.py" in text
    assert '"${TREE_PATHS[@]}"' in text
    assert "search.py" in text
    assert "--batch-search 25" in text
    assert "build.py" not in text


def test_laion_clip768v2_100m_build_config_pins_exact_contract() -> None:
    config = yaml.safe_load(CLIP768V2_100M_CONFIG.read_text(encoding="utf-8"))

    assert config["experiment"]["seed"] == 42
    assert config["dataset"]["base_path"].endswith("laion2B-en-clip768v2-n=100M-f32.npy")
    assert config["dataset"]["query_path"].endswith("public-queries-10k-clip768v2.h5")
    assert config["dataset"]["ground_truth_path"].endswith(
        "laion2B-en-public-gold-standard-v2-100M.h5"
    )
    assert config["dataset"]["subset_size"] == 102_144_212
    assert config["dataset"]["metric"] == "angular"
    assert config["dataset"]["normalize"] is False
    assert config["dataset"]["storage_mode"] == "memmap"
    assert config["model"]["branching_factor"] == 256
    assert config["model"]["tree_height"] == 2
    assert config["model"]["embedding_dim"] == 768
    assert config["model"]["num_trees"] == 4
    assert config["training"]["batch_size"] == 4096
    assert config["training"]["neighbor_search_subset"] == 1_021_443
    assert config["training"]["neighbor_search_mode"] == "sequential_chunked"
    assert config["training"]["neighbor_search_backend"] == "faiss_gpu"
    assert config["training"]["tree_update_cache_embeddings"] is False
    assert config["training"]["tree_update_batch_size"] == "auto"
    assert config["training"]["device"] == "cuda"
    assert config["evaluation"]["beam_size"] == 100
    assert config["evaluation"]["num_leaves"] == [4, 10, 20, 40, 80, 100]
    assert config["evaluation"]["rerank_backend"] == "numpy_cpu"


def test_laion_clip768v2_100m_search_config_only_changes_search_capacity() -> None:
    build_config = yaml.safe_load(CLIP768V2_100M_CONFIG.read_text(encoding="utf-8"))
    search_config = yaml.safe_load(CLIP768V2_100M_BEAM300_CONFIG.read_text(encoding="utf-8"))

    assert search_config["evaluation"]["beam_size"] == 300
    assert search_config["evaluation"]["num_leaves"] == [100, 150, 200, 250, 300]
    for config in (build_config, search_config):
        config["evaluation"].pop("beam_size")
        config["evaluation"].pop("num_leaves")
    assert search_config == build_config


def test_laion_clip768v2_100m_prepare_job_is_atomic_and_reproducible() -> None:
    text = CLIP768V2_100M_PREPARE.read_text(encoding="utf-8")

    assert "#PBS -l select=1:ncpus=4:mem=32gb:scratch_ssd=350gb" in text
    assert "#PBS -l walltime=12:00:00" in text
    assert "ngpus=" not in text
    assert "laion2B-en-clip768v2-n=100M.h5" in text
    assert "laion2B-en-clip768v2-n=100M-f32.npy" in text
    assert "laion2B-en-clip768v2-n=100M-f32.manifest.json" in text
    assert "9d8ee3347b1edf136b3ef38162ac05c3" in text
    assert "257b9eb3f7f25776e0d33b22451b7b32" in text
    assert "35de58992c6446c85c56e710b144c90c" in text
    assert "EXPECTED_ROWS=102144212" in text
    assert "EXPECTED_DIM=768" in text
    assert "prepare_laion_memmap" in text
    assert text.index('cp -r "$SOURCE_REPO" ./BATL') < text.index(
        "A complete conversion already exists"
    )
    assert text.index('export PYTHONPATH="$SCRATCHDIR/BATL"') < text.index(
        "A complete conversion already exists"
    )
    assert '"$PBS_JOB_ID_SAFE"' in text
    assert ".partial.${PBS_JOB_ID_SAFE}" in text
    assert "mv -n" in text
    assert " convert " in text
    assert " verify " in text
    assert 'git -C "$SCRATCHDIR/BATL" rev-parse HEAD' in text
    assert "pbs_job_id" in text


def test_laion_clip768v2_100m_has_four_fobos_tree_wrappers() -> None:
    for tree_index, script in enumerate(CLIP768V2_100M_TREE_JOBS):
        text = script.read_text(encoding="utf-8")

        assert f"#PBS -N batl_laion_clip768_100m_tree_{tree_index}" in text
        assert (
            "select=1:ncpus=16:ngpus=1:gpu_mem=16gb:gpu_cap=sm_89:"
            "cluster=fobos:mem=700gb:scratch_ssd=400gb"
        ) in text
        assert "#PBS -l walltime=24:00:00" in text
        assert f"TREE_INDEX={tree_index}" in text
        assert f"metacentrum_laion_clip768v2_100m_gpu_tree_{tree_index}.sh" in text
        assert "run_laion_clip768v2_100m_gpu_tree_common.sh" in text


def test_laion_clip768v2_100m_tree_runner_validates_and_builds_one_tree() -> None:
    text = CLIP768V2_100M_TREE_COMMON.read_text(encoding="utf-8")

    assert 'EXPECTED_GPU_NAME="NVIDIA L40S"' in text
    assert "laion2B-en-clip768v2-n=100M-f32.npy" in text
    assert "laion2B-en-clip768v2-n=100M-f32.manifest.json" in text
    assert "public-queries-10k-clip768v2.h5" in text
    assert "laion2B-en-public-gold-standard-v2-100M.h5" in text
    assert "257b9eb3f7f25776e0d33b22451b7b32" in text
    assert "35de58992c6446c85c56e710b144c90c" in text
    assert "prepare_laion_memmap verify" in text
    assert 'mmap_mode="r"' in text
    assert "expected_shape = (102_144_212, 768)" in text
    assert "np.dtype(np.float32)" in text
    assert "database_size=cfg.subset_size" in text
    assert "faiss_gpu_available" in text
    assert "laion5b_100m_clip768v2_lmi.yaml" in text
    assert "TREE_SEED=$((BASE_SEED + TREE_INDEX))" in text
    assert 'echo "seed=$TREE_SEED"' in text
    assert 'git -C "$SCRATCHDIR/BATL" status --short' in text
    assert "build.py" in text
    assert '--tree-index "$TREE_INDEX"' in text
    assert "index_confidence_tree_${TREE_INDEX}.pkl" in text
    assert "submitted_config.yaml" in text
    assert "submitted_common.sh" in text
    assert "data_manifest.json" in text
    assert "search.py" not in text


def test_laion_clip768v2_100m_search_job_validates_trees_and_data() -> None:
    text = CLIP768V2_100M_BEAM300_SEARCH.read_text(encoding="utf-8")

    assert "#PBS -N batl_laion_clip768_100m_b300" in text
    assert (
        "select=1:ncpus=16:ngpus=1:gpu_mem=16gb:gpu_cap=sm_89:"
        "cluster=fobos:mem=512gb:scratch_ssd=400gb"
    ) in text
    assert "#PBS -l walltime=24:00:00" in text
    assert 'EXPECTED_GPU_NAME="NVIDIA L40S"' in text
    assert "laion2B-en-clip768v2-n=100M-f32.npy" in text
    assert "laion2B-en-clip768v2-n=100M-f32.manifest.json" in text
    assert "public-queries-10k-clip768v2.h5" in text
    assert "laion2B-en-public-gold-standard-v2-100M.h5" in text
    assert "257b9eb3f7f25776e0d33b22451b7b32" in text
    assert "35de58992c6446c85c56e710b144c90c" in text
    assert "prepare_laion_memmap verify" in text
    assert "laion5b_100m_clip768v2_lmi.yaml" in text
    assert "laion5b_100m_clip768v2_lmi_beam300.yaml" in text
    assert "resolve_tree_index" in text
    assert "TREE_${tree_index}_INDEX" in text
    assert "exactly one saved index" in text
    assert 'SUMMARY_STATUS=$(summary_value "$SUMMARY_PATH" "run_status")' in text
    assert 'SUMMARY_TREE=$(summary_value "$SUMMARY_PATH" "tree_index")' in text
    assert 'SUMMARY_GPU=$(summary_value "$SUMMARY_PATH" "gpu_name")' in text
    assert 'cmp -s "$REFERENCE_CONFIG" "$TREE_CONFIG"' in text
    assert "Only beam_size and num_leaves may differ" in text
    assert "cfg.beam_size != 300" in text
    assert "cfg.num_leaves != [100, 150, 200, 250, 300]" in text
    assert "expected_seed = cfg.seed + tree_index" in text
    assert "merge_index.py" in text
    assert '"${TREE_PATHS[@]}"' in text
    assert "search.py" in text
    assert '--index-path "$MERGED_INDEX"' in text
    assert "--batch-search 25" in text
    assert "tree_build_config.yaml" in text
    assert "data_manifest.json" in text
    assert "build.py" not in text
