set -euo pipefail

: "${TREE_INDEX:?The wrapper must set TREE_INDEX.}"
: "${JOB_RELATIVE:?The wrapper must set JOB_RELATIVE.}"
if [[ ! "$TREE_INDEX" =~ ^[0-3]$ ]]; then
    echo "TREE_INDEX must be one of 0, 1, 2, or 3; got $TREE_INDEX." >&2
    exit 2
fi

RESULT_NAME="laion_clip768v2_100m_gpu_tree_${TREE_INDEX}_k256"
STORAGE_TREE_DIR="/storage/brno2/home/jozefsprlak/results/batl_laion_clip768v2_100m_gpu_parallel_trees"
SOURCE_REPO="/auto/brno2/home/jozefsprlak/repos/batl2"
SOURCE_DATA="/storage/brno2/home/jozefsprlak/repos/data/laion5b"
ENV_PREFIX="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128"
CONFIG_RELATIVE="experiments/configs/laion5b/laion5b_100m_clip768v2_lmi.yaml"
COMMON_RELATIVE="experiments/scripts/laion5b/run_laion_clip768v2_100m_gpu_tree_common.sh"
PREPARED_NAME="laion2B-en-clip768v2-n=100M-f32.npy"
MANIFEST_NAME="laion2B-en-clip768v2-n=100M-f32.manifest.json"
SOURCE_MD5="9d8ee3347b1edf136b3ef38162ac05c3"
QUERY_NAME="public-queries-10k-clip768v2.h5"
QUERY_MD5="257b9eb3f7f25776e0d33b22451b7b32"
GROUND_TRUTH_NAME="laion2B-en-public-gold-standard-v2-100M.h5"
GROUND_TRUTH_MD5="35de58992c6446c85c56e710b144c90c"
EXPECTED_ROWS=102144212
EXPECTED_DIM=768

: "${SCRATCHDIR:?PBS did not provide SCRATCHDIR}"
trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

EXPECTED_GPU_NAME="NVIDIA L40S"
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
ACTUAL_GPU_NAME="${GPU_NAMES[0]:-unknown}"
echo "Job ${PBS_JOBID:-unknown} on $(hostname -f): GPU=$ACTUAL_GPU_NAME"
if [ "$ACTUAL_GPU_NAME" != "$EXPECTED_GPU_NAME" ]; then
    echo "Expected $EXPECTED_GPU_NAME for hardware-comparable runs, got $ACTUAL_GPU_NAME." >&2
    exit 2
fi

for required in \
    "$SOURCE_DATA/$PREPARED_NAME" \
    "$SOURCE_DATA/$MANIFEST_NAME" \
    "$SOURCE_DATA/$QUERY_NAME" \
    "$SOURCE_DATA/$GROUND_TRUTH_NAME"; do
    if [ ! -f "$required" ]; then
        echo "Required prepared/SISAP file is missing: $required" >&2
        exit 2
    fi
done

echo "Copying BATL checkout..."
cp -r "$SOURCE_REPO" ./BATL
module load mambaforge
mamba activate "$ENV_PREFIX"
PYTHON_EXEC="$ENV_PREFIX/bin/python"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTHONPATH="$SCRATCHDIR/BATL"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "Verifying persistent memmap before its 292 GiB scratch copy..."
"$PYTHON_EXEC" -u -m experiments.utils.prepare_laion_memmap verify \
    --output "$SOURCE_DATA/$PREPARED_NAME" \
    --manifest "$SOURCE_DATA/$MANIFEST_NAME" \
    --expected-source-md5 "$SOURCE_MD5" \
    --expected-shape "$EXPECTED_ROWS" "$EXPECTED_DIM"

DATA_DIR="$SCRATCHDIR/BATL/data/laion5b"
mkdir -p "$DATA_DIR" "$STORAGE_TREE_DIR"
cp "$SOURCE_DATA/$PREPARED_NAME" "$DATA_DIR/$PREPARED_NAME"
cp "$SOURCE_DATA/$MANIFEST_NAME" "$DATA_DIR/data_manifest.json"
cp "$SOURCE_DATA/$QUERY_NAME" "$DATA_DIR/$QUERY_NAME"
cp "$SOURCE_DATA/$GROUND_TRUTH_NAME" "$DATA_DIR/$GROUND_TRUTH_NAME"

echo "Verifying local memmap SHA-256 after staging..."
"$PYTHON_EXEC" -u -m experiments.utils.prepare_laion_memmap verify \
    --output "$DATA_DIR/$PREPARED_NAME" \
    --manifest "$DATA_DIR/data_manifest.json" \
    --expected-source-md5 "$SOURCE_MD5" \
    --expected-shape "$EXPECTED_ROWS" "$EXPECTED_DIM"

require_md5() {
    local path="$1"
    local expected="$2"
    local digest_output
    local actual
    digest_output=$(md5sum "$path")
    actual="${digest_output%% *}"
    echo "$path: md5=$actual"
    if [ "$actual" != "$expected" ]; then
        echo "$path: expected md5 $expected, got $actual." >&2
        return 2
    fi
}
require_md5 "$DATA_DIR/$QUERY_NAME" "$QUERY_MD5"
require_md5 "$DATA_DIR/$GROUND_TRUTH_NAME" "$GROUND_TRUTH_MD5"

cd "$SCRATCHDIR/BATL"
CONFIG_PATH="$SCRATCHDIR/BATL/$CONFIG_RELATIVE"
echo "Checking config, memmap layout, ground truth, CUDA, and FAISS-GPU..."
"$PYTHON_EXEC" -u - "$CONFIG_PATH" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import faiss
import h5py
import numpy as np
import torch
from batl.utils.config_parsing import load_experiment_config

cfg = load_experiment_config(sys.argv[1])
expected_shape = (102_144_212, 768)
if cfg.subset_size != expected_shape[0]:
    raise SystemExit(f"Expected subset_size={expected_shape[0]}, got {cfg.subset_size}.")
if cfg.model.embedding_dim != expected_shape[1]:
    raise SystemExit(
        f"Expected embedding_dim={expected_shape[1]}, got {cfg.model.embedding_dim}."
    )
if cfg.train.neighbor_search_subset != 1_021_443:
    raise SystemExit(
        "Expected the exact 1% label-search subset of 1,021,443 vectors."
    )
if cfg.dataset_storage_mode != "memmap":
    raise SystemExit("The 100M run requires dataset.storage_mode=memmap.")

database = np.load(Path(cfg.dataset_base_path or ""), mmap_mode="r", allow_pickle=False)
print(f"database shape={database.shape} dtype={database.dtype} type={type(database).__name__}")
if not isinstance(database, np.memmap):
    raise SystemExit("Prepared database did not open as a NumPy memmap.")
if tuple(database.shape) != expected_shape:
    raise SystemExit(f"Expected database shape {expected_shape}, got {database.shape}.")
if database.dtype != np.dtype(np.float32):
    raise SystemExit(f"Expected float32 database, got {database.dtype}.")

query_path = Path(cfg.dataset_query_path or "")
with h5py.File(query_path, "r") as handle:
    key = "emb" if "emb" in handle else next(iter(handle.keys()))
    queries = handle[key]
    print(f"queries shape={queries.shape} dtype={queries.dtype}")
    if queries.ndim != 2 or tuple(queries.shape) != (cfg.num_queries, expected_shape[1]):
        raise SystemExit(f"Unexpected query shape: {queries.shape}.")
    if not np.issubdtype(queries.dtype, np.floating):
        raise SystemExit("Queries must use a floating-point dtype.")

ground_truth_path = Path(cfg.dataset_ground_truth_path or "")
with h5py.File(ground_truth_path, "r") as handle:
    if "knns" not in handle:
        raise SystemExit(f"{ground_truth_path}: missing required 'knns' dataset.")
    knns = handle["knns"]
    print(f"ground truth shape={knns.shape} dtype={knns.dtype}")
    if knns.ndim != 2 or cfg.num_queries not in knns.shape:
        raise SystemExit("Ground truth does not contain the configured query count.")
    ids = np.asarray(knns)
    min_id = int(ids.min())
    max_id = int(ids.max())
    print(f"ground truth one_based_id_range=[{min_id}, {max_id}]")
    if min_id < 1 or max_id > cfg.subset_size:
        raise SystemExit(
            f"Ground truth ID range [{min_id}, {max_id}] is outside "
            f"database_size=cfg.subset_size ({cfg.subset_size})."
        )

print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible to PyTorch; refusing GPU build.")
print(f"torch.cuda.device_name[0]: {torch.cuda.get_device_name(0)}")
if torch.cuda.get_device_name(0) != "NVIDIA L40S":
    raise SystemExit("PyTorch is not using the required NVIDIA L40S.")
faiss_gpu_available = hasattr(faiss, "StandardGpuResources")
print(f"faiss_gpu_available: {faiss_gpu_available}")
if not faiss_gpu_available:
    raise SystemExit("FAISS-GPU bindings are unavailable; refusing GPU build.")
PY

RESULT_DIR="$SCRATCHDIR/results/$RESULT_NAME"
INDEX_PATH="$RESULT_DIR/index_confidence.pkl"
TREE_INDEX_PATH="$RESULT_DIR/index_confidence_tree_${TREE_INDEX}.pkl"
mkdir -p "$RESULT_DIR"
cp "$DATA_DIR/data_manifest.json" "$RESULT_DIR/data_manifest.json"

BASE_SEED=$("$PYTHON_EXEC" - "$CONFIG_PATH" <<'PY'
import sys

from batl.utils.config_parsing import load_experiment_config

print(load_experiment_config(sys.argv[1]).seed)
PY
)
TREE_SEED=$((BASE_SEED + TREE_INDEX))
if GIT_COMMIT=$(git -C "$SCRATCHDIR/BATL" rev-parse HEAD 2>/dev/null); then
    :
else
    GIT_COMMIT="unknown"
fi
if ! git -C "$SCRATCHDIR/BATL" status --short > "$RESULT_DIR/git_status.txt"; then
    echo "git status unavailable" > "$RESULT_DIR/git_status.txt"
fi

echo "Building BATL 100M tree $TREE_INDEX with seed $TREE_SEED from $GIT_COMMIT..."
nvidia-smi
BUILD_START_EPOCH=$(date +%s)
BUILD_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
set +e
"$PYTHON_EXEC" -u build.py \
    "$CONFIG_PATH" \
    --log \
    --index-path "$INDEX_PATH" \
    --result-dir "$RESULT_DIR" \
    --tree-index "$TREE_INDEX"
RUN_STATUS=$?
set -e
BUILD_END_EPOCH=$(date +%s)
BUILD_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if [ "$RUN_STATUS" -eq 0 ] && [ ! -f "$TREE_INDEX_PATH" ]; then
    echo "Build returned success but did not produce $TREE_INDEX_PATH" >&2
    RUN_STATUS=3
fi

{
    echo "pbs_job_id=${PBS_JOBID:-unknown}"
    echo "hostname=$(hostname -f)"
    echo "gpu_name=$ACTUAL_GPU_NAME"
    echo "git_commit=$GIT_COMMIT"
    echo "tree_index=$TREE_INDEX"
    echo "seed=$TREE_SEED"
    echo "build_start_utc=$BUILD_START_UTC"
    echo "build_end_utc=$BUILD_END_UTC"
    echo "build_command_wall_time_s=$((BUILD_END_EPOCH - BUILD_START_EPOCH))"
    echo "run_status=$RUN_STATUS"
} > "$RESULT_DIR/job_summary.txt"

RESULT_TS=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="$STORAGE_TREE_DIR/tree_${TREE_INDEX}_$RESULT_TS"
mkdir -p "$OUTPUT_DIR"
echo "Copying build artifacts to $OUTPUT_DIR"
cp -r "$RESULT_DIR/." "$OUTPUT_DIR/"
cp "$CONFIG_PATH" "$OUTPUT_DIR/submitted_config.yaml"
cp "$SCRATCHDIR/BATL/$JOB_RELATIVE" "$OUTPUT_DIR/submitted_job.sh"
cp "$SCRATCHDIR/BATL/$COMMON_RELATIVE" "$OUTPUT_DIR/submitted_common.sh"

if [ "$RUN_STATUS" -ne 0 ]; then
    echo "BATL build failed with status $RUN_STATUS; partial artifacts were preserved." >&2
fi
exit "$RUN_STATUS"
