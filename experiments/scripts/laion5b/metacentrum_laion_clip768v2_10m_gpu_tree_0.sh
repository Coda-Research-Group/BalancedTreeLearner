#!/bin/bash
#PBS -N batl_laion_clip768_10m_tree_0
# Build-only pilot on the exact SISAP 2023 LAION-2B 10M CLIP768v2 base used
# by the LMI comparison. This builds tree 0 of a possible T=4 BATL ensemble;
# it does not run search and therefore avoids charging query work to build time.
# Tree 0 used 87.5 GiB RAM and 34:22 walltime; keep 28% RAM and 3.5x time headroom.
#PBS -l select=1:ncpus=16:ngpus=1:gpu_mem=16gb:gpu_cap=sm_89:mem=112gb:scratch_local=50gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/laion_clip768_10m_tree_0.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

set -euo pipefail

TREE_INDEX=0
RESULT_NAME="laion_clip768v2_10m_gpu_tree_${TREE_INDEX}_k256"
STORAGE_TREE_DIR="/storage/brno2/home/jozefsprlak/results/batl_laion_clip768v2_10m_gpu_parallel_trees"
SOURCE_REPO="/auto/brno2/home/jozefsprlak/repos/batl2"
SOURCE_DATA="/storage/brno2/home/jozefsprlak/repos/data/laion5b"
ENV_PREFIX="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128"
CONFIG_RELATIVE="experiments/configs/laion5b/laion5b_10m_clip768v2_lmi.yaml"
JOB_RELATIVE="experiments/scripts/laion5b/metacentrum_laion_clip768v2_10m_gpu_tree_0.sh"

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

echo "Copying BATL checkout and exact SISAP 2023 LAION-2B files..."
cp -r "$SOURCE_REPO" ./BATL
DATA_DIR="$SCRATCHDIR/BATL/data/laion5b"
mkdir -p "$DATA_DIR" "$STORAGE_TREE_DIR"

for filename in \
    "laion2B-en-clip768v2-n=10M.h5" \
    "public-queries-10k-clip768v2.h5" \
    "laion2B-en-public-gold-standard-v2-10M.h5"; do
    if [ ! -f "$SOURCE_DATA/$filename" ]; then
        echo "Required dataset file is missing: $SOURCE_DATA/$filename" >&2
        exit 2
    fi
    cp "$SOURCE_DATA/$filename" "$DATA_DIR/$filename"
done

module load mambaforge
mamba activate "$ENV_PREFIX"
PYTHON_EXEC="$ENV_PREFIX/bin/python"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTHONPATH="$SCRATCHDIR/BATL"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "Checking hashes, HDF5 layout, CUDA, and FAISS-GPU before training..."
cd "$SCRATCHDIR/BATL"
CONFIG_PATH="$SCRATCHDIR/BATL/$CONFIG_RELATIVE"
"$PYTHON_EXEC" -u - "$CONFIG_PATH" <<'PY'
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import faiss
import h5py
import numpy as np
import torch
from batl.utils.config_parsing import load_experiment_config


def require_md5(path: Path, expected: str) -> None:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    print(f"{path}: md5={actual}")
    if actual != expected:
        raise SystemExit(f"{path}: expected md5 {expected}, got {actual}.")


def require_float_matrix(
    path: Path,
    *,
    minimum_rows: int,
    expected_dim: int,
    selected_rows: int,
) -> None:
    with h5py.File(path, "r") as handle:
        key = "emb" if "emb" in handle else next(iter(handle.keys()))
        dataset = handle[key]
        print(f"{path}:{key} shape={dataset.shape} dtype={dataset.dtype}")
        if dataset.ndim != 2 or int(dataset.shape[1]) != expected_dim:
            raise SystemExit(
                f"{path}:{key} expected dimension {expected_dim}, got shape {dataset.shape}."
            )
        if int(dataset.shape[0]) < minimum_rows:
            raise SystemExit(
                f"{path}:{key} needs at least {minimum_rows} rows, got {dataset.shape[0]}."
            )
        if selected_rows > int(dataset.shape[0]):
            raise SystemExit(f"{path}:{key} cannot select {selected_rows} rows.")
        if not np.issubdtype(dataset.dtype, np.floating):
            raise SystemExit(f"{path}:{key} must contain floating-point vectors.")
        print(f"{path}:{key} selected_rows={selected_rows}")


def require_ground_truth(path: Path, expected_query_count: int, database_size: int) -> None:
    with h5py.File(path, "r") as handle:
        if "knns" not in handle:
            raise SystemExit(f"{path}: missing required 'knns' dataset.")
        knns = handle["knns"]
        print(f"{path}:knns shape={knns.shape} dtype={knns.dtype}")
        if knns.ndim != 2 or expected_query_count not in knns.shape:
            raise SystemExit(
                f"{path}:knns must be a 2D matrix with {expected_query_count} queries."
            )
        ids = np.asarray(knns)
        min_id = int(ids.min())
        max_id = int(ids.max())
        print(f"{path}:knns one_based_id_range=[{min_id}, {max_id}]")
        if min_id < 1 or max_id > database_size:
            raise SystemExit(f"{path}:knns contains an ID outside the published 10M database.")


cfg = load_experiment_config(sys.argv[1])
if cfg.subset_size is None:
    raise SystemExit("The exact LAION/LMI comparison requires dataset.subset_size.")
base = Path(cfg.dataset_base_path or "")
queries = Path(cfg.dataset_query_path or "")
ground_truth = Path(cfg.dataset_ground_truth_path or "")

# Official SISAP 2023 checksums:
# https://sisap-challenges.github.io/2023/datasets/
require_md5(base, "c05e4b1d2b2a0c7663ac9767753e25e1")
require_md5(queries, "257b9eb3f7f25776e0d33b22451b7b32")
require_md5(ground_truth, "b68b17693253d95e1fc94c217af25e95")
# The official "10M" file has 10,120,191 rows. The LMI SISAP-2023 runner loads
# the complete HDF5 array, so BATL must index every row for compatible recall.
require_float_matrix(
    base,
    minimum_rows=cfg.subset_size,
    expected_dim=cfg.model.embedding_dim,
    selected_rows=cfg.subset_size,
)
require_float_matrix(
    queries,
    minimum_rows=cfg.num_queries,
    expected_dim=cfg.model.embedding_dim,
    selected_rows=cfg.num_queries,
)
require_ground_truth(
    ground_truth,
    expected_query_count=cfg.num_queries,
    database_size=cfg.subset_size,
)

print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible to PyTorch; refusing GPU build.")
print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
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

echo "Building BATL tree $TREE_INDEX with seed $TREE_SEED from commit $GIT_COMMIT..."
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
OUTPUT_DIR="$STORAGE_TREE_DIR/tree_${TREE_INDEX}_${RESULT_TS}"
mkdir -p "$OUTPUT_DIR"
echo "Copying build artifacts to $OUTPUT_DIR"
cp -r "$RESULT_DIR/." "$OUTPUT_DIR/"
cp "$CONFIG_PATH" "$OUTPUT_DIR/submitted_config.yaml"
cp "$SCRATCHDIR/BATL/$JOB_RELATIVE" "$OUTPUT_DIR/submitted_job.sh"

if [ "$RUN_STATUS" -ne 0 ]; then
    echo "BATL build failed with status $RUN_STATUS; partial artifacts were preserved." >&2
fi
exit "$RUN_STATUS"
